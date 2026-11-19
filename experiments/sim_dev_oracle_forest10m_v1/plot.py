"""Render recorded Odom, recorded plans and optional native map occupancy."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from query_minco.integration.native_grid_truth_v1.grid import Grid
from experiments.sim_dev_oracle_forest10m_v1.proof_evidence import observed
from query_minco.integration.yopo_execution_v4.trajectory import Plan


def obstacles(base, xyz, goal):
    grid = Grid(base)
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    if goal is not None:
        lo = np.minimum(lo, goal)
        hi = np.maximum(hi, goal)
    padding = [2., 2., .3]
    first = np.maximum(0, np.floor((lo-padding-grid.origin)/grid.h).astype(int))
    last = np.minimum(grid.shape, np.ceil((hi+padding-grid.origin)/grid.h).astype(int))
    region = tuple(slice(a, b) for a,b in zip(first,last))
    cells = np.argwhere((grid.data[region] > 0).any(axis=2))
    return grid.origin[:2] + (cells + first[:2] + .5)*grid.h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--after-cleanup', action='store_true')
    a = parser.parse_args()
    episode = a.run
    if not (episode/'supervisor_data/EVENTS.jsonl').is_file():
        episode = a.run/'none'
    file = episode/'supervisor_data/EVENTS.jsonl'
    result_file = a.run/'RESULT.json'
    result = json.loads(result_file.read_text()) if result_file.is_file() else {}
    if a.after_cleanup and (result.get('lifecycle') != 'TERMINATED' or
                            result.get('cleanup_status') != 'PASS' or
                            (result.get('terminal') or {}).get('terminal_state') != 'OWNED_PROCESS_TERMINATED'):
        parser.error('run must have recorded Plant termination and completed cleanup')
    rows = [json.loads(s) for s in file.read_text().splitlines() if s.strip()]
    od = []
    for e in rows:
        if e.get('kind') == 'odom' and e.get('accepted'):
            q = e['payload']
            sample = np.asarray([q['p'],q['v']], dtype=float)
            if sample.shape == (2,3) and np.isfinite(sample).all() and np.isfinite(e['stamp']):
                od.append((e['stamp'],sample))
    if not od:
        parser.error('NO_REAL_ODOM')
    t = np.asarray([stamp for stamp,_ in od])
    xyz,vel = np.asarray([sample for _,sample in od]).transpose(1,0,2)
    scene_file = a.run/'SCENE_IDENTITY.json'
    scene = json.loads(scene_file.read_text()).get('scene',{}) if scene_file.is_file() else {}
    goal = scene.get('goal')
    plans, proofs, rejected = {}, [], []
    recorded = set()
    for e in rows:
        q = e.get('payload',{})
        if q.get('event') == 'CANDIDATE_GENERATED':
            identity = q['plan_id']
            recorded.add(identity)
            if identity not in plans:
                try:
                    plan = Plan(q['plan'])
                    if plan.id != identity:
                        raise ValueError('PLAN_ID_MISMATCH')
                    plans[identity] = plan
                except (KeyError, ValueError) as exc:
                    rejected.append(dict(plan_id=identity,reason=str(exc)))
        if q.get('event') == 'ORACLE_PREFIX_COMMITTED':
            proof = q.get('oracle',{})
            if q.get('plan_id') == proof.get('plan_id') and all(k in proof for k in ('issued','expires','prefix')):
                proofs.append(proof)

    a.output.mkdir(parents=True, exist_ok=False)
    fig, ax = plt.subplots()
    grid_base = episode/'native_grid'
    has_map = all(grid_base.with_suffix(s).is_file() for s in ('.json','.u8'))
    if has_map:
        obs = obstacles(grid_base,xyz,goal)
        ax.scatter(obs[:,0],obs[:,1],s=2,color='.6',label='native occupancy near flight height')
    if goal is not None:
        ax.scatter(*goal[:2],color='red',marker='*',label='recorded goal')
    ax.plot(xyz[:,0],xyz[:,1],'.-',color='darkorange',label='actual Odom')
    ax.set(xlabel='x (m)',ylabel='y (m)',aspect='equal')
    ax.legend()
    fig.savefig(a.output/'xy.png',dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(xyz[:,0],xyz[:,2],'.-',color='darkorange')
    ax.set(xlabel='x (m)',ylabel='z (m)')
    fig.savefig(a.output/'xz.png',dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2,1,sharex=True)
    axes[0].plot(t-t[0],xyz[:,0]-xyz[0,0])
    axes[0].set_ylabel('Forward progress (m)')
    axes[1].plot(t-t[0],np.linalg.norm(vel,axis=1))
    axes[1].set(xlabel='ROS source time from start (s)',ylabel='Speed (m/s)')
    fig.savefig(a.output/'progress_speed.png',dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    for i,plan in enumerate(plans.values()):
        q = np.asarray([plan.sample(s)[0] for s in np.linspace(0,plan.d.sum(),81)])
        ax.plot(q[:,0],q[:,1],color='steelblue',alpha=.35,label='recorded candidates' if not i else None)
    drawn = 0
    for proof in proofs:
        plan = plans.get(proof['plan_id'])
        if plan is None:
            continue
        start = max(plan.epoch,proof['issued'])
        end = min(proof['expires'],proof['issued']+proof['prefix'],plan.epoch+plan.d.sum())
        if end <= start:
            continue
        q = np.asarray([plan.at(s)[0] for s in np.linspace(start,end,12)])
        ax.plot(q[:,0],q[:,1],color='green',alpha=.55,label='historical Guardian prefixes' if not drawn else None)
        drawn += 1
    ax.plot(xyz[:,0],xyz[:,1],'.-',color='darkorange',label='actual Odom')
    ax.set(title=f'{len(recorded)} recorded candidates',xlabel='x (m)',ylabel='y (m)',aspect='equal')
    ax.legend()
    fig.savefig(a.output/'plan_actual.png',dpi=150)
    plt.close(fig)

    metrics = dict(input=str(file),run_status=result.get('status'),odom_samples=len(od),
                   recorded_candidates=len(recorded),drawn_candidates=len(plans),
                   rejected_candidates=rejected,drawn_guardian_prefixes=drawn,
                   obstacle_projection=has_map,goal=goal,
                   forward_progress=float(xyz[-1,0]-xyz[0,0]),
                   travel_distance=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()),
                   physical_success_claimed=False)
    evidence=observed(a.run,result)
    metrics['observed_execution']=evidence
    windows=evidence['proof_windows']
    switches=evidence['switches']
    ids=list(dict.fromkeys([w['plan_id'] for w in windows]+[w['plan_id'] for w in switches]))
    lanes={pid:i for i,pid in enumerate(ids)}
    fig,ax=plt.subplots(figsize=(10,max(3,.3*len(ids)+2)))
    for w in windows:
        ax.hlines(lanes[w['plan_id']],w['issued']-t[0],w['expires']-t[0],color='green',alpha=.5,lw=3)
    for w in switches:
        ax.plot(w['ros_time']-t[0],lanes[w['plan_id']],marker='x',color='black')
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([pid[:8] for pid in ids])
    ax.set(xlabel='ROS source time from first Odom (s)',ylabel='Plan ID',
           title='Guardian proof windows (green) and Controller switches (?)')
    fig.tight_layout()
    fig.savefig(a.output/'proof_timeline.png',dpi=150)
    plt.close(fig)
    (a.output/'METRICS.json').write_text(json.dumps(metrics,indent=2)+'\n')


if __name__ == '__main__':
    main()
