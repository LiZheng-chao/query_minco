"""Time-bounded depth history and continuous swept-ball containment.

Native ray depths are NOT continuous pixel-footprint lower bounds. They create
ray observations only, never certified volume. Positive numerical certificates
are explicitly scoped to analytic fixtures or an external empty-room supervisor.
No maps, GT labels, ROS or neural safety scores are accepted here.
"""
from dataclasses import dataclass,field
from enum import Enum
import hashlib
import numpy as np
from query_minco.inference.shadow import K
from query_minco.inference.planner_adapter import InferenceAdapter
from query_minco.evaluation.validation import bernstein_controls,split_controls

class Scope(str,Enum):
    NATIVE_RAYS='NATIVE_RAYS'
    ANALYTIC_FIXTURE='ANALYTIC_FIXTURE'
    EXTERNAL_SUPERVISOR='EXTERNAL_SUPERVISOR'

def readonly(x):
    a=np.array(x,dtype=np.float64,copy=True);a.setflags(write=False);return a

@dataclass(frozen=True)
class ErrorBudget:
    body_radius:float=.12
    margin:float=.03
    tracking:float=.02
    translation:float=.02
    extrinsic:float=.005
    rotation_rad:float=np.deg2rad(.5)
    depth_reserve:float=.15
    numerical:float=1e-6
    def __post_init__(self):
        if not all(np.isfinite(x) and x>=0 for x in self.__dict__.values()):raise ValueError('INVALID_ERROR_BUDGET')
    def radius(self,distance):
        return self.body_radius+self.margin+self.tracking+self.translation+self.extrinsic+2*max(0,distance)*np.sin(self.rotation_rad/2)

@dataclass(frozen=True)
class DepthFrame:
    stamp:float
    depth:np.ndarray
    origin:np.ndarray
    rotation:np.ndarray
    frame_id:str
    def __post_init__(self):
        InferenceAdapter.inputs(np.asarray(self.depth)[None],np.zeros((1,12),np.float32),None)
        R=readonly(self.rotation);p=readonly(self.origin)
        if p.shape!=(3,) or R.shape!=(3,3) or not np.isfinite(p).all() or not np.isfinite(R).all():raise ValueError('POSE')
        if not np.allclose(R.T@R,np.eye(3),atol=1e-7) or abs(np.linalg.det(R)-1)>1e-7:raise ValueError('ROTATION')
        if not np.isfinite(self.stamp) or not self.frame_id:raise ValueError('STAMP_ID')
        d=np.array(self.depth,copy=True);d.setflags(write=False)
        object.__setattr__(self,'rotation',R);object.__setattr__(self,'origin',p);object.__setattr__(self,'depth',d)

class DepthHistory:
    def __init__(self,max_age=.25,max_frames=8,errors=ErrorBudget()):
        if not 0<max_age<=.25 or not 1<=max_frames<=64:raise ValueError('HISTORY_BOUNDS')
        self.max_age=max_age;self.max_frames=max_frames;self.errors=errors;self.frames=[];self.last_stamp=None
    def add(self,frame,intrinsics):
        if not np.array_equal(np.asarray(intrinsics),K):raise ValueError('CALIBRATION_NOT_EXACT')
        if self.last_stamp is not None and frame.stamp<=self.last_stamp:
            self.frames.clear();self.last_stamp=None;raise ValueError('CLOCK_RESET_HISTORY_CLEARED')
        self.last_stamp=frame.stamp;self.frames.append(frame);self.frames=self.frames[-self.max_frames:]
        self.frames=[f for f in self.frames if frame.stamp-f.stamp<=self.max_age]
    def fresh(self,now):
        return [f for f in self.frames if 0<=now-f.stamp<=self.max_age]
    def diagnostics(self,points_W,now):
        """Point ray support only. Does not test or prove any body volume."""
        pts=np.asarray(points_W,float).reshape(-1,3);n=len(pts)
        front=np.zeros(n,bool);occluded=np.zeros(n,bool);inside=np.zeros(n,bool);valid=np.zeros(n,bool)
        frames=self.fresh(now)
        for f in frames:
            p=(pts-f.origin)@f.rotation;x=p[:,0];positive=x>.04
            u=np.zeros(n);v=np.zeros(n);u[positive]=79.5-100*p[positive,1]/x[positive];v[positive]=63.5-100*p[positive,2]/x[positive]
            fov=positive&(u>=0)&(u<=159)&(v>=0)&(v<=127)
            idx=np.flatnonzero(fov);inside|=fov
            ui=np.rint(u[idx]).astype(int);vi=np.rint(v[idx]).astype(int)
            known=f.depth[1,vi,ui]==1;z=f.depth[0,vi,ui]*20
            valid[idx[known]]=True
            front[idx[known&(x[idx]<z-self.errors.depth_reserve)]]=True
            occluded[idx[known&(x[idx]>z+self.errors.depth_reserve)]]=True
        return dict(fresh_frames=len(frames),point_front_fraction=float(front.mean()) if n else 0.,
            point_occluded_fraction=float(occluded.mean()) if n else 0.,point_outside_all_fov_fraction=float((~inside).mean()) if n else 1.,
            point_invalid_or_unobserved_fraction=float((~valid).mean()) if n else 1.,
            continuous_volume_available=False,reason='PIXEL_AND_RAYSTEP_COVERAGE_UNPROVEN' if frames else 'NO_FRESH_DEPTH')
    def volume(self,now):
        return FreeVolume((),Scope.NATIVE_RAYS,self.errors,
            reason='PIXEL_AND_RAYSTEP_COVERAGE_UNPROVEN' if self.fresh(now) else 'NO_FRESH_DEPTH')

@dataclass(frozen=True)
class ConvexCell:
    """Closed bounded free-volume halfspaces A*x <= b, in world coordinates."""
    A:np.ndarray
    b:np.ndarray
    valid_until:float
    evidence_id:str
    anchor:np.ndarray=field(default_factory=lambda:np.zeros(3))
    def __post_init__(self):
        A=readonly(self.A);b=readonly(self.b)
        if A.ndim!=2 or A.shape[1]!=3 or b.shape!=(len(A),) or len(A)<4 or not np.isfinite(A).all() or not np.isfinite(b).all() or (np.linalg.norm(A,axis=1)==0).any():raise ValueError('CELL_HALFSPACES')
        if not np.isfinite(self.valid_until) or not self.evidence_id:raise ValueError('CELL_EVIDENCE')
        anchor=readonly(self.anchor)
        if anchor.shape!=(3,) or not np.isfinite(anchor).all():raise ValueError('CELL_ANCHOR')
        object.__setattr__(self,'A',A);object.__setattr__(self,'b',b);object.__setattr__(self,'anchor',anchor)

def box_cell(lo,hi,valid_until,evidence_id):
    lo=np.asarray(lo,float);hi=np.asarray(hi,float)
    if lo.shape!=(3,) or hi.shape!=(3,) or (lo>=hi).any():raise ValueError('CELL_BOUNDS')
    return ConvexCell(np.r_[np.eye(3),-np.eye(3)],np.r_[hi,-lo],valid_until,evidence_id)

def trajectory_digest(coefficients,durations):
    h=hashlib.sha256()
    for x in [coefficients,durations]:h.update(np.asarray(x,dtype='<f8').tobytes())
    return h.hexdigest()

@dataclass(frozen=True)
class FreeVolume:
    cells:tuple
    scope:Scope
    errors:ErrorBudget=ErrorBudget()
    reason:str=''
    def __post_init__(self):
        if not isinstance(self.scope,Scope):raise ValueError('EVIDENCE_SCOPE')
        object.__setattr__(self,'cells',tuple(self.cells))
        if not all(isinstance(c,ConvexCell) for c in self.cells):raise ValueError('CELL_TYPE')
        if self.scope==Scope.NATIVE_RAYS and self.cells:raise ValueError('RAYS_CANNOT_CREATE_CONTINUOUS_CELLS')
    def certify(self,coefficients_W,durations,now,required_until=None,max_depth=12,max_nodes=4096):
        """Bernstein hull + ball halfspace support; no finite trajectory sampling.

        Floating-point subdivision with an outward reserve, NOT interval arithmetic.
        The guarantee is conditional on supplied continuous cells and error bounds.
        """
        c=np.asarray(coefficients_W,float);d=np.asarray(durations,float)
        if c.ndim!=3 or c.shape[1:]!=(3,6) or d.shape!=(len(c),) or not np.isfinite(c).all() or not np.isfinite(d).all() or (d<=0).any():raise ValueError('TRAJECTORY_ARRAYS')
        if not np.isfinite(now) or required_until is not None and (not np.isfinite(required_until) or required_until<now):raise ValueError('CERTIFICATE_TIME')
        required=now if required_until is None else required_until
        cells=[x for x in self.cells if x.valid_until>=required]
        base=dict(scope=self.scope.value,trajectory_sha256=trajectory_digest(c,d),checked_at=now,
            guarantee='CONDITIONAL_FLOAT64_BERNSTEIN_CONTAINMENT',error_budget=self.errors.__dict__,selected_permission=False)
        if not cells:return dict(base,status='UNKNOWN',reason=self.reason or 'NO_FRESH_CONTINUOUS_VOLUME',leaves=0,nodes=0)
        stack=[(bernstein_controls(a,t),0,i) for i,(a,t) in enumerate(zip(c,d))]
        leaves=[];nodes=0;minimum=np.inf
        while stack:
            controls,depth,piece=stack.pop();nodes+=1
            if nodes>max_nodes:return dict(base,status='UNKNOWN',reason='SUBDIVISION_BUDGET',leaves=len(leaves),nodes=nodes)
            found=None
            for cell in cells:
                radius=self.errors.radius(float(np.linalg.norm(controls-cell.anchor,axis=1).max()))
                slack=cell.b-np.max(controls@cell.A.T,axis=0)-radius*np.linalg.norm(cell.A,axis=1)-self.errors.numerical
                if slack.min()>=0:
                    found=cell;minimum=min(minimum,float((slack/np.linalg.norm(cell.A,axis=1)).min()));break
            if found is not None:
                leaves.append(dict(piece=piece,depth=depth,cell=found.evidence_id,valid_until=found.valid_until));continue
            if depth>=max_depth:return dict(base,status='UNKNOWN',reason='SWEPT_BODY_NOT_CONTAINED',leaves=len(leaves),nodes=nodes,unproven_piece=piece)
            left,right=split_controls(controls);stack.extend([(right,depth+1,piece),(left,depth+1,piece)])
        return dict(base,status='CERTIFIED_FREE',reason='CONTINUOUS_INFLATED_BODY_CONTAINED',leaves=len(leaves),nodes=nodes,
            minimum_halfspace_slack_m=minimum,valid_until=min(x['valid_until'] for x in leaves),evidence=leaves)

def footprint_cell(frame,u_bounds,v_bounds,lower_depth,valid_until,scope,evidence_id,errors=ErrorBudget()):
    """A continuous angular-patch lower bound, NOT nearest-pixel interpolation.

    Caller must possess an independent full-footprint bound; current native depth
    has none. Therefore native-ray scope is forbidden. Tests/externally surveyed
    bounds are labelled explicitly and can never enter online sensor Ranking.
    """
    if scope==Scope.NATIVE_RAYS:raise ValueError('NATIVE_PIXEL_HAS_NO_FOOTPRINT_GUARANTEE')
    u0,u1=u_bounds;v0,v1=v_bounds
    if not 0<=u0<u1<=159 or not 0<=v0<v1<=127 or not .04+errors.depth_reserve<lower_depth<20 or not frame.stamp<=valid_until:raise ValueError('FOOTPRINT_BOUNDS')
    ylo,yhi=(79.5-u1)/100,(79.5-u0)/100;zlo,zhi=(63.5-v1)/100,(63.5-v0)/100
    A=np.array([[-1,0,0],[1,0,0],[-yhi,1,0],[ylo,-1,0],[-zhi,0,1],[zlo,0,-1]],float)
    b=np.array([-.04,lower_depth-errors.depth_reserve,0,0,0,0],float)
    world_A=A@frame.rotation.T
    return ConvexCell(world_A,b+world_A@frame.origin,valid_until,evidence_id,frame.origin)
