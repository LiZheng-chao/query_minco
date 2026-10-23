import numpy as np
from .validation import evaluate

def numpy_box_sdf(points,box):
    center=(np.asarray(box["min"])+np.asarray(box["max"]))/2
    half=(np.asarray(box["max"])-np.asarray(box["min"]))/2
    q=np.abs(np.asarray(points)-center)-half
    return np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(-1),0)

def fine_geometry(coeffs,durations,scene,points=4097):
    positions=[];times=[];start=0.
    for c,T in zip(coeffs,durations):
        local=np.linspace(0,float(T),points)
        positions.append(evaluate(c,local));times.append(start+local);start+=float(T)
    p=np.concatenate(positions);t=np.concatenate(times)
    distances=np.stack([numpy_box_sdf(p,b) for b in scene["obstacles"]])
    bi,pi=np.unravel_index(distances.argmin(),distances.shape)
    box=scene["obstacles"][bi];pos=p[pi];face=np.clip(pos,box["min"],box["max"])
    # Sphere surface clearance; no failed-verifier sentinel is involved.
    surface=float(distances[bi,pi]-.12)
    return {"minimum_signed_surface_clearance_m":surface,"closest_time_s":float(t[pi]),
            "closest_position_B0_m":pos.tolist(),"closest_box":box["name"],
            "closest_box_surface_B0_m":face.tolist(),
            "center_to_obstacle_signed_distance_m":float(distances[bi,pi]),
            "horizontal_outer_face_gap_m":float(pos[1]-box["max"][1] if pos[1]>=0 else box["min"][1]-pos[1]),
            "sampling_points":len(p)}
