"""Typed renderer-world occupancy. UNKNOWN is an error, never a free distance."""
import hashlib
import numpy as np
import torch
from scipy.spatial import cKDTree

_SEAL=object()


def world_points(points,origin,rotation):
    return points @ rotation.T + origin


def voxel_boxes(counts,meta,roi):
    """Materialize exact symmetricIndex extension in a finite, declared world ROI."""
    origin=np.asarray(meta["grid_origin"],np.float32).astype(np.float64)
    h=float(np.float32(meta["resolution_m"]))
    low=np.floor((roi[0]-origin)/h).astype(int)
    high=np.ceil((roi[1]-origin)/h).astype(int)
    # z<=0 is the independent solid floor; z>=nz remains UNKNOWN.
    low[2]=max(low[2],1);high[2]=min(high[2],counts.shape[2])
    indices=np.stack(np.meshgrid(*[np.arange(a,b) for a,b in zip(low,high)],indexing="ij"),-1).reshape(-1,3)
    folded=indices.copy()
    for axis,n in enumerate(counts.shape[:2]):
        folded[:,axis]%=2*n-2
        folded[:,axis]=np.minimum(folded[:,axis],2*n-2-folded[:,axis])
    occupied=counts[tuple(folded.T)]>meta["occupy_threshold"]
    i=indices[occupied]
    lo=origin+i*h-1e-5;hi=origin+(i+1)*h+1e-5
    return np.stack([lo,hi],1)


class TypedBoundOccupancy:
    """Only hash-checked production loader may construct this type.

    Occupancy completeness is ONLY for the generated, deterministic renderer world.
    Occluded cells are known through exported source counts, not inferred from depth.
    No-hit depth alone supplies no geometry authority. Outside ROI/top is UNKNOWN.
    """
    def __init__(self,*,seal,counts,meta,pose,roi,domain,identity,revalidate):
        if seal is not _SEAL:raise ValueError("UNQUALIFIED_GEOMETRY")
        self.counts=counts.copy();self.counts.flags.writeable=False
        self.meta=dict(meta);self.identity=dict(identity)
        self.origin=np.asarray(pose["position_world"],np.float64)
        self.rotation=np.asarray(pose["R_WB0"],np.float64)
        self.roi=np.asarray(roi,np.float64);self.domain=torch.tensor(domain,dtype=torch.float32)
        if (self.rotation.shape!=(3,3) or not np.allclose(self.rotation.T@self.rotation,np.eye(3),atol=1e-10) or
                not np.isclose(np.linalg.det(self.rotation),1)):raise ValueError("INVALID_ROTATION")
        top=float(np.float32(meta["grid_origin"][2]))+counts.shape[2]*float(np.float32(meta["resolution_m"]))
        if self.roi.shape!=(2,3) or (self.roi[1]<=self.roi[0]).any() or self.roi[1,2]>top-1e-5:
            raise ValueError("UNKNOWN_TOP_OR_ROI")
        self.floor=float(np.float32(meta["grid_origin"][2]))+float(np.float32(meta["resolution_m"]))+1e-5
        self.boxes=voxel_boxes(counts,meta,self.roi)
        self.centers=self.boxes.mean(1);self.halves=(self.boxes[:,1]-self.boxes[:,0])/2
        self.tree=cKDTree(self.centers)
        self.circumradius=float(np.linalg.norm(self.halves,axis=1).max())
        self._revalidate=revalidate
        self._fingerprint=self._memory_hash()

    def _memory_hash(self):
        import json
        h=hashlib.sha256()
        for a in (self.counts,self.origin,self.rotation,self.roi,self.boxes,self.centers,self.halves,self.domain.numpy()):
            h.update(np.asarray(a).tobytes())
        h.update(json.dumps(dict(meta=self.meta,identity=self.identity),sort_keys=True).encode())
        return h.hexdigest()

    def validate(self):
        self._revalidate()
        if self._memory_hash()!=self._fingerprint:raise ValueError("GEOMETRY_MEMORY_DRIFT")

    def sdf(self,points):
        self.validate()
        if (points.device.type!="cpu" or points.dtype not in (torch.float32,torch.float64) or
                points.shape[-1]!=3 or not bool(torch.isfinite(points).all())):raise ValueError("POINT_TYPE")
        p=world_points(points,torch.as_tensor(self.origin,dtype=points.dtype),
                       torch.as_tensor(self.rotation,dtype=points.dtype))
        return self._sdf_world_known(p)

    def _sdf_world_known(self,p):
        """Private exact known-world evaluator; no B0 roundtrip or ROI tolerance."""
        self.validate()
        if (p.device.type!="cpu" or p.dtype not in (torch.float32,torch.float64) or
                p.shape[-1]!=3 or not bool(torch.isfinite(p).all())):raise ValueError("POINT_TYPE")
        flat=p.reshape(-1,3);x=flat.detach().numpy().astype(np.float64)
        if ((x<self.roi[0])|(x>self.roi[1])).any():raise ValueError("UNKNOWN_OUTSIDE_CERTIFIED_ROI")
        # Exact nearest AABB selection; no EDT/interpolation. Any closer box must have
        # its center within d(nearest center)+circumradius. eps=0, sorted tie handling.
        d,_=self.tree.query(x)
        candidates=self.tree.query_ball_point(x,d+self.circumradius+1e-6,eps=0)
        chosen=[]
        for point,ids in zip(x,candidates):
            ids=np.sort(ids);q=np.abs(point-self.centers[ids])-self.halves[ids]
            sd=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(1),0)
            chosen.append(ids[np.argmin(sd)])
        centers=torch.as_tensor(self.centers[chosen],dtype=p.dtype)
        halves=torch.as_tensor(self.halves[chosen],dtype=p.dtype)
        q=(flat-centers).abs()-halves
        distance=q.clamp_min(0).norm(dim=-1)+q.amax(-1).clamp_max(0)
        floor_distance=flat[:,2]-self.floor
        boundary=torch.minimum(flat-torch.as_tensor(self.roi[0],dtype=p.dtype),
                               torch.as_tensor(self.roi[1],dtype=p.dtype)-flat).amin(-1)
        return torch.minimum(torch.minimum(distance,floor_distance),boundary).reshape(p.shape[:-1])
