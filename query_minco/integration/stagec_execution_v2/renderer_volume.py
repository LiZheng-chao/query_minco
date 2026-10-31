"""Full native-ray covering proof for the declared voxel renderer model only.

Does NOT certify physical point-cloud geometry, state/tracking/braking bounds, or
an online execution candidate. Query depth alone has insufficient covering density.
"""
from dataclasses import dataclass
import hashlib,math
import numpy as np
from query_minco.inference.observed_space import Scope,ConvexCell,FreeVolume,ErrorBudget
@dataclass(frozen=True)
class NativeModel:
    width:int=800; height:int=512; fx:float=400.; fy:float=400.; cx:float=400.; cy:float=254.
    resolution:float=.1; max_depth:float=20.; origin_linf_bound:float=10.; grid_origin_linf_bound:float=100.
    def proof(self):
        vals=list(self.__dict__.values())
        if not all(np.isfinite(x) for x in vals) or min(self.fx,self.fy,self.resolution,self.max_depth)<=0 or min(self.origin_linf_bound,self.grid_origin_linf_bound)<0 or type(self.width)!=int or type(self.height)!=int:raise ValueError('RENDER_MODEL')
        if not 0<self.cx<self.width-1 or not 0<self.cy<self.height-1:raise ValueError('FOV')
        h=.5*self.resolution;N=math.ceil(self.max_depth/h)+1;u=2.**-24
        gamma=lambda n:n*u/(1-n*u)
        ty=max(self.cx,self.width-1-self.cx)/self.fx;tz=max(self.cy,self.height-1-self.cy)/self.fy
        L=math.sqrt(1+ty*ty+tz*tz)
        # Source-level FP32 bound: N accumulated adds; <=128 rounded elementary
        # operations for fixed quaternion step and inverse projection. Identity
        # or normalized pose and finite bounded coordinates are required.
        e=math.sqrt(3)*(gamma(N)*(self.origin_linf_bound+N*h*L)+N*gamma(128)*h*3*(1+ty+tz)+gamma(128)*(self.origin_linf_bound+N*h*L))
        e+=math.sqrt(3)*gamma(4)*(self.grid_origin_linf_bound+self.origin_linf_bound+N*h*L) # Pos2Vox subtraction/division registration roundoff
        cover=math.sqrt((h/2)**2+((self.max_depth+h)/(2*self.fx))**2+((self.max_depth+h)/(2*self.fy))**2)
        return dict(step=h,steps=N,float_error_m=e,lattice_cover_radius_m=cover,total_cover_radius_m=cover+e,inscribed_voxel_radius_m=self.resolution/2,covered=cover+e<self.resolution/2,assumptions='IEEE FP32 round-to-nearest bounded source operations; normalized rigid pose; grid origin bounded by declared100m contract; declared constant voxel occupancy and exact native model; NOT unvoxelized physical geometry')
class RendererHistory:
    def __init__(self,model=NativeModel(),max_age=.25):
        if not 0<max_age<=.25:raise ValueError('AGE')
        self.model=model;self.max_age=max_age;self.frames=[];self.last=None;self.last_now=None
    def add(self,stamp,depth_m,origin,rotation):
        m=self.model;d=np.array(depth_m,float,copy=True);o=np.array(origin,float);R=np.array(rotation,float)
        if not np.isfinite(stamp) or stamp<=0:raise ValueError('STAMP')
        if self.last is not None and stamp<=self.last:self.frames.clear();self.last=None;raise ValueError('CLOCK_RESET_HISTORY_CLEARED')
        if d.shape!=(m.height,m.width) or o.shape!=(3,) or R.shape!=(3,3) or not np.isfinite(o).all() or not np.isfinite(R).all() or np.max(abs(o))>m.origin_linf_bound or not np.allclose(R.T@R,np.eye(3),atol=1e-7,rtol=0) or abs(np.linalg.det(R)-1)>1e-7:raise ValueError('SENSOR_MODEL_POSE')
        self.frames.append((stamp,d,o,R));self.last=stamp;self.frames=[f for f in self.frames[-8:] if stamp-f[0]<=self.max_age]
    def volume(self,now,errors=ErrorBudget()):
        m=self.model;proof=m.proof();cells=[];reasons=[]
        if self.last_now is not None and now<self.last_now or self.last is not None and now<self.last:
            self.frames.clear();self.last=None;self.last_now=now;return FreeVolume((),Scope.RENDERER_VOXELS,errors,reason='CLOCK_RESET_HISTORY_CLEARED'),proof
        self.last_now=now
        if not proof['covered']:return FreeVolume((),Scope.RENDERER_VOXELS,errors,reason='RAY_LATTICE_COVERAGE_NOT_PROVED'),proof
        for stamp,d,o,R in self.frames:
            if not 0<=now-stamp<=self.max_age:reasons.append('STALE');continue
            # Global prefix is intentionally conservative: ANY invalid or max-range
            # pixel invalidates this one cell. No interpolation/unknown filling.
            if not np.isfinite(d).all() or np.any(d<=0) or np.any(d>=m.max_depth):reasons.append('INVALID_OR_MAXRANGE');continue
            rho=math.sqrt(3)*m.resolution/2;e=proof['float_error_m'];reserve=rho+proof['total_cover_radius_m']
            free_end=float(d.min())-rho-e-proof['step']-errors.depth_reserve
            yl=(m.cx-(m.width-2))/m.fx;yh=(m.cx-1)/m.fx;zl=(m.cy-(m.height-2))/m.fy;zh=(m.cy-1)/m.fy
            A=np.array([[-1,0,0],[1,0,0],[-yh,1,0],[yl,-1,0],[-zh,0,1],[zl,0,-1]],float)
            b=np.array([-proof['step'],free_end,0,0,0,0])-reserve*np.linalg.norm(A,axis=1)
            W=A@R.T;identity=hashlib.sha256(d.astype('<f8').tobytes()+o.tobytes()+R.tobytes()+np.array([stamp],dtype='<f8').tobytes()).hexdigest()
            cells.append(ConvexCell(W,b+W@o,stamp+self.max_age,'native_voxel_conditional:'+identity,o))
        return FreeVolume(tuple(cells),Scope.RENDERER_VOXELS,errors,reason='NO_PROVED_NATIVE_VOXEL_CELL:'+','.join(reasons)),proof
    def certify(self,c,d,now,errors=ErrorBudget()):
        if errors.body_radius!=.12 or errors.margin<.03 or min(errors.tracking,errors.translation,errors.extrinsic,errors.rotation_rad,errors.depth_reserve)<=0:raise ValueError('UNQUALIFIED_ZERO_OR_REDUCED_BODY_ERROR_BUDGET')
        vol,proof=self.volume(now,errors);z=vol.certify(c,d,now,required_until=now+float(np.sum(d)))
        z.update(covering_proof=proof,physical_status='UNKNOWN',physical_geometry_status='UNQUALIFIED_VOXEL_MODEL_ONLY',tracking_localization_latency_braking_bounds='UNQUALIFIED_NOT_ZERO',online_permission=False)
        assert not z['selected_permission'];return z
