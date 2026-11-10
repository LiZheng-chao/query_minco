"""CPU-only camera interpretation of the actual offline renderer export."""
import numpy as np

OPTICAL_TO_B0 = np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]], dtype=np.float32)


def camera_input(depth_m, metadata):
    camera=metadata["camera"]
    if (camera["width"],camera["height"],camera["max_depth_m"],camera["normalized"])!=(160,90,20,False):
        raise ValueError("unsupported raw renderer camera")
    if metadata["body_quaternion_wxyz"] != [1,0,0,0] or metadata["camera_pitch_deg"]!=0:
        raise ValueError("this qualification profile requires identity body and zero camera pitch")
    if depth_m.dtype!=np.float32 or depth_m.shape!=(90,160) or not np.isfinite(depth_m).all():
        raise ValueError("invalid raw depth")
    # A max-range value is ambiguous/no-return, never certified free space.
    valid=(depth_m>=.04)&(depth_m<20.)
    normalized=np.where(valid,depth_m/np.float32(20),0).astype(np.float32)
    depth_input=np.zeros((2,128,160),np.float32)
    depth_input[0,19:109]=normalized
    depth_input[1,19:109]=valid.astype(np.float32)
    K=[[camera["fx"],0,camera["cx"]],[0,camera["fy"],camera["cy"]],[0,0,1]]
    processed_K=[[camera["fx"],0,camera["cx"]],[0,camera["fy"],camera["cy"]+19],[0,0,1]]
    binding=dict(raw_K=K,processed_K=processed_K,raw_size=[90,160],processed_size=[128,160],
                 optical_to_B0=OPTICAL_TO_B0.tolist(),camera_origin_B0=[0,0,0],
                 body_origin_world=metadata["position_world"],distortion="none",
                 transform_source="actual renderer pose, not ROS display TF",
                 normalization="linear optical Z /20; invalid zero; pad19 top/bottom",
                 valid_return_pixels=int(valid.sum()),ambiguous_far_pixels=int((depth_m>=20).sum()),
                 padding_invalid_pixels=38*160,valid_depth_does_not_imply_free_trajectory=True)
    return depth_input,binding


def replay_depth(counts, metadata):
    """Independent NumPy replay of source stepping/voxel-center depth semantics.

    This checks source conformance, not exact surface-ray intersections or continuous safety.
    """
    c=metadata["camera"]
    width,height=c["width"],c["height"]
    u,v=np.meshgrid(np.arange(width,dtype=np.float32),np.arange(height,dtype=np.float32))
    y=-(u-np.float32(c["cx"]))/np.float32(c["fx"])
    z=-(v-np.float32(c["cy"]))/np.float32(c["fy"])
    length=np.sqrt(np.float32(1)+y*y+z*z)
    x=np.ones_like(y)/length;y=y/length;z=z/length
    resolution=np.float32(metadata["resolution_m"])
    dx=np.float32(.5)*resolution
    dy=(y/x)*dx;dz=(z/x)*dx
    origin=np.asarray(metadata["grid_origin"],np.float32)
    position=np.asarray(metadata["position_world"],np.float32)
    xx=np.full((height,width),position[0],np.float32)
    yy=np.full_like(xx,position[1]);zz=np.full_like(xx,position[2])
    active=np.ones_like(xx,dtype=bool);result=np.zeros_like(xx)
    point_x=np.float32(0)
    for _ in range(402):
        point_x=np.float32(point_x+dx)
        xx+=dx;yy+=dy;zz+=dz
        ix=np.floor((xx-origin[0])/resolution).astype(np.int64)
        iy=np.floor((yy-origin[1])/resolution).astype(np.int64)
        iz=np.floor((zz-origin[2])/resolution).astype(np.int64)
        mx=ix%(2*counts.shape[0]-2);my=iy%(2*counts.shape[1]-2)
        mx=np.where(mx>=counts.shape[0],2*counts.shape[0]-2-mx,mx)
        my=np.where(my>=counts.shape[1],2*counts.shape[1]-2-my,my)
        occupied=iz<=0
        in_z=(iz>0)&(iz<counts.shape[2])
        occupied[in_z]=counts[mx[in_z],my[in_z],iz[in_z]]>metadata["occupy_threshold"]
        hit=active&occupied
        centers=(ix.astype(np.float32)+np.float32(.5))*resolution+origin[0]
        result[hit]=centers[hit]-position[0]
        active[hit]=False
        if point_x>=np.float32(c["max_depth_m"]):
            result[active]=np.float32(c["max_depth_m"]);active[:]=False
        if not active.any():return result
    raise ValueError("ray replay failed to terminate within source range budget")
