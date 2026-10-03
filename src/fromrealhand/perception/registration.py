"""Known upright CAD registration using Open3D ICP with full-yaw multistart."""
import time
import numpy as np
from fromrealhand.tabletop.camera import backproject,transform,project


def depth_foreground(depth,K,T_world_camera,box):
    import cv2
    import open3d as o3d
    world=transform(backproject(depth,K),T_world_camera)
    valid=np.isfinite(depth)&(depth>.1)&(depth<2.)
    if valid.sum()<1000: raise ValueError('Insufficient valid depth')
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(world[valid][::8]))
    cloud=cloud.voxel_down_sample(.008)
    o3d.utility.random.seed(17)
    plane,_=cloud.segment_plane(distance_threshold=.002,ransac_n=3,num_iterations=500)
    plane=np.asarray(plane); plane/=np.linalg.norm(plane[:3])
    if plane[2]<0: plane=-plane
    if plane[2]<.98: raise ValueError('Dominant support plane is not horizontal')
    height=world @ plane[:3]+plane[3]
    x1,y1,x2,y2=box; h,w=depth.shape
    x1,y1=max(0,int(x1)-3),max(0,int(y1)-3); x2,y2=min(w,int(x2)+4),min(h,int(y2)+4)
    roi=np.zeros(depth.shape,bool); roi[y1:y2,x1:x2]=True
    mask=roi&valid&(height>.003)&(height<.13)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    if count<2: raise ValueError('No foreground component')
    component=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])); mask=labels==component
    if mask.sum()<120: raise ValueError('Too few cup depth pixels')
    return mask,world[mask],plane


class UprightRegistration:
    def __init__(self,mesh_file):
        import open3d as o3d
        self.o3d=o3d
        with np.load(mesh_file,allow_pickle=False) as m:
            self.vertices=m['vertices'].copy(); faces=m['faces'].copy()
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(self.vertices),o3d.utility.Vector3iVector(faces))
        mesh.compute_vertex_normals(); o3d.utility.random.seed(31)
        self.model=mesh.sample_points_uniformly(45000).voxel_down_sample(.0012)
        self.model.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.006,max_nn=30))
        self.model_points=np.asarray(self.model.points)

    def estimate(self,points,plane):
        o3d=self.o3d; reg=o3d.pipelines.registration; start=time.monotonic()
        observed=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(.0015)
        if len(observed.points)<100: raise ValueError('Insufficient observed surface')
        center=np.asarray(observed.points).mean(axis=0); model_center=self.model_points.mean(axis=0)
        candidates=[]
        for yaw in np.arange(0,360,15):
            a=np.deg2rad(yaw); R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
            T=np.eye(4); T[:3,:3]=R; T[:3,3]=center-R@model_center
            T[2,3]=-plane[3]/plane[2]-float(self.vertices[:,2].min())
            fit=reg.registration_icp(observed,self.model,.025,np.linalg.inv(T),
                reg.TransformationEstimationPointToPlane(),reg.ICPConvergenceCriteria(max_iteration=45))
            fine=reg.registration_icp(observed,self.model,.005,fit.transformation,
                reg.TransformationEstimationPointToPlane(),reg.ICPConvergenceCriteria(max_iteration=25))
            result=np.linalg.inv(fine.transformation)
            tilt=np.rad2deg(np.arccos(np.clip(result[2,2],-1,1)))
            bottom=float((self.vertices@result[:3,:3].T+result[:3,3])[:,2].min())
            tablez=-plane[3]/plane[2]
            if tilt>15 or abs(bottom-tablez)>.012: continue
            score=float(fine.inlier_rmse+.012*(1-fine.fitness)+.15*abs(bottom-tablez))
            candidates.append(dict(T=result,score=score,fitness=float(fine.fitness),rmse_m=float(fine.inlier_rmse),
                tilt_deg=float(tilt),bottom_to_plane_m=bottom-tablez))
        if not candidates: raise ValueError('No physically plausible upright registration')
        best=min(candidates,key=lambda x:x['score'])
        best.update(latency_s=time.monotonic()-start,candidates=len(candidates),
            observed_points=len(observed.points),method='Open3D point-to-plane ICP; full 360-degree multistart; upright prior')
        best['accepted']=bool(best['fitness']>.85 and best['rmse_m']<.003 and abs(best['bottom_to_plane_m'])<.006)
        return best
