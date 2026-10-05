"""Candidate visible-surface tracking; unlike initialization, permits a lifted mug."""
import numpy as np
from .registration import UprightRegistration, depth_foreground
from ..tabletop.camera import backproject, transform


def bounded_box(box, shape):
    values = np.asarray(box, dtype=float)
    if len(shape) != 2 or values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError('Invalid detection box or depth shape')
    h, w = shape
    if h <= 0 or w <= 0 or values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError('Empty detection box')
    values[[0, 2]] = np.clip(values[[0, 2]], 0, w)
    values[[1, 3]] = np.clip(values[[1, 3]], 0, h)
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError('Detection box outside depth image')
    return values


class CupTracker:
    def __init__(self, mesh, max_tracking_tilt_deg=45.):
        self.registration = UprightRegistration(mesh)
        self.previous = None
        self.plane = None
        self.max_tracking_tilt_deg = float(max_tracking_tilt_deg)

    def estimate(self, depth, k, twc, box):
        box = bounded_box(box, depth.shape)
        import cv2
        if self.previous is None:
            mask, points, plane = depth_foreground(depth, k, twc, box)
            result = self.registration.estimate(points, plane)
            if result['accepted']:
                self.previous, self.plane = result['T'].copy(), plane.copy()
            return mask, result
        world = transform(backproject(depth, k), twc)
        valid = np.isfinite(depth) & (depth > .1) & (depth < 2.)
        height = world @ self.plane[:3]+self.plane[3]
        h, w = depth.shape
        x1, y1, x2, y2 = np.asarray(box, dtype=int)
        roi = np.zeros((h, w), dtype=bool)
        roi[max(0, y1-3):min(h, y2+4), max(0, x1-3):min(w, x2+4)] = True
        # A loose detection may include the forearm. Bound candidates by the last
        # accepted visual CAD pose and the permitted inter-frame motion budget.
        local_world = (world-self.previous[:3,3]) @ self.previous[:3,:3]
        vertices = self.registration.vertices
        tracking_volume = np.all((local_world >= vertices.min(0)-.035) &
                                 (local_world <= vertices.max(0)+.035),axis=-1)
        mask = roi & valid & tracking_volume & (height > .003) & (height < .55)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        if count < 2: raise ValueError('No tracked foreground')
        mask = labels == 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if mask.sum() < 120: raise ValueError('Insufficient tracked surface')
        o3d = self.registration.o3d
        observed = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(world[mask])).voxel_down_sample(.0015)
        raw_observed = observed
        prior_local = o3d.geometry.PointCloud(observed).transform(np.linalg.inv(self.previous))
        prior_distances = np.asarray(prior_local.compute_point_cloud_distance(self.registration.model))
        prior_keep = np.flatnonzero(prior_distances < .018)
        if len(prior_keep) < 300:
            raise ValueError('Insufficient surface near previous visual pose')
        observed = observed.select_by_index(prior_keep.tolist())
        reg = o3d.pipelines.registration
        estimator = reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=.008))
        coarse = reg.registration_icp(observed, self.registration.model, .035, np.linalg.inv(self.previous),
            estimator, reg.ICPConvergenceCriteria(max_iteration=50))
        prior_rotation=np.linalg.inv(self.previous)[:3,:3]
        coarse_angle=np.rad2deg(np.arccos(np.clip((np.trace(prior_rotation.T @ coarse.transformation[:3,:3])-1)/2,-1,1)))
        rotation_fallback=bool(coarse_angle>20.)
        if rotation_fallback:
            # A partly hidden cylindrical surface underconstrains point-to-plane
            # yaw. Refit fresh points from the prior with point-to-point ICP.
            estimator=reg.TransformationEstimationPointToPoint()
            coarse=reg.registration_icp(observed,self.registration.model,.018,np.linalg.inv(self.previous),
                estimator,reg.ICPConvergenceCriteria(max_iteration=50))
        # An RGB box also contains the approaching hand. Trim geometric outliers,
        # but retain raw support/coverage gates so a tiny matching patch cannot pass.
        local = o3d.geometry.PointCloud(observed).transform(coarse.transformation)
        distances = np.asarray(local.compute_point_cloud_distance(self.registration.model))
        keep = np.flatnonzero(distances < .006)
        retained = len(keep)/max(1,len(distances))
        selected = observed.select_by_index(keep.tolist())
        cells = len(np.unique(np.floor(np.asarray(local.points)[keep]/.005).astype(int),axis=0))
        if len(keep)<300 or retained<.5 or cells<40:
            raise ValueError('Insufficient visible model surface under occlusion: points=%d retained=%.3f cells=%d' %
                             (len(keep),retained,cells))
        fine = reg.registration_icp(selected, self.registration.model, .005, coarse.transformation,
            estimator, reg.ICPConvergenceCriteria(max_iteration=30))
        raw = reg.evaluate_registration(raw_observed,self.registration.model,.005,fine.transformation)
        t = np.linalg.inv(fine.transformation)
        tilt = float(np.rad2deg(np.arccos(np.clip(t[2, 2], -1, 1))))
        result = dict(T=t, accepted=bool(fine.fitness > .85 and fine.inlier_rmse < .003 and tilt < self.max_tracking_tilt_deg),
            fitness=float(fine.fitness), rmse_m=float(fine.inlier_rmse), tilt_deg=tilt,
            tracking_tilt_limit_deg=self.max_tracking_tilt_deg,initial_upright_prior_deg=15.,
            observed_points=len(observed.points), selected_points=len(keep), surface_cells=cells,
            raw_observed_points=len(raw_observed.points),prior_retained_fraction=len(prior_keep)/len(raw_observed.points),
            retained_fraction=retained, raw_fitness=float(raw.fitness),fitness_scope='model-gated observed surface',
            point_to_point_rotation_fallback=rotation_fallback,unconstrained_coarse_rotation_deg=float(coarse_angle),
            method='visual-initialized Tukey ICP with visible-surface gates; airborne allowed')
        pixels=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(world[mask])).transform(fine.transformation)
        valid_pixels=np.asarray(pixels.compute_point_cloud_distance(self.registration.model))<.005
        refined=np.zeros_like(mask); refined[mask]=valid_pixels; mask=refined
        if result['accepted']: self.previous = t.copy()
        return mask, result
