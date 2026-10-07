"""Evidence collection for the platform milestone, not grasp-success metrics."""
import numpy as np
from .motion import rest_to_rest


def move(scene, target, on_step=None, curve_spec=None):
    dt = scene.config['timestep_s']
    duration, curve = curve_spec or rest_to_rest(scene.position(), target, scene.config)
    peak_v, peak_a = np.zeros(3), np.zeros(3)
    peak_error, violation, collision_steps = 0., 0., 0
    m, d = scene.sim.model, scene.sim.data
    robot_bodies = set()
    for bid in range(1, m.nbody):
        ancestor = bid
        while ancestor:
            if m.body_id2name(ancestor) == 'gantry_x_body':
                robot_bodies.add(bid); break
            ancestor = int(m.body_parentid[ancestor])
    for index in range(int(np.ceil((duration+1.)/dt))):
        t = min((index+1)*dt, duration)
        commanded = curve(t)
        scene.step(commanded)
        peak_v = np.maximum(peak_v, np.abs(d.qvel[scene.vids]))
        peak_a = np.maximum(peak_a, np.abs(d.qacc[scene.vids]))
        peak_error = max(peak_error, float(np.max(np.abs(scene.position()-commanded))))
        violation = max(violation, float(np.max(np.maximum(
            np.asarray(scene.config['joint_min_m'])-scene.position(),
            scene.position()-np.asarray(scene.config['joint_max_m'])))))
        collision_steps += int(any(c.dist <= 0 and
            ((int(m.geom_bodyid[c.geom1]) in robot_bodies) != (int(m.geom_bodyid[c.geom2]) in robot_bodies))
            for c in d.contact[:d.ncon]))
        if on_step: on_step(index)
    error = float(np.max(np.abs(scene.position()-target)))
    site_error = float(np.linalg.norm(d.get_site_xpos('S_grasp')-(np.asarray(target)+[0, 0, .1])))
    passed = (error <= scene.config['position_tolerance_m'] and site_error <= .003
              and np.all(peak_v <= np.asarray(scene.config['max_velocity_m_s'])+.001)
              and np.all(peak_a <= np.asarray(scene.config['max_acceleration_m_s2'])+.01)
              and violation <= scene.config['joint_violation_tolerance_m'] and collision_steps == 0
              and scene.peak_penetration <= scene.config['max_penetration_m'])
    return dict(passed=bool(passed), target_m=np.asarray(target).tolist(),
                duration_s=duration+1, final_joint_error_m=error, grasp_site_error_m=site_error,
                peak_tracking_error_m=peak_error, max_velocity_m_s=peak_v.tolist(),
                max_acceleration_m_s2=peak_a.tolist(), joint_violation_m=max(0., violation),
                robot_environment_collision_steps=collision_steps,
                max_penetration_m=scene.peak_penetration)


def screenshot(scene, path):
    from ..desktop.rendering import stream_context
    from PIL import Image
    context = stream_context(scene.sim)
    context.cam.lookat[:] = [0, 0, .36]
    context.cam.distance = 3.9
    context.cam.azimuth = 135
    context.cam.elevation = -25
    context.vopt.geomgroup[2] = 0
    context.vopt.geomgroup[4] = 0
    context.render(1200, 900)
    pixels = context.read_pixels(1200, 900, depth=False)[::-1]
    if np.std(pixels) < 5: raise RuntimeError('Blank rendering')
    Image.fromarray(pixels).save(str(path))
    return dict(pixel_std=float(np.std(pixels)), width=1200, height=900)
