"""Opt-in, verified-demonstration reset distribution for the first DAPG curriculum."""
import hashlib
import json
import os
import pickle
from pathlib import Path


def load_admitted_demos():
    admission_path = Path(os.environ['FROMREALHAND_ADMISSION'])
    admission = json.loads(admission_path.read_text())
    if admission.get('training_ready') is not True:
        raise ValueError('Physical admission has not passed')
    demo_path = Path(os.environ['FROMREALHAND_VERIFIED_DEMO'])
    raw = demo_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != admission['demo_sha256']:
        raise ValueError('Demonstration hash differs from the verified artifact')
    demos = pickle.loads(raw)
    return demos, admission


def make_verified_environment(env_name=None):
    if env_name != 'relocate-mug-0.8':
        raise ValueError('Verified curriculum only supports relocate-mug-0.8')
    from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate
    from hand_imitation.env.environments.dapg_env.dapg_wrapper import DAPGWrapper
    demonstrations, _ = load_admitted_demos()

    class VerifiedRelocate(YCBRelocate):
        def __init__(self):
            self.demonstrations = list(demonstrations.values())
            self.trajectory_names = list(demonstrations)
            super().__init__(has_renderer=False, object_name='mug', object_scale=.8,
                             friction=(1, .5, .01), solref='-6000 -300', randomness_scale=.25)
            self.horizon = len(self.demonstrations[0]['actions'])

        def _reset_internal(self):
            super()._reset_internal()
            self.demo_index = int(self.np_random.randint(len(self.demonstrations)))
            demo = self.demonstrations[self.demo_index]
            self.sim.reset()
            self.pack_mujoco_model(demo['model_data'][0])
            self.pack(demo['sim_data'][0])
            self.sim.forward()

    class SeededWrapper(DAPGWrapper):
        def seed(self, seed=None):
            super().seed(seed)
            return self.env.seed(seed)

    return SeededWrapper(VerifiedRelocate())


def install_verified_factory():
    from importlib import import_module
    names = ['mjrl.utils.get_environment', 'mjrl.samplers.base_sampler',
             'mjrl.samplers.batch_sampler', 'mjrl.samplers.evaluation_sampler']
    for name in names:
        import_module(name).get_environment = make_verified_environment
