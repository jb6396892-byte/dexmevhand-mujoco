"""Opt-in, verified-demonstration reset distribution for the first DAPG curriculum."""
import hashlib
import json
import os
import pickle
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np


@contextmanager
def local_parameter_exports(directory):
    """Confine the legacy trainer's fixed desktop export to this training job."""
    directory = str(Path(directory).resolve())
    expanduser = os.path.expanduser

    def scoped_expanduser(path):
        if path == '~/Desktop/trpo_params':
            return directory
        return expanduser(path)

    with patch.object(os.path, 'expanduser', scoped_expanduser):
        yield


def restore_contact_model(model, demonstration):
    parameters = demonstration.get('physics_model', {})
    if set(parameters) - {'geom_margin', 'geom_gap'}:
        raise ValueError('Unsupported demonstration physics parameters')
    validated = {}
    for name, value in parameters.items():
        target = getattr(model, name)
        value = np.asarray(value, dtype=float)
        if value.shape != target.shape or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError('Invalid demonstration contact parameter: '+name)
        validated[name] = value
    for name, value in validated.items():
        getattr(model, name)[:] = value


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
            restore_contact_model(self.sim.model, demo)
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
