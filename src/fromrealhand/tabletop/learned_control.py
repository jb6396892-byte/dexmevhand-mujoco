"""BC action inference using the same pre-action inputs as demonstrations."""
from types import SimpleNamespace
import numpy as np
import torch
from .training_inputs import FEATURE_DIM, FEATURE_VERSION, VIDEOS, SKILLS


class LearnedControl:
    def __init__(self, checkpoint):
        from mjrl.policies.gaussian_mlp import MLP
        from ..routed_residual import RoutedNetwork
        saved = torch.load(str(checkpoint), map_location='cpu')
        if saved['feature_version'] != FEATURE_VERSION or saved['feature_dim'] != FEATURE_DIM:
            raise ValueError('Incompatible visual policy features')
        self.mode = saved['mode']
        if self.mode not in ('shared', 'routed-residual', 'structured-residual'):
            raise ValueError('Unknown visual policy mode')
        self.network=None
        if self.mode=='structured-residual':
            from .structured_control import StructuredResidual
            self.structured=StructuredResidual(saved['gain'])
        else:
            base = MLP(SimpleNamespace(observation_dim=FEATURE_DIM, action_dim=30),
                       hidden_sizes=tuple(saved['hidden_sizes']), seed=0).model
            self.network = RoutedNetwork(base, 8) if self.mode == 'routed-residual' else base
            self.network.load_state_dict(saved['state_dict'], strict=True)
            self.network.eval()
        self.mean = np.asarray(saved['normalization']['mean'])
        self.std = np.asarray(saved['normalization']['std'])
        if (self.mean.shape != (FEATURE_DIM,) or self.std.shape != (FEATURE_DIM,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or np.any(self.std <= 0)):
            raise ValueError('Invalid policy normalization')
        self.calls = 0
        self.clipped_calls = 0

    def action(self, inputs, reference, video, phase):
        x = np.asarray(inputs)
        reference = np.asarray(reference)
        if x.shape != (FEATURE_DIM,) or reference.shape != (30,) or not np.isfinite(x).all():
            raise ValueError('Invalid policy inputs')
        route = np.eye(8, dtype=np.float32)[VIDEOS.index(video)*4+SKILLS.index(phase)]
        if self.mode=='structured-residual':
            prediction=self.structured.predict(x)
        else:
            x = torch.as_tensor((x-self.mean)/self.std, dtype=torch.float32)[None]
            with torch.no_grad():
                prediction = (self.network(x, torch.as_tensor(route)[None])
                              if self.mode == 'routed-residual' else self.network(x)).numpy()[0]
        action = reference+prediction if self.mode.endswith('residual') else prediction
        if action.shape != (30,) or not np.isfinite(action).all():
            raise ValueError('Nonfinite learned action')
        self.calls += 1
        self.clipped_calls += int(np.max(np.abs(action)) > 1)
        return np.clip(action, -1., 1.)
