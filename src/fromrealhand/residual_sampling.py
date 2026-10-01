"""Keep latent residual likelihoods separate from executed normalized controls."""
import numpy as np
from .multivideo import conditioned_features
from .reference_tracking import ReferenceContext, residual_action


class ResidualSampler:
    def __init__(self, checkpoint, experiment, video, contact_reader, deterministic=False):
        self.checkpoint, self.exp, self.video = checkpoint, experiment, video
        self.deterministic = deterministic
        self.observations, self.latent_actions, self.executed_actions = [], [], []
        self.context = None
        actual = [experiment.env.control_timestep, experiment.duration, video['control']['time_scale']]
        if not np.allclose(checkpoint['clocks'][video['name']], actual, rtol=0., atol=1e-10):
            raise ValueError('Sampling clock differs from training reference')
        if checkpoint.get('feature_mode') == 'reference_contact':
            self.context = ReferenceContext(experiment, video, checkpoint['reference_demos'][video['name']],
                                            checkpoint['nominal_geometry'][video['name']], contact_reader)

    def __len__(self):
        return self.video['horizon']

    def __getitem__(self, step):
        e = self.exp.env
        if self.context is None:
            x = conditioned_features(e._get_observations(), e.sim.data.qpos, e.sim.data.qvel, step,
                                      e.control_timestep, self.exp.duration,
                                      self.video['control']['time_scale'], self.video['id'])
            reference = self.checkpoint['references'][self.video['name']][step]
        else:
            x = self.context.features(step)
            reference = self.context.actions[step]
        sample, info = self.checkpoint['policy'].get_action(x)
        latent = info['evaluation'] if self.deterministic else sample
        action = residual_action(reference, latent, self.checkpoint['residual_limits'][self.video['name']])
        self.observations.append(x.copy())
        # DAPG evaluates the density of the *unclipped* sampled residual.
        self.latent_actions.append(latent.copy())
        self.executed_actions.append(action.copy())
        return action

    def path(self, demo):
        observations, actions = np.asarray(self.observations), np.asarray(self.latent_actions)
        rewards = np.asarray(demo['rewards'])
        if len(observations) != len(rewards) or not all(np.isfinite(x).all() for x in (observations, actions, rewards)):
            raise ValueError('Invalid sampled residual trajectory')
        return dict(observations=observations, actions=actions, rewards=rewards,
                    terminated=True, agent_infos={}, env_infos={},
                    executed_actions=np.asarray(self.executed_actions))
