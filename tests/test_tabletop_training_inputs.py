import unittest
import numpy as np
from fromrealhand.tabletop.training_inputs import features,episode_arrays,FEATURE_DIM


class TabletopInputTests(unittest.TestCase):
    def sample(self,video='second',skill='grasp',step=600):
        return features(np.zeros(30),np.ones(30),np.zeros(3),np.eye(4),np.ones(3),
            video,skill,step,(550,601),np.zeros(30),np.ones(30))

    def test_phase_and_reference_clock_are_explicit(self):
        x=self.sample(); self.assertEqual(x.shape,(FEATURE_DIM,))
        np.testing.assert_array_equal(x[72:76],[0,1,0,0])
        self.assertEqual(x[76],1.); np.testing.assert_array_equal(x[77:79],[0,1])

    def test_hold_steps_repeat_source_phase_not_advance_it(self):
        np.testing.assert_array_equal(self.sample(),self.sample())
        with self.assertRaises(ValueError): self.sample(step=601)

    def test_episode_preserves_action_alignment_and_residual(self):
        arrays=episode_arrays([self.sample()]*2,[np.ones(30)*.2]*2,[np.ones(30)*.1]*2,[600,600],[1,1],[28,29])
        np.testing.assert_allclose(arrays['residual_actions'],.1)
        self.assertEqual(len(arrays['inputs']),len(arrays['actions']))

    def test_partial_failed_action_cannot_be_positive_data(self):
        with self.assertRaises(ValueError):
            episode_arrays([self.sample()],[np.zeros(30)]*2,[np.zeros(30)],[600],[1],[28])


if __name__=='__main__': unittest.main()
