import unittest
import numpy as np
from fromrealhand.tabletop.structured_control import fit_diagonal_residual,StructuredResidual


class StructuredResidualTests(unittest.TestCase):
    def test_known_feedback_is_recovered_without_simulator_parameters(self):
        x=np.random.RandomState(8).normal(size=(100,139))
        gain=np.r_[np.arange(1,7),np.zeros(24)]
        fitted,metrics=fit_diagonal_residual(x,x[:,109:]*gain,device='cpu')
        np.testing.assert_allclose(fitted,gain,atol=1e-12)
        self.assertLess(metrics['mse'],1e-20)
        unseen=np.random.RandomState(9).normal(size=139)*2
        np.testing.assert_allclose(StructuredResidual(fitted).predict(unseen),unseen[109:]*gain)

    def test_unexcited_zero_dimensions_remain_zero(self):
        fitted,_=fit_diagonal_residual(np.zeros((2,139)),np.zeros((2,30)),device='cpu')
        np.testing.assert_array_equal(fitted,0)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError): fit_diagonal_residual(np.zeros((2,139)),np.zeros((3,30)),device='cpu')
        with self.assertRaises(ValueError): StructuredResidual(np.zeros(6))


if __name__=='__main__': unittest.main()
