import unittest
import numpy as np
from types import SimpleNamespace
from fromrealhand.whole_table.local_adapter import MotionBridge


class MotorLimitTests(unittest.TestCase):
    def test_contact_response_limits_only_motor_command(self):
        data=SimpleNamespace(ctrl=np.array([2.,-2.,.2]),qpos=np.array([.1,.2,.3]),qacc=np.zeros(3))
        model=SimpleNamespace(actuator_ctrlrange=np.array([[-3.,3.]]*3))
        sim=SimpleNamespace(data=data,model=model)
        sim.forward=lambda: setattr(data,'qacc',data.ctrl*100)
        bridge=MotionBridge.__new__(MotionBridge);bridge.sim=sim;bridge.tcols=np.arange(3)
        bridge.acceleration=lambda:data.qacc
        # The bound on each correction is conservative, so test near the gate.
        data.ctrl[:]=[.006,-.005,.001]
        old=data.qpos.copy()
        bridge.limit_translation_acceleration(.35)
        self.assertLessEqual(abs(data.qacc).max(),.3501)
        np.testing.assert_array_equal(old,data.qpos)

    def test_inactive_limit_leaves_command(self):
        data=SimpleNamespace(ctrl=np.array([.1,.1,.1]),qacc=np.array([.1,.1,.1]))
        sim=SimpleNamespace(data=data,model=None,forward=lambda:None)
        bridge=MotionBridge.__new__(MotionBridge);bridge.sim=sim;bridge.acceleration=lambda:data.qacc
        bridge.limit_translation_acceleration(.35)
        np.testing.assert_array_equal(data.ctrl,[.1,.1,.1])

    def test_solver_seed_is_identical_for_prediction_and_execution(self):
        data=SimpleNamespace(ctrl=np.array([.006,-.005,.001]),qacc=np.zeros(3),qacc_warmstart=np.array([.03,.02,.01]))
        model=SimpleNamespace(actuator_ctrlrange=np.array([[-3.,3.]]*3))
        def forward():
            data.qacc=data.ctrl*100+data.qacc_warmstart
            data.qacc_warmstart[:]=data.qacc
        bridge=MotionBridge.__new__(MotionBridge)
        bridge.sim=SimpleNamespace(data=data,model=model,forward=forward)
        bridge.tcols=np.arange(3);bridge.acceleration=lambda:data.qacc
        seed=data.qacc_warmstart.copy()
        bridge.limit_translation_acceleration(.3)
        predicted=data.qacc.copy()
        np.testing.assert_array_equal(data.qacc_warmstart,seed)
        forward()
        np.testing.assert_allclose(data.qacc,predicted)
        self.assertLessEqual(abs(data.qacc).max(),.3001)
