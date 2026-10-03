import unittest
import numpy as np
from fromrealhand.tabletop.camera import intrinsics,metric_depth,backproject,transform,project


class TabletopCameraTests(unittest.TestCase):
    def test_intrinsics(self):
        K=intrinsics(640,480,90)
        np.testing.assert_allclose(K,[[240,0,319.5],[0,240,239.5],[0,0,1]])

    def test_depth_clipping(self):
        np.testing.assert_allclose(metric_depth(np.array([0.,1.]),.1,10),[.1,10])

    def test_depth_roundtrip(self):
        z=np.array([.2,.5,1.,3.]); near,far=.05,10.
        buffer=(1-near/z)/(1-near/far)
        np.testing.assert_allclose(metric_depth(buffer,near,far),z,rtol=1e-6)

    def test_invalid_depth(self):
        for b in (np.array([-1]),np.array([1.2]),np.array([np.nan])):
            with self.assertRaises(ValueError): metric_depth(b,.1,10)
        with self.assertRaises(ValueError): metric_depth(np.array([.2]),10,.1)

    def test_backproject_project(self):
        d=np.full((24,32),.75); K=intrinsics(32,24,50)
        v,u=np.indices(d.shape)
        np.testing.assert_allclose(project(backproject(d,K),K),np.stack([u,v],axis=-1),atol=1e-12)

    def test_extrinsics_roundtrip(self):
        T=np.array([[0,-1,0,.3],[1,0,0,-.1],[0,0,1,.7],[0,0,0,1.]])
        points=np.array([[.2,.5,1.],[-.4,.2,2.]])
        np.testing.assert_allclose(transform(transform(points,T),np.linalg.inv(T)),points,atol=1e-12)

    def test_behind_camera(self):
        with self.assertRaises(ValueError): project(np.array([[0,0,-1.]]),np.eye(3))


if __name__=='__main__': unittest.main()
