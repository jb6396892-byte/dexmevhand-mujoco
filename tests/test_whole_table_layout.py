import json
from pathlib import Path
import unittest
import numpy as np
from fromrealhand.whole_table.layout import sample, center_bounds, rectangles_clear, OBJECTS
from fromrealhand.whole_table.motion import rest_to_rest, braking

ROOT=Path(__file__).resolve().parents[1]


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/'configs/whole-table-f1.json').read_text())
        self.catalog={n: np.array([[-.04,-.05,-.03],[.06,.05,.1]]) for n in OBJECTS}

    def test_layout_deterministic_and_full_surface(self):
        self.assertEqual(sample(1,self.catalog,self.cfg), sample(1,self.catalog,self.cfg))
        xy=np.array([sample(i,self.catalog,self.cfg)['objects'][0]['xy'] for i in range(100)])
        self.assertTrue(np.all(xy.min(0)<-.25) and np.all(xy.max(0)>.25))

    def test_bounds_and_pair_separation(self):
        for seed in range(100):
            row=sample(seed,self.catalog,self.cfg,4)
            for i,obj in enumerate(row['objects']):
                lo,hi=np.array(obj['bounds_xy'])
                half=np.array(self.cfg['table_size_xy_m'])/2
                self.assertTrue(np.all(lo>=-half+.005) and np.all(hi<=half-.005))
                for other in row['objects'][:i]:
                    self.assertTrue(rectangles_clear((lo,hi), np.array(other['bounds_xy']), .01))

    def test_corner_override_and_invalid_input(self):
        lo,hi=center_bounds(self.catalog['mug'],0,self.cfg)
        self.assertEqual(sample(1,self.catalog,self.cfg,cup_xy=lo)['objects'][0]['xy'],lo.tolist())
        for xy in ([1,0],[0,float('nan')],[0]):
            with self.assertRaises(ValueError): sample(0,self.catalog,self.cfg,cup_xy=xy)
        for count in (-1,5,1.5):
            with self.assertRaises(ValueError): sample(0,self.catalog,self.cfg,count=count)

    def test_quintic_limits_and_endpoints(self):
        a=np.array([-.4,-.35,.1]); b=np.array([.4,.35,.4])
        duration,curve=rest_to_rest(a,b,self.cfg)
        np.testing.assert_allclose(curve(0),a)
        np.testing.assert_allclose(curve(duration),b)
        for degree,field in ((1,'max_velocity_m_s'),(2,'max_acceleration_m_s2'),(3,'max_jerk_m_s3')):
            values=np.abs(curve.derivative(degree)(np.linspace(0,duration,1001)))
            self.assertTrue(np.all(values.max(0)<=self.cfg[field]))
        np.testing.assert_allclose(curve.derivative()(duration),0,atol=1e-10)

    def test_braking_reference_continuity(self):
        duration, curve = rest_to_rest([0,0,.3],[.35,.25,.4],self.cfg)
        t = duration/2
        stop_duration, stop = braking(curve(t), curve.derivative()(t), curve.derivative(2)(t))
        for degree in range(3):
            np.testing.assert_allclose(stop.derivative(degree)(0),curve.derivative(degree)(t),atol=1e-10)
        for degree,field in ((1,'max_velocity_m_s'),(2,'max_acceleration_m_s2'),(3,'max_jerk_m_s3')):
            self.assertTrue(np.all(np.abs(stop.derivative(degree)(np.linspace(0,stop_duration,1001))).max(0)<=self.cfg[field]))
        np.testing.assert_allclose(stop.derivative()(stop_duration),0,atol=1e-10)


if __name__=='__main__': unittest.main()
