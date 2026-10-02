import json
from pathlib import Path
import tempfile
import unittest
import hashlib
import numpy as np
from fromrealhand.skill_inputs import skill_features,fit_normalization,SkillSequenceDataset,FEATURE_DIM


class SkillInputTests(unittest.TestCase):
    def test_global_reference_clock_is_not_reset_at_skill_entry(self):
        entry=dict(video='second',horizon=10,dt=.01,control=dict(time_scale=5.))
        segment=dict(skill='lift',start=5,stop=8)
        geometry=dict(source_frames=np.array([1,73]),fps=30.,object_poses=np.tile(np.eye(4),(2,1,1)))
        ref=np.tile(np.arange(10)[:,None],(1,30))*.01
        q=np.zeros(37);q[33]=1.
        x=skill_features(np.zeros(39),q,np.zeros(36),5,entry,segment,geometry,ref)
        self.assertEqual(x.shape,(FEATURE_DIM,))
        self.assertEqual(x[88],0.)
        np.testing.assert_array_equal(x[92:],ref[5])
        np.testing.assert_array_equal(x[84:88],[0,0,1,0])
        with self.assertRaises(ValueError):
            skill_features(np.zeros(39),q,np.zeros(36),8,entry,segment,geometry,ref)

    def test_statistics_use_only_supplied_training_arrays(self):
        train=[np.array([[1.,2.],[3.,2.]])]
        stats=fit_normalization(train)
        np.testing.assert_array_equal(stats['mean'],[2.,2.])
        np.testing.assert_array_equal(stats['std'],[1.,1.])
        self.assertEqual(stats['fitted_frames'],2)

    def test_windows_pad_with_mask_not_next_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            mean=dict(mean=[0.]*FEATURE_DIM,std=[1.]*FEATURE_DIM)
            (root/'norm.json').write_text(json.dumps(mean))
            x=np.zeros((3,FEATURE_DIM));x[:,0]=[10,20,30]
            np.savez(root/'ep.npz',inputs=x,actions=np.zeros((3,30)),global_steps=np.array([5,6,7]))
            record=dict(split='train',artifact='ep.npz',video='first',skill='grasp',
                        sha256=hashlib.sha256((root/'ep.npz').read_bytes()).hexdigest())
            (root/'index.json').write_text(json.dumps(dict(normalization='norm.json',
                normalization_sha256=hashlib.sha256((root/'norm.json').read_bytes()).hexdigest(),records=[record])))
            ds=SkillSequenceDataset(root/'index.json','train')
            batch=ds.window(0,2,4)
            np.testing.assert_array_equal(batch['mask'].ravel(),[True,False,False,False])
            np.testing.assert_array_equal(batch['global_steps'],[7,7,7,7])
            self.assertEqual(ds[0]['inputs'][0,0],10)
            with self.assertRaises(IndexError): ds[3]
            first=ds.sample_batch(8,2,42)['inputs']
            np.testing.assert_array_equal(first,ds.sample_batch(8,2,42)['inputs'])


if __name__=='__main__': unittest.main()
