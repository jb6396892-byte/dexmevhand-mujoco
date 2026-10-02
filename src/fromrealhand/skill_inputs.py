"""Skill-aware training inputs with episode-local padding and grouped sampling."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .multivideo import conditioned_features
from .skills import SKILLS

FEATURE_DIM = 122


def skill_features(observation,qpos,qvel,global_step,entry,segment,geometry,reference):
    lo,hi = segment['start'],segment['stop']
    if not lo <= global_step < hi or reference.shape != (entry['horizon'],30):
        raise ValueError('Reference clock or skill bounds mismatch')
    duration = float((geometry['source_frames'][-1]-geometry['source_frames'][0])/geometry['fps'])
    identity = np.zeros(4); identity[SKILLS.index(segment['skill'])]=1.
    base = conditioned_features(observation,qpos,qvel,global_step,entry['dt'],duration,
                                entry['control']['time_scale'],0 if entry['video']=='first' else 1)
    x = np.r_[base,identity,(global_step-lo)/max(hi-lo-1,1),geometry['object_poses'][-1,:3,3],reference[global_step]]
    if x.shape!=(FEATURE_DIM,) or not np.isfinite(x).all():
        raise ValueError('Invalid skill feature vector')
    return x


def fit_normalization(arrays):
    if not arrays:
        raise ValueError('Training arrays are required')
    x = np.concatenate(arrays)
    if x.ndim!=2 or not np.isfinite(x).all():
        raise ValueError('Finite training features required')
    mean,std = x.mean(axis=0),x.std(axis=0)
    std[std<1e-6]=1.
    return dict(mean=mean.tolist(),std=std.tolist(),fitted_frames=len(x),scope='training_split_only')


class SkillSequenceDataset:
    """No sequence crosses a skill boundary; padded positions have a false loss mask."""
    def __init__(self,index_path,split):
        path = Path(index_path)
        index = json.loads(path.read_text())
        if split not in ('train','validation'):
            raise ValueError('Unknown split')
        stats_path = path.parent/index['normalization']
        if hashlib.sha256(stats_path.read_bytes()).hexdigest()!=index['normalization_sha256']:
            raise ValueError('Normalization hash changed')
        stats = json.loads(stats_path.read_text())
        self.mean,self.std = np.asarray(stats['mean']),np.asarray(stats['std'])
        self.entries,self.arrays,self.groups = [],[],{}
        for record in index['records']:
            if record['split']!=split:
                continue
            file = path.parent/record['artifact']
            if hashlib.sha256(file.read_bytes()).hexdigest()!=record['sha256']:
                raise ValueError('Training input changed')
            with np.load(file) as saved:
                arrays = {key:saved[key].copy() for key in saved.files}
            group = (record['video'],record['skill'])
            self.groups.setdefault(group,[]).append(len(self.entries))
            self.entries.append(record); self.arrays.append(arrays)
        if not self.entries:
            raise ValueError('Empty split')
        self.cumulative = np.cumsum([len(a['inputs']) for a in self.arrays])

    def __len__(self): return int(self.cumulative[-1])

    def __getitem__(self,index):
        if index<0 or index>=len(self): raise IndexError(index)
        episode = int(np.searchsorted(self.cumulative,index,side='right'))
        previous = int(self.cumulative[episode-1]) if episode else 0
        return self.window(episode,index-previous,1)

    def window(self,episode,start,length):
        arrays = self.arrays[episode]
        count = len(arrays['inputs'])
        if length<1 or not 0<=start<count:
            raise ValueError('A sequence must start inside a skill')
        indices = np.minimum(start+np.arange(length),count-1)
        result = {key:value[indices].copy() for key,value in arrays.items()}
        result['inputs'] = ((result['inputs']-self.mean)/self.std).astype(np.float32)
        result['mask'] = (start+np.arange(length)<count)[:,None]
        result['record'] = self.entries[episode]
        return result

    def sample_batch(self,batch_size,sequence_length,seed):
        if batch_size<1: raise ValueError('Positive batch size required')
        rng = np.random.RandomState(seed)
        groups = sorted(self.groups)
        rows = []
        for _ in range(batch_size):
            group = groups[rng.randint(len(groups))]
            episode = int(rng.choice(self.groups[group]))
            rows.append(self.window(episode,int(rng.randint(len(self.arrays[episode]['inputs']))),sequence_length))
        batch = {key:np.stack([row[key] for row in rows]) for key in rows[0] if key!='record'}
        batch['records']=[row['record'] for row in rows]
        return batch
