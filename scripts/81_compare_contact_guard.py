#!/usr/bin/env python3
"""Three preregistered geometric unloading gains; development only."""
import importlib
import json
import pickle
import copy
study=importlib.import_module('77_run_v12_study')


def main():
    if (study.RUN/'heldout').exists(): raise RuntimeError('No tuning after heldout')
    amendment=json.loads((study.ROOT/'configs/v12-contact-guard-amendment.json').read_text())
    original=pickle.loads((study.RUN/'aligned_bc/policy.pickle').read_bytes())
    for gain in amendment['gains']:
        mode='guard_%03d'%round(gain*100);folder=study.RUN/mode
        if not folder.exists():
            folder.mkdir()
            cp=copy.deepcopy(original)
            cp['contact_guard']={k:amendment[k] for k in ['target_depth_m','max_joint_delta_rad','filter_alpha']}
            cp['contact_guard']['gain']=gain
            cp['training_label']=mode
            with (folder/'policy.pickle').open('xb') as stream: pickle.dump(cp,stream)
        study.develop(mode)


if __name__=='__main__': main()
