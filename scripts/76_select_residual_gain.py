#!/usr/bin/env python3
"""Development-only residual attenuation; retain the original candidate freeze."""
import json
import pickle
from importlib import import_module
from pathlib import Path
from v11_common import RUN,protocol
from v10_common import ROOT,digest

learning=import_module('72_train_corrective_residual')


def scale_residual(checkpoint,gain):
    if not 0 < gain <= 1:
        raise ValueError('Residual gain must be in (0,1]')
    if checkpoint['method']!='residual_bc':
        raise ValueError('Only a residual policy can be attenuated')
    for model in (checkpoint['policy'].model,checkpoint['policy'].old_model):
        model.set_transformations(model.in_shift.detach().cpu().numpy(),model.in_scale.detach().cpu().numpy(),
                                  model.out_shift.detach().cpu().numpy()*gain,model.out_scale.detach().cpu().numpy()*gain)
    checkpoint['residual_output_gain']=gain
    return checkpoint


def main():
    study,parent,sha=protocol()
    amendment_path=ROOT/'configs/v11-residual-gain-amendment.json'
    amendment=json.loads(amendment_path.read_text())
    if amendment['parent_protocol_sha256']!=sha: raise ValueError('Amendment protocol mismatch')
    if (RUN/'heldout').exists(): raise RuntimeError('Cannot select a candidate after new testing begins')
    folder=RUN/'learning'
    base_path=folder/'frozen_policy.json';base=json.loads(base_path.read_text())
    if digest(base['selected']['policy'])!=base['selected']['policy_sha256']: raise ValueError('Base model changed')
    output=folder/'safety_frozen_policy.json'
    if output.exists() or (folder/'safety_comparison.json').exists(): raise FileExistsError('Do not overwrite gain study')
    admission=json.loads((RUN/'experts/admission.json').read_text())
    demos=pickle.loads((RUN/'experts/demonstrations.pkl').read_bytes())
    comparisons=[]
    for gain in amendment['additional_residual_gains']:
        cp=scale_residual(pickle.loads(Path(base['selected']['policy']).read_bytes()),gain)
        label='residual_gain_%03d'%round(gain*100)
        cp['training_label']=label;cp['base_policy_sha256']=base['selected']['policy_sha256']
        cp['amendment_sha256']=digest(amendment_path)
        result=learning.evaluate(cp,label,folder,admission['reports'],demos,parent,study['safety']['preferred_penetration_m'])
        result['residual_gain']=gain
        comparisons.append(result)
    (folder/'safety_comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    best=min([base['selected']]+comparisons,key=learning.rank)
    result={**base,'selected':{k:v for k,v in best.items() if k!='reports'},
            'base_freeze_sha256':digest(base_path),'amendment_sha256':digest(amendment_path),
            'residual_gain':best.get('residual_gain',1.),'heldout_evaluated':False}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print('FROZEN_GAIN',json.dumps(result),flush=True)


if __name__=='__main__': main()
