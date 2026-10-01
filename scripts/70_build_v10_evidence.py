#!/usr/bin/env python3
"""Export small reproducible figures and metrics for the v10 presentation."""
import json
import pickle
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'data/processed/dual_video_v10'
OUT=ROOT/'docs/presentation/v10'


def read(path):
    return json.loads(path.read_text())


def compact(report):
    keys=['surface_physics_passed','fidelity_passed','final_distance_m','max_hand_scene_penetration_m',
          'initial_hand_scene_penetration_m','max_loaded_gap_m','tail_slip_m','saturation',
          'tail_mean_tip_error_m','tail_finger_contact_fraction','tail_finger_force_n','phase_contacts']
    result={k:report[k] for k in keys if k in report}
    if 'scene_contact_peaks' in report:
        result['largest_contact_peaks']=report['scene_contact_peaks'][:5]
    return result


def write(name,value):
    (OUT/'evidence'/name).write_text(json.dumps(value,indent=2)+'\n')


def main():
    (OUT/'evidence').mkdir(parents=True,exist_ok=True)
    (OUT/'assets').mkdir(parents=True,exist_ok=True)
    a=read(RUN/'development/admission.json')
    rows=[]
    for r in a['reports']:
        rows.append(dict(case=r['case'],admitted=r['admitted'],selected=r.get('selected'),
                         replay_error=r.get('replay_error'),candidates=[dict(name=c['name'],report=compact(c['report'])) for c in r['candidates']],
                         replay=compact(r.get('replay',{})),half_timestep=compact(r.get('half_timestep',{}))))
    write('development.json',dict(protocol_sha256=a['protocol_sha256'],admitted_count=a['admitted_count'],
                                 planned_count=a['planned_count'],demo_sha256=a['demo_sha256'],reports=rows,
                                 yaw_repair_optimizer=read(RUN/'development/yaw_plus/repair/optimization.json')))
    meta=read(RUN/'training_input/metadata.json')
    write('training-input.json',meta)
    for step in [0,828,1449]:
        source=RUN/'v9_cpu_screenshots'/('step_%04d.jpg'%step)
        if source.exists(): shutil.copy2(str(source),str(OUT/'assets'/('v9-step-%04d.jpg'%step)))
    write('v9-render-audit.json',read(RUN/'v9_cpu_screenshots/comparison.json'))
    x=np.arange(len(rows));labels=[r['case']['name'] for r in rows]
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for ax,key,factor,title,limit in [(axes[0],'final_distance_m',1000,'Goal error (mm)',20),
                                    (axes[1],'max_hand_scene_penetration_m',1000,'Max hand-scene penetration (mm)',1)]:
        ax.bar(x-.18,[r['candidates'][0]['report'][key]*factor for r in rows],.36,label='Frozen v9 actions',color='#C36B38')
        ax.bar(x+.18,[r['replay'][key]*factor for r in rows],.36,label='Admitted actions',color='#278477')
        ax.axhline(limit,color='#B53548',linestyle='--',label='Gate')
        ax.set_ylabel(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend(ncol=3);axes[1].set_xticks(x);axes[1].set_xticklabels(labels,rotation=30,ha='right')
    fig.suptitle('Second video: development repair (11/11 admitted, replay + half timestep)')
    fig.tight_layout();fig.savefig(str(OUT/'assets/development.png'),dpi=150);plt.close(fig)
    frozen_path=RUN/'learning/frozen_policies.json'
    if not frozen_path.exists(): return
    frozen=read(frozen_path);write('frozen-policies.json',frozen)
    comparisons={}
    fig,ax=plt.subplots(figsize=(8,4))
    for method in ['direct_bc','residual_bc']:
        entries=read(RUN/'learning'/method/'comparison.json')
        comparisons[method]=[{**{k:v for k,v in e.items() if k!='reports'},
                             'reports':[dict(video=r['video'],case=r['case'],full_pass=r['full_pass'],lift_pass=r['lift_pass'],
                                             failure_labels=r['failure_labels'],phases=r['phases'],report=compact(r['report'])) for r in e['reports']]} for e in entries]
        losses=read(RUN/'learning'/method/'losses.json')
        ax.semilogy([r['epoch'] for r in losses],np.sqrt([r['balanced_action_mse'] for r in losses]),label=method)
    ax.set_xlabel('BC epoch');ax.set_ylabel('Balanced offline action RMSE');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(str(OUT/'assets/bc-loss.png'),dpi=150);plt.close(fig)
    write('development-policy-comparison.json',comparisons)
    protocol=read(ROOT/'configs/v10-study.json')
    fig,axes=plt.subplots(2,2,figsize=(11,7));divergences=[]
    for i,video in enumerate(protocol['videos']):
        nominal='video_seed_0' if i==0 else 'nominal'
        paths={'expert':ROOT/video['paths']['rollout']}
        for method,entry in frozen['policies'].items():
            paths[method]=RUN/'learning'/method/('epoch_%03d'%entry['epoch'])/video['name']/nominal/'diagnostic_rollout.pkl'
        geometry=np.load(ROOT/video['paths']['geometry']);goal=geometry['object_poses'][-1,:3,3]
        expert=pickle.loads(paths['expert'].read_bytes())['video_faithful']
        expert_q=np.array([s['qpos'] for s in expert['sim_data']])
        for method,path in paths.items():
            demo=pickle.loads(path.read_bytes())['video_faithful']
            q=np.array([s['qpos'] for s in demo['sim_data']]);pos=q[:,30:33];t=np.arange(len(pos))*.01
            axes[i,0].plot(t,pos[:,2]*1000,label=method)
            axes[i,1].plot(t,np.linalg.norm(pos-goal,axis=1)*1000,label=method)
            if method!='expert':
                errors=dict(action_rmse=np.sqrt(np.mean((demo['actions']-expert['actions'])**2,axis=1)),
                            root_translation_m=np.linalg.norm(q[:,:3]-expert_q[:,:3],axis=1),
                            cup_position_m=np.linalg.norm(q[:,30:33]-expert_q[:,30:33],axis=1))
                crossings={}
                for key,error in errors.items():
                    bad=np.flatnonzero(error>.01)
                    crossings[key]=dict(threshold=.01,first_step=int(bad[0]) if len(bad) else None,
                                        time_s=float(bad[0]*.01) if len(bad) else None,initial_error=float(error[0]))
                divergences.append(dict(video=video['name'],method=method,epoch=frozen['policies'][method]['epoch'],crossings=crossings))
        for j in range(2):
            axes[i,j].set_xlabel('Control time (s)');axes[i,j].set_title(video['name']+' video: development nominal')
            axes[i,j].grid(alpha=.2)
            for frame in [30,40,55]: axes[i,j].axvline(.5+frame/30.*5.,color='gray',linestyle=':',alpha=.5)
        axes[i,0].set_ylabel('Cup center height (mm)');axes[i,1].set_ylabel('Goal distance (mm)')
    axes[0,0].legend();fig.tight_layout();fig.savefig(str(OUT/'assets/development-trajectories.png'),dpi=150);plt.close(fig)
    write('selected-policy-divergence.json',divergences)
    test_path=RUN/'heldout/summary.json'
    if not test_path.exists(): return
    test=read(test_path)
    write('heldout.json',{**{k:v for k,v in test.items() if k!='reports'},
                         'reports':[dict(video=r['video'],case=r['case'],method=r['method'],full_pass=r['full_pass'],lift_pass=r['lift_pass'],
                                         failure_labels=r['failure_labels'],phases=r['phases'],report=compact(r['report'])) for r in test['reports']]})
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    methods=['fixed','direct_bc','residual_bc'];colors=['#777777','#C36B38','#278477']
    for ax,key,title in [(axes[0],'full_count','Full gate success'),(axes[1],'lift_count','Stable lift only (not full success)')]:
        values=[test['summary'][m][key] for m in methods]
        ax.bar(methods,values,color=colors);ax.set_ylim(0,22);ax.set_title(title);ax.set_ylabel('Cases / 20')
        for i,v in enumerate(values): ax.text(i,v+.3,str(v)+'/20',ha='center')
    fig.tight_layout();fig.savefig(str(OUT/'assets/heldout.png'),dpi=150);plt.close(fig)
    print(json.dumps(test['summary'],indent=2))


if __name__=='__main__': main()
