#!/usr/bin/env python3
"""Export compact, traceable v11 evidence without copying private datasets."""
import json
import pickle
import shutil
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'data/processed/dual_video_v11'
OUT = ROOT/'docs/presentation/v11'


def read(path):
    return json.loads(path.read_text())


def compact(report):
    keys = ['surface_physics_passed', 'fidelity_passed', 'finite', 'final_distance_m',
            'max_hand_scene_penetration_m', 'initial_hand_scene_penetration_m',
            'max_loaded_gap_m', 'tail_slip_m', 'saturation', 'mean_tip_error_m',
            'tail_mean_tip_error_m', 'tail_finger_contact_fraction',
            'tail_finger_force_n', 'phase_contacts']
    result = {k: report[k] for k in keys if k in report}
    result['largest_contact_peaks'] = report.get('scene_contact_peaks', [])[:5]
    return result


def write(name, value):
    (OUT/'evidence'/name).write_text(json.dumps(value, indent=2)+'\n')


def policy_row(row):
    return {**{k: v for k, v in row.items() if k != 'report'}, 'report': compact(row['report'])}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(str(OUT/'assets'/name), dpi=150)
    plt.close(fig)


def main():
    (OUT/'evidence').mkdir(parents=True, exist_ok=True)
    (OUT/'assets').mkdir(parents=True, exist_ok=True)
    admission = read(RUN/'experts/admission.json')
    rows = []
    for row in admission['reports']:
        item = {k: v for k, v in row.items() if k not in ['candidates', 'report', 'half']}
        item['candidates'] = [dict(name=c['name'], report=compact(c['report'])) for c in row['candidates']]
        item['report'] = compact(row.get('report', {}))
        item['half'] = compact(row.get('half', {}))
        search=RUN/'experts'/row['video']/row['name']/'search'
        if (search/'safety_search.json').exists():
            log=read(search/'safety_search.json')
            item['search']={k:v for k,v in log.items() if k!='history'}
            item['search']['best']=read(search/'safety_best.json')
        rows.append(item)
    write('experts.json', {**{k: v for k, v in admission.items() if k != 'reports'}, 'reports': rows})
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    x = np.arange(len(rows))
    for ax, key, title, gate in [(axes[0], 'max_hand_scene_penetration_m', 'Peak penetration (mm)', 1),
                                (axes[1], 'final_distance_m', 'Final goal error (mm)', 20)]:
        ax.bar(x-.18, [r['candidates'][0]['report'][key]*1000 for r in rows], .36,
               label='Initial actions', color='#B56B42')
        ax.bar(x+.18, [r['report'].get(key, np.nan)*1000 for r in rows], .36,
               label='Admitted expert', color='#278477')
        ax.axhline(gate, color='#B53548', linestyle='--', label='Unchanged hard gate')
        ax.set_ylabel(title)
        ax.grid(axis='y', alpha=.2)
    axes[0].axhline(.75, color='#526A9B', linestyle=':', label='Preferred headroom')
    axes[0].scatter(x+.18,[r['half'].get('max_hand_scene_penetration_m',np.nan)*1000 for r in rows],
                    marker='x',s=18,color='black',zorder=4,label='Half timestep')
    axes[0].legend(ncol=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([r['video'][0]+':'+r['name'] for r in rows], rotation=70, ha='right', fontsize=7)
    fig.suptitle('Development experts: %d/%d admitted; %d/%d with preferred headroom' %
                 (admission['admitted_count'],admission['planned_count'],admission['preferred_count'],admission['planned_count']))
    save(fig, 'expert-safety.png')
    for label in ['expert_second', 'policy_first', 'policy_second', 'heldout_failure']:
        source = RUN/'renders'/label
        if not source.exists():
            continue
        for path in sorted(source.glob('step_*.jpg')):
            shutil.copy2(str(path), str(OUT/'assets'/(label+'-'+path.name)))
        if (source/'comparison.json').exists():
            write(label+'-render.json', read(source/'comparison.json'))
    learning = RUN/'learning'
    if not (learning/'frozen_policy.json').exists():
        return
    # This exact-byte receipt must be committed BEFORE the heldout evaluator runs.
    shutil.copy2(str(learning/'frozen_policy.json'), str(OUT/'evidence/frozen-policy.json'))
    for name in ['input.json', 'teacher_pilot.json']:
        value = read(learning/name)
        if name == 'teacher_pilot.json':
            for item in value['results']:
                item['reports'] = [compact(r) for r in item['reports']]
        write(name.replace('_', '-'), value)
    collections = read(learning/'collections.json')
    for collection in collections:
        collection['rows'] = [policy_row(r) for r in collection['rows']]
        collection['accepted_by_video'] = dict(Counter(r['video'] for r in collection['rows'] if r['admitted_labels']))
    write('corrections.json', collections)
    comparisons = read(learning/'comparison.json')
    for item in comparisons:
        item['reports'] = [policy_row(r) for r in item['reports']]
    write('development-policies.json', comparisons)
    fig, ax = plt.subplots(figsize=(8, 4))
    for path in sorted(learning.glob('*_training/losses.json')):
        losses = read(path)
        ax.semilogy([r['epoch'] for r in losses], np.sqrt([r['weighted_mse'] for r in losses]), label=path.parent.name)
    ax.set_xlabel('Epoch within run')
    ax.set_ylabel('Weighted offline action RMSE (not physical success)')
    ax.grid(alpha=.2)
    ax.legend()
    save(fig, 'learning-loss.png')
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [r['label'] for r in comparisons]
    ax.bar(labels, [r['full_fraction']*100 for r in comparisons], color='#278477')
    ax.set_ylim(0, 105)
    ax.set_ylabel('Equal-video full-gate success (%)')
    for i, r in enumerate(comparisons):
        ax.text(i, r['full_fraction']*100+1, '%d/%d' % (r['full_count'], r['case_count']), ha='center')
    ax.set_title('Development selection; not independent test performance')
    save(fig, 'development-policies.png')
    frozen = read(learning/'frozen_policy.json')
    selected = next(r for r in comparisons if r['label'] == frozen['selected']['label'])
    refs = pickle.loads((RUN/'experts/references.pkl').read_bytes())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for i, name in enumerate(['first', 'second']):
        case = 'video_seed_0' if i == 0 else 'nominal'
        row = next(r for r in selected['reports'] if r['video'] == name and r['case'] == case)
        entry = next(r for r in admission['reports'] if r['video'] == name and r['name'] == case)
        goal = np.load(entry['geometry'])['object_poses'][-1, :3, 3]
        policy = pickle.loads(Path(row['rollout']).read_bytes())['video_faithful']
        for label, demo in [('Expert', refs[name]), ('Frozen residual', policy)]:
            q = np.array([s['qpos'] for s in demo['sim_data']])
            t = np.arange(len(q))*.01
            axes[i, 0].plot(t, q[:, 32]*1000, label=label)
            axes[i, 1].plot(t, np.linalg.norm(q[:, 30:33]-goal, axis=1)*1000, label=label)
        for ax in axes[i]:
            ax.set_title(name+' video: development nominal')
            ax.set_xlabel('Control time (s)')
            ax.grid(alpha=.2)
            for boundary in [5.5, 7.1666666667, 9.6666666667]:
                ax.axvline(boundary, color='gray', alpha=.5, linestyle=':')
        axes[i, 0].set_ylabel('Cup center height (mm)')
        axes[i, 1].set_ylabel('Goal distance (mm)')
    axes[0, 0].legend()
    save(fig, 'phase-trajectories.png')
    force_files=[RUN/'renders'/name/'finger_forces.npy' for name in ['expert_second','policy_second']]
    if all(p.exists() for p in force_files):
        fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,sharey=True)
        for ax,path,title in zip(axes,force_files,['Admitted expert','Frozen residual policy']):
            forces=np.load(path)
            for i,finger in enumerate(['Thumb','Index','Middle','Ring','Little']):
                ax.plot(np.arange(len(forces))*.01,forces[:,i],label=finger,linewidth=1)
            ax.set_title(title);ax.set_ylabel('Simulated normal force (N)');ax.grid(alpha=.2)
        axes[0].legend(ncol=5);axes[1].set_xlabel('Control time (s)')
        fig.suptitle('Second-video nominal: simulated forces, not measured human forces')
        save(fig,'second-contact-forces.png')
    if not (RUN/'heldout/summary.json').exists():
        return
    test = read(RUN/'heldout/summary.json')
    test['reports'] = [policy_row(r) for r in test['reports']]
    write('heldout.json', test)
    write('freeze-receipt.json', read(RUN/'heldout/freeze_receipt.json'))
    methods = ['fixed_v10', 'residual_v10', 'fixed_v11', 'residual_v11']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in zip(axes, ['full_count', 'preferred_count', 'mean_goal_mm'],
                              ['Full-gate passes / 24', 'Full gate + <=0.75 mm / 24', 'Mean goal error (mm)']):
        vals = [test['summary'][m][key] for m in methods]
        ax.bar(methods, vals, color=['#777777', '#B56B42', '#526A9B', '#278477'])
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=25)
        if key != 'mean_goal_mm':
            ax.set_ylim(0, 27)
        for i, value in enumerate(vals):
            ax.text(i, value+.25, '%.2f' % value if key == 'mean_goal_mm' else str(value), ha='center')
    save(fig, 'heldout-comparison.png')
    print(json.dumps(test['summary'], indent=2))


if __name__ == '__main__':
    main()
