#!/usr/bin/env python3
"""Validate early-closure takeovers before admitting on-policy expert labels."""
import argparse
import copy
import importlib
import json
import pickle
from pathlib import Path
import numpy as np
from v10_common import digest, run_case
from fromrealhand.corrective_learning import aligned_phase_indices
from fromrealhand.multivideo import phase_diagnostics
from fromrealhand.routed_residual import TakeoverActions, RoutedNetwork

study = importlib.import_module('83_run_v13_routing')
RUN, PLAN, write_json = study.RUN, study.PLAN, study.write_json


def collect():
    plan, old, parent, admission, demos, refs, base, data = study.inputs()
    spec = plan['correction']
    choice = json.loads((RUN/'comparison.json').read_text())
    if choice['protocol_sha256'] != digest(PLAN):
        raise ValueError('Changed protocol')
    selected = choice['selected']
    path = Path(selected['path'])
    if digest(path) != selected['sha256']:
        raise ValueError('Changed student')
    checkpoint = pickle.loads(path.read_bytes())
    dev = json.loads((RUN/selected['mode']/'development.json').read_text())
    cases = [r for r in dev['reports'] if r['video'] == 'second' and not r['strict_pass']]
    cases.sort(key=lambda r: (r['task_pass'], -r['phases']['closure']['cup_position_rmse_vs_expert_m']))
    cases = cases[:spec['max_cases']]
    folder = RUN/'corrections'
    folder.mkdir(exist_ok=False)
    write_json(folder/'selection.json', dict(protocol_sha256=digest(PLAN), student=selected,
        case_names=[r['case'] for r in cases], source='development only',
        order='task failures first, then descending closure cup RMSE', heldout_used=False))
    records = []
    for row in cases:
        entry = next(e for e in admission['reports'] if e['video'] == 'second' and e['name'] == row['case'])
        video = parent['videos'][entry['video_id']]
        geometry = np.load(entry['geometry'])
        phase = aligned_phase_indices(video['horizon'], geometry, video['control']['time_scale'])
        closure = int(np.flatnonzero(phase == 2)[0])
        lift = int(np.flatnonzero(phase == 3)[0])
        expert = demos['second/'+entry['name']]
        student_rollout = pickle.loads((RUN/selected['mode']/'rollouts/second'/(entry['name']+'.pkl')).read_bytes())['video_faithful']
        history = []
        accepted = None
        for offset in spec['takeover_offsets_steps']:
            for gain in spec['gains']:
                controllers = []
                start = max(0, closure+offset)
                end = min(video['horizon'], lift+spec['label_end_steps_after_lift'])
                def factory(exp):
                    c = TakeoverActions(checkpoint, exp, video, expert, start,
                                        spec['transition_steps'], gain, end)
                    controllers.append(c)
                    return c
                report, rollout = run_case(video, dict(geometry=entry['geometry']), None,
                                            seed=entry['seed'], action_factory=factory)
                prefix_action_error = float(np.max(np.abs(rollout['actions'][:start+1]-student_rollout['actions'][:start+1])))
                prefix_observation_error = float(np.max(np.abs(rollout['observations'][:start+1]-student_rollout['observations'][:start+1])))
                if max(prefix_action_error, prefix_observation_error) > 1e-8:
                    raise RuntimeError('Takeover must start from the unchanged student prefix')
                status = dict(start_step=start, closure_step=closure, lift_step=lift, gain=gain,
                              report=report, **study.study.gates(report, old), admitted=False,
                              student_prefix_action_error=prefix_action_error,
                              student_prefix_observation_error=prefix_observation_error)
                if status['task_pass']:
                    replay, rd = study.execute(checkpoint, video, entry, actions=rollout['actions'])
                    half, _ = study.execute(checkpoint, video, entry, half=True, actions=rollout['actions'])
                    error = float(np.max(np.abs(rd['observations']-rollout['observations'])))
                    status.update(replay_error=error, replay=study.study.gates(replay, old),
                                  half=study.study.gates(half, old), half_report=half)
                    if status['replay']['task_pass'] and status['half']['task_pass'] and error < spec['require_replay_error_below']:
                        c = controllers[0]
                        steps = np.asarray(c.steps)
                        features, labels = np.asarray(c.features), np.asarray(c.labels)
                        if not len(steps):
                            raise RuntimeError('No fully executed expert labels')
                        label_error = float(np.max(np.abs(labels+checkpoint['references']['second'][steps]-rollout['actions'][steps])))
                        if label_error > 1e-12:
                            raise RuntimeError('Unexecuted advice cannot be labeled as demonstrated action')
                        q = np.array([s['qpos'] for s in rollout['sim_data']])[steps]
                        target = np.array([s['qpos'] for s in expert['sim_data']])[steps]
                        destination = folder/entry['name']
                        destination.mkdir()
                        np.savez_compressed(destination/'labels.npz', features=features, labels=labels,
                            steps=steps, feedback=np.asarray(c.feedback), video_id=video['id'])
                        with (destination/'rollout.pkl').open('xb') as stream:
                            pickle.dump({'video_faithful': rollout}, stream)
                        status.update(admitted=True, label_frames=len(steps), label_action_error=label_error,
                            mean_cup_deviation_from_expert_m=float(np.linalg.norm(q[:,30:33]-target[:,30:33], axis=1).mean()),
                            mean_hand_joint_deviation_from_expert_rad=float(np.abs(q[:,6:30]-target[:,6:30]).mean()),
                            phases=phase_diagnostics(rollout, checkpoint['references']['second'], video, geometry, expert),
                            labels=str(destination/'labels.npz'), labels_sha256=digest(destination/'labels.npz'))
                        accepted = status
                history.append(status)
                write_json(folder/(entry['name']+'-attempts.json'), history)
                print('TAKEOVER', entry['name'], start, gain, status['task_pass'], status['strict_pass'],
                      status['admitted'], round(report['max_hand_scene_penetration_m']*1000, 3), flush=True)
                if accepted is not None:
                    break
            if accepted is not None:
                break
        records.append(dict(case=entry['name'], before=row, accepted=accepted, attempts=len(history)))
        write_json(folder/'admission.json', dict(protocol_sha256=digest(PLAN), student=selected,
            completed=len(records)==len(cases), reports=records,
            accepted_cases=sum(r['accepted'] is not None for r in records),
            label_frames=sum(r['accepted']['label_frames'] for r in records if r['accepted']),
            heldout_used=False, long_training=False))


def finetune():
    plan, old, parent, admission, demos, refs, base, data = study.inputs()
    spec = plan['correction']
    record = json.loads((RUN/'corrections/admission.json').read_text())
    if not record['completed'] or not record['accepted_cases']:
        raise ValueError('No completed, physically admitted corrective data')
    if record['protocol_sha256'] != digest(PLAN):
        raise ValueError('Changed protocol')
    source = Path(record['student']['path'])
    if digest(source) != record['student']['sha256']:
        raise ValueError('Changed student')
    original = pickle.loads(source.read_bytes())
    if 'routing_mode' not in original:
        original = copy.deepcopy(original)
        original.update(routing_mode='shared', network=RoutedNetwork(original.pop('policy').model, 1),
                        routes={v['name']:np.ones((v['horizon'], 1)) for v in parent['videos']})
    x, y, w = [data[k] for k in ('features', 'labels', 'weights')]
    routes = study.batch_routes(original, admission)
    cx, cy, cr, cw = [], [], [], []
    for row in record['reports']:
        a = row['accepted']
        if a is None:
            continue
        if digest(a['labels']) != a['labels_sha256']:
            raise ValueError('Changed corrective labels')
        labels = np.load(a['labels'])
        cx.append(labels['features']); cy.append(labels['labels'])
        cr.append(original['routes']['second'][labels['steps']])
        cw.append(np.full(len(labels['steps']), 1./record['accepted_cases']/len(labels['steps'])))
    for mode in ('base_finetune', 'corrected'):
        folder = RUN/mode
        folder.mkdir(exist_ok=False)
        checkpoint = copy.deepcopy(original)
        xx, yy, ww, rr = x, y, w, routes
        if mode == 'corrected':
            xx, yy, rr = [np.concatenate([a]+b) for a, b in ((x,cx), (y,cy), (routes,cr))]
            ww = np.r_[w*(1.-spec['correction_mass']), np.concatenate(cw)*spec['correction_mass']]
        # Equal draw count, even though the corrective dataset has additional rows.
        fit_fixed_budget(checkpoint['network'], xx, yy, ww, rr, len(x), plan, folder/'training')
        checkpoint.update(protocol_sha256=digest(PLAN), training_label='v13_'+mode)
        with (folder/'policy.pickle').open('xb') as stream:
            pickle.dump(checkpoint, stream)
        write_json(folder/'input.json', dict(protocol_sha256=digest(PLAN), parent_sha256=digest(source),
            correction_admission_sha256=digest(RUN/'corrections/admission.json'),
            epochs=spec['finetune_epochs'], samples_per_epoch=len(x), frames=len(xx),
            correction_mass=spec['correction_mass'] if mode == 'corrected' else 0., heldout_used=False))
        print('FINETUNED', mode, flush=True)


def fit_fixed_budget(network, x, y, weights, routes, count, plan, folder):
    import torch
    spec = plan['correction']
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable')
    torch.set_num_threads(1)
    network.to_runtime('cuda')
    optimizer = torch.optim.Adam(network.parameters(), lr=spec['finetune_learning_rate'])
    x, y, routes = [torch.as_tensor(a, dtype=torch.float32, device='cuda') for a in (x, y, routes)]
    rng = np.random.RandomState(plan['seed'])
    losses = []
    for epoch in range(1, spec['finetune_epochs']+1):
        ids = rng.choice(len(x), size=count, replace=True, p=weights/weights.sum())
        values = []
        for start in range(0, count, plan['batch_size']):
            j = torch.as_tensor(ids[start:start+plan['batch_size']], device='cuda')
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(network(x[j], routes[j]), y[j])
            loss.backward(); optimizer.step()
            values.append(float(loss.detach().cpu()))
        if not np.isfinite(values).all():
            raise RuntimeError('Nonfinite correction training')
        losses.append(dict(epoch=epoch, sampled_mse=float(np.mean(values))))
        if epoch % 5 == 0:
            print('FINETUNE', folder.parent.name, epoch, losses[-1]['sampled_mse'], flush=True)
    network.to_runtime('cpu')
    write_json(folder/'losses.json', losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['collect', 'finetune'])
    args = parser.parse_args()
    collect() if args.command == 'collect' else finetune()
