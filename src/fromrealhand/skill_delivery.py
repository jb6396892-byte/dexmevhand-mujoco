"""Small, auditable qualification summary independent of the simulator."""
from collections import Counter


def failure_reason(case):
    if case['passed']:
        return None
    for contract in case['contracts']:
        if contract['status'] in ('failed', 'timeout'):
            return contract['reason'] or contract['status']
    if not case['complete'] or any(c['status'] != 'success' for c in case['contracts']):
        return 'reference_exhausted_before_event'
    return 'state_replay_mismatch'


def entry_summary(cases):
    groups = {}
    for case in cases:
        kind = 'perturbed' if any(case['offset_m']) else 'nominal'
        key = case['trajectory'].split('/')[0] + '/' + case['skills'][0] + '/' + kind
        group = groups.setdefault(key, dict(total=0, passed=0, failures={}))
        group['total'] += 1
        group['passed'] += int(case['passed'])
        reason = failure_reason(case)
        if reason:
            group['failures'][reason] = group['failures'].get(reason, 0) + 1
    failures = Counter(failure_reason(c) for c in cases if not c['passed'])
    return dict(groups=groups, failure_counts=dict(failures),
                robust_skill_ready=all(c['passed'] for c in cases),
                scope='Reference action sensitivity, not a learned policy evaluation')
