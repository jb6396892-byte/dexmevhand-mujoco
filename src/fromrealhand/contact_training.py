"""Auditable contact costs and bounded updates around the legacy DAPG agent."""
import numpy as np


def contact_costs(step_metrics, configuration):
    values=[]
    for row in step_metrics:
        if not np.isfinite([row[k] for k in ('penetration_m','source_frame','th_force_n','ff_force_n',
                                             'mf_force_n','rf_force_n','lf_force_n')]).all():
            raise ValueError('Nonfinite contact measurement')
        depth=max(0.,float(row['penetration_m'])-configuration['penetration_target_m'])
        cost=configuration['penetration_weight']*min(5.,depth/configuration['penetration_scale_m'])**2
        if row['source_frame']>=configuration['contact_phase_source_frame']:
            support=sum(row[f+'_force_n']>.01 for f in ('th','ff','mf','rf','lf'))
            cost+=configuration['missing_support_weight']*(max(0,3-support)+(row['th_force_n']<=.01))
        values.append(cost)
    result=np.asarray(values)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite contact cost')
    return result


def bounded_update(agent, paths, max_kl):
    policy=agent.policy
    before=policy.get_param_values().copy()
    statistics=agent.train_from_paths(paths)
    proposed=policy.get_param_values().copy()
    obs=np.concatenate([p['observations'] for p in paths])
    actions=np.concatenate([p['actions'] for p in paths])
    policy.set_param_values(before,set_new=False,set_old=True)
    fraction=1.; measured=float('inf')
    for _ in range(16):
        candidate=before+fraction*(proposed-before)
        policy.set_param_values(candidate,set_new=True,set_old=False)
        measured=float(agent.kl_old_new(obs,actions).detach().numpy().ravel()[0])
        if np.isfinite(measured) and measured<=max_kl: break
        fraction*=.5
    if not np.isfinite(measured) or measured>max_kl:
        policy.set_param_values(before)
        raise RuntimeError('Unable to enforce measured KL bound')
    policy.set_param_values(policy.get_param_values())
    return statistics,dict(raw_kl=float(agent.logger.log['kl_dist'][-1]),measured_kl=measured,
         accepted_fraction=fraction,parameter_delta_l2=float(np.linalg.norm(policy.get_param_values()-before)))
