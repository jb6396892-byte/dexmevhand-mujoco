"""Explicit physical admission criteria for the initial grasp curriculum."""


def physical_gate(report):
    return (
        report['hold_s'] >= 1.
        and report['tail_min_bottom_m'] > .015
        and report['tail_min_force_n'] > .05
        and report['tail_min_fingers'] >= 2
        and report['max_penetration_m'] <= .005
        and report['saturation'] < .1
        and report['final_distance_m'] < .1
    )
