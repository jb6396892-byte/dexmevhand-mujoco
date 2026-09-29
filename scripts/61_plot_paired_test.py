#!/usr/bin/env python3
"""Plot every frozen held-out pair, including regressions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source = json.loads(args.evaluation.read_text())
    if source['completed_cases'] != source['planned_cases']:
        raise ValueError('Incomplete evaluation')
    rows = source['reports']
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12, 5.4))
    for key, offset, color, label in [('nominal_actions', -.19, '#70777b', 'Fixed reference'),
                                       ('student', .19, '#14836c', 'Learned residual')]:
        ax.bar(x+offset, [1000*r[key]['final_distance_m'] for r in rows], .38, color=color, label=label)
    ax.axhline(20, color='#be4141', linestyle='--', label='20 mm acceptance')
    ax.set_xticks(x)
    ax.set_xticklabels([r['name'].replace('test_', '').replace('_', '\n') for r in rows], fontsize=9)
    ax.set_ylabel('Final target error (mm)')
    ax.set_title('Frozen v6 policy: 10 held-out synthetic scenes, no retraining')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 31)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    if args.output.exists():
        raise FileExistsError(args.output)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
