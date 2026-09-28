#!/usr/bin/env python3
"""Plot recorded free-object trajectories before and after transport feedback."""
import argparse
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]/'data/processed'
    cases = [('First video', root/'seq_dexycb_001/scene_fidelity_v2/retarget/geometry.npz',
              root/'seq_dexycb_001/scene_fidelity_v3/approach_feedback/best', root/'seq_dexycb_001/transport_v6/best'),
             ('Second video', root/'seq_dexycb_002/retarget_v1/geometry.npz',
              root/'seq_dexycb_002/opposition_verified_v3/best', root/'seq_dexycb_002/transport_v5/best')]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex='col')
    for column, (name, geometry, before, after) in enumerate(cases):
        goal = np.load(geometry)['object_poses'][-1, :3, 3]
        for label, directory, color in [('Before', before, '#b45309'), ('Transport feedback', after, '#047857')]:
            with (directory/'diagnostic_rollout.pkl').open('rb') as stream:
                demo = pickle.load(stream)['video_faithful']
            positions = np.stack([state['qpos'][30:33] for state in demo['sim_data']])
            times = np.arange(len(positions))*.01
            axes[0, column].plot(times, np.linalg.norm(positions-goal, axis=1)*1000, label=label, color=color)
            axes[1, column].plot(times, positions[:, 2]*1000, color=color)
        axes[0, column].axhline(20., color='#6b7280', linestyle='--', label='20 mm threshold')
        axes[0, column].set_title(name)
        axes[0, column].set_ylabel('Distance to final target (mm)')
        axes[1, column].set_ylabel('Mug center height (mm)')
        axes[1, column].set_xlabel('Simulation time (s)')
        for ax in axes[:, column]:
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
