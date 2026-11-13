"""Re-run greedy evaluation + plots for saved two-robot models (no training, no updates).

Usage: python replot_two_robot.py seed7_60k seed11_60k ...
"""
import os
import sys

import torch

from dqn import QNetwork
from plotting import RESULTS_DIR, plot_two_robot_trajectory
from two_robot_dqn import format_rollout_table, greedy_rollout_two
from warehouse_env import GOAL_A, GOAL_B, N_JOINT_ACTIONS

for tag in sys.argv[1:]:
    net = QNetwork(50, N_JOINT_ACTIONS)
    net.load_state_dict(torch.load(os.path.join(RESULTS_DIR, f"dqn_two_robot_{tag}.pt")))
    for starts, suffix, label in [(((2, 0), (2, 4)), "", "Greedy two-robot rollout"),
                                  (((0, 0), (4, 4)), "extra_", "Two-robot rollout from A(0,0), B(4,4)")]:
        g = greedy_rollout_two(net, starts)
        status = "success" if g["success"] else ("collision" if g["collision"] else "timeout")
        plot_two_robot_trajectory(g["states"], f"{label} [{tag}] - {status} in {g['steps']} steps",
                                  os.path.join(RESULTS_DIR, f"two_robot_trajectory_{suffix}{tag}.png"),
                                  (GOAL_A, GOAL_B))
        print(tag, starts, status, g["steps"])
