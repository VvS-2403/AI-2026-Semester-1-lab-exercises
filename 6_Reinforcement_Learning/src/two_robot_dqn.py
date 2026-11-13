"""
Part III - Two robots, one central DQN controller (Task 7).

State  : (pA, pB) encoded as two concatenated length-25 one-hots (50 inputs).
Action : joint action j = 5*a_A + a_B (25 linear outputs); decoded with divmod(j, 5).
Reward : -10 on attempted robot-robot collision, else -0.1 - b + 5k + 5I.
Termination on collision or when both robots are at their goals; truncation at 40 joint steps.

No path planner and no collision-filtering of actions: the greedy policy is
evaluated on exactly the joint actions the network chooses.

Run:  python two_robot_dqn.py                 (seed 7, 60,000 env steps)
      python two_robot_dqn.py --seed 11 --steps 100000
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import numpy as np
import torch

from dqn import DQNConfig, train_dqn
from plotting import RESULTS_DIR, plot_training_curves, plot_two_robot_trajectory
from warehouse_env import (ACTION_NAMES, GOAL_A, GOAL_B, N_JOINT_ACTIONS, START_A, START_B,
                           TwoRobotWarehouseEnv, decode_joint_action, two_robot_one_hot)


def greedy_rollout_two(online, starts=(START_A, START_B)):
    env = TwoRobotWarehouseEnv()
    state = env.reset(starts)
    states, rows = [state], []
    invalid, ret, info = 0, 0.0, {}
    online.eval()
    while True:
        with torch.no_grad():
            q = online(torch.from_numpy(two_robot_one_hot(state)).unsqueeze(0)).squeeze(0)
        j = int(torch.argmax(q).item())
        a_A, a_B = decode_joint_action(j)
        old = state
        state, r, terminated, truncated, info = env.step(j)
        states.append(state)
        invalid += info["invalid"]
        ret += r
        rows.append((env.steps, old[0], old[1], ACTION_NAMES[a_A], ACTION_NAMES[a_B],
                     state[0], state[1], r, "COLLISION" if info["collision"] else ""))
        if terminated or truncated:
            break
    success = terminated and not info["collision"]
    return {"states": states, "rows": rows, "success": success, "collision": info["collision"],
            "timeout": truncated, "steps": env.steps, "invalid": invalid, "return": ret,
            "at_goal": env.at_goal()}


def format_rollout_table(res) -> str:
    lines = [f"{'t':>3}  {'A old':>7} {'B old':>7}  {'A act':>6} {'B act':>6}  "
             f"{'A new':>7} {'B new':>7}  {'reward':>7}"]
    for t, oA, oB, aA, aB, nA, nB, r, note in res["rows"]:
        lines.append(f"{t:>3}  {str(oA):>7} {str(oB):>7}  {aA:>6} {aB:>6}  "
                     f"{str(nA):>7} {str(nB):>7}  {r:>7.2f} {note}")
    status = "SUCCESS" if res["success"] else ("COLLISION" if res["collision"] else
                                               "TIMEOUT" if res["timeout"] else "?")
    lines.append(f"=> {status}: joint steps={res['steps']}, invalid moves={res['invalid']}, "
                 f"team return={res['return']:.2f}, at goal (A,B)={res['at_goal']}")
    return "\n".join(lines)


def run(seed: int, steps: int, tag: str | None = None):
    tag = tag or f"seed{seed}_{steps // 1000}k"
    out = []
    p = lambda s="": (print(s), out.append(s))  # noqa: E731
    cfg = DQNConfig(total_steps=steps, seed=seed)
    p(f"=== Part III: two-robot DQN [{tag}] ===")
    p(f"config: {json.dumps(asdict(cfg))}")
    env = TwoRobotWarehouseEnv()
    online, log = train_dqn(env, two_robot_one_hot, 50, N_JOINT_ACTIONS, cfg,
                            extra_info_keys=("collision",))
    torch.save(online.state_dict(), os.path.join(RESULTS_DIR, f"dqn_two_robot_{tag}.pt"))

    n_ep = len(log["return"])
    last = slice(-200, None)
    p(f"episodes: {n_ep}, optimizer updates: {log['updates']}, target syncs: {log['target_syncs']}")
    p(f"training outcomes overall: success {sum(log['success'])}, collision {sum(log['collision'])}, "
      f"timeout {sum(log['timeout'])}")
    p(f"last 200 training episodes: success {np.mean(log['success'][last]):.2f}, "
      f"collision {np.mean(log['collision'][last]):.2f}, timeout {np.mean(log['timeout'][last]):.2f}")
    plot_training_curves(log["return"], log["success"], log["epsilon"],
                         f"Two-robot DQN ({steps} env steps, seed {seed})",
                         os.path.join(RESULTS_DIR, f"two_robot_training_{tag}.png"), window=50)

    p("\nGreedy joint rollout from A=(2,0), B=(2,4):")
    g = greedy_rollout_two(online)
    p(format_rollout_table(g))
    plot_two_robot_trajectory(g["states"], f"Greedy two-robot rollout [{tag}] - "
                              f"{'success' if g['success'] else 'fail'} in {g['steps']} steps",
                              os.path.join(RESULTS_DIR, f"two_robot_trajectory_{tag}.png"),
                              (GOAL_A, GOAL_B))

    p("\nAdditional start A=(0,0), B=(4,4), goals unchanged, no retraining:")
    e = greedy_rollout_two(online, ((0, 0), (4, 4)))
    p(format_rollout_table(e))
    plot_two_robot_trajectory(e["states"], f"Two-robot rollout from A(0,0), B(4,4) [{tag}]",
                              os.path.join(RESULTS_DIR, f"two_robot_trajectory_extra_{tag}.png"),
                              (GOAL_A, GOAL_B))

    with open(os.path.join(RESULTS_DIR, f"two_robot_results_{tag}.txt"), "w") as f:
        f.write("\n".join(out) + "\n")
    strip = lambda d: {k: v for k, v in d.items() if k not in ("rows",)}  # noqa: E731
    with open(os.path.join(RESULTS_DIR, f"summary_two_robot_{tag}.json"), "w") as f:
        json.dump({"budget": f"{steps} env steps", "seed": seed, "default": strip(g),
                   "extra_start": strip(e)}, f, indent=2, default=str)
    return g, e


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=60_000)
    args = ap.parse_args()
    run(args.seed, args.steps)
