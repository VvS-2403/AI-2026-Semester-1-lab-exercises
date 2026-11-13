"""
Part I - Tabular Q-learning for the single-robot warehouse (Tasks 3 and 5).

Settings (from the lab): alpha = 0.2, gamma = 0.95, Q initialised to 0, seed 7,
1,500 episodes, epsilon decreased linearly 1 -> 0.05 over the first 1,200
episodes, 40-step truncation.

Run:  python qlearning.py
Outputs go to ../results/.
"""
from __future__ import annotations

import json
import os

import numpy as np

from plotting import RESULTS_DIR, plot_training_curves, plot_trajectory
from warehouse_env import (ACTION_NAMES, GAMMA, GOAL, N_ACTIONS, N_CELLS, START,
                           WarehouseEnv, cell_index)

ALPHA = 0.2
SEED = 7
N_EPISODES = 1500
EPS_START, EPS_END, EPS_DECAY_EPISODES = 1.0, 0.05, 1200
MA_WINDOW = 50


# ---------------------------------------------------------------- policy ---
def linear_epsilon(episode: int, start=EPS_START, end=EPS_END, decay=EPS_DECAY_EPISODES) -> float:
    """Epsilon used for episode ``episode`` (0-based): linear from start to end, then constant."""
    if decay <= 0:
        return end
    frac = min(episode / decay, 1.0)
    return start + frac * (end - start)


def epsilon_greedy(q_row: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """Training action choice: uniform over all 5 actions w.p. epsilon,
    otherwise a greedy action with ties broken uniformly at random."""
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    best = np.flatnonzero(q_row == q_row.max())
    return int(rng.choice(best))


def greedy_action(q_row: np.ndarray) -> int:
    """Evaluation action choice: fixed tie-break (lowest action code wins)."""
    return int(np.argmax(q_row))


# ---------------------------------------------------------------- update ---
def td_target(reward: float, next_max: float, terminated: bool, gamma: float = GAMMA) -> float:
    """y = r on true termination, else r + gamma * max_a' Q(s', a').
    Truncation is NOT termination, so a truncated step still bootstraps."""
    return reward if terminated else reward + gamma * next_max


def q_update(Q: np.ndarray, s: int, a: int, r: float, s_next: int, terminated: bool,
             alpha: float = ALPHA, gamma: float = GAMMA):
    """One Q-learning update of the single entry Q[s, a]. Returns (target, new value)."""
    y = td_target(r, float(Q[s_next].max()), terminated, gamma)
    Q[s, a] += alpha * (y - Q[s, a])
    return y, Q[s, a]


# -------------------------------------------------------------- training ---
def train_q_learning(n_episodes=N_EPISODES, alpha=ALPHA, gamma=GAMMA, seed=SEED,
                     eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY_EPISODES):
    rng = np.random.default_rng(seed)
    env = WarehouseEnv()
    Q = np.zeros((N_CELLS, N_ACTIONS))  # 25 x 5; obstacle rows stay unused
    log = {k: [] for k in ("return", "success", "epsilon", "length", "invalid", "timeout")}

    for ep in range(n_episodes):
        eps = linear_epsilon(ep, eps_start, eps_end, eps_decay)
        pos = env.reset()
        ep_return, invalid = 0.0, 0
        while True:
            s = cell_index(pos)
            a = epsilon_greedy(Q[s], eps, rng)
            next_pos, r, terminated, truncated, info = env.step(a)
            # next_pos is the state *before* any reset -> used in every update
            q_update(Q, s, a, r, cell_index(next_pos), terminated, alpha, gamma)
            ep_return += r
            invalid += int(info["invalid"])
            pos = next_pos
            if terminated or truncated:
                break
        log["return"].append(ep_return)
        log["success"].append(int(terminated))
        log["epsilon"].append(eps)
        log["length"].append(env.steps)
        log["invalid"].append(invalid)
        log["timeout"].append(int(truncated))
    return Q, log


# ------------------------------------------------------------ evaluation ---
def greedy_rollout(Q: np.ndarray, start=START):
    """epsilon = 0, no learning updates, fixed tie-break."""
    env = WarehouseEnv()
    pos = env.reset(start)
    path, actions, invalid, ret = [pos], [], 0, 0.0
    while True:
        a = greedy_action(Q[cell_index(pos)])
        pos, r, terminated, truncated, info = env.step(a)
        path.append(pos)
        actions.append(ACTION_NAMES[a])
        invalid += int(info["invalid"])
        ret += r
        if terminated or truncated:
            break
    return {"path": path, "actions": actions, "success": terminated, "timeout": truncated,
            "steps": env.steps, "invalid": invalid, "return": ret}


def random_policy_eval(n_runs=100, seed=SEED, start=START):
    rng = np.random.default_rng(seed)
    env = WarehouseEnv()
    succ, timeouts, steps_succ, invalids = 0, 0, [], []
    for _ in range(n_runs):
        env.reset(start)
        inv = 0
        while True:
            _, _, terminated, truncated, info = env.step(int(rng.integers(N_ACTIONS)))
            inv += int(info["invalid"])
            if terminated or truncated:
                break
        invalids.append(inv)
        if terminated:
            succ += 1
            steps_succ.append(env.steps)
        else:
            timeouts += 1
    return {"runs": n_runs, "success_frac": succ / n_runs, "timeout_frac": timeouts / n_runs,
            "mean_steps_success": float(np.mean(steps_succ)) if steps_succ else None,
            "mean_invalid": float(np.mean(invalids)), "total_invalid": int(np.sum(invalids))}


def format_q_table(Q: np.ndarray) -> str:
    lines = ["cell    " + "  ".join(f"{n:>6}" for n in ACTION_NAMES)]
    for idx in range(N_CELLS):
        r, c = divmod(idx, 5)
        if not Q[idx].any():
            continue
        lines.append(f"({r},{c})  " + "  ".join(f"{v:6.2f}" for v in Q[idx]))
    return "\n".join(lines)


def describe_rollout(res) -> str:
    status = "SUCCESS" if res["success"] else ("TIMEOUT" if res["timeout"] else "?")
    return (f"{status}: steps={res['steps']}, invalid={res['invalid']}, return={res['return']:.2f}\n"
            f"  positions: {' -> '.join(map(str, res['path']))}\n"
            f"  actions  : {', '.join(res['actions'])}")


# ------------------------------------------------------------------ main ---
def main():
    out = []
    p = lambda s="": (print(s), out.append(s))  # noqa: E731

    p("=== Part I: tabular Q-learning ===")
    p(f"alpha={ALPHA}, gamma={GAMMA}, seed={SEED}, episodes={N_EPISODES}, "
      f"epsilon {EPS_START}->{EPS_END} over first {EPS_DECAY_EPISODES} episodes, cutoff 40 steps")
    Q, log = train_q_learning()
    np.save(os.path.join(RESULTS_DIR, "q_table.npy"), Q)

    plot_training_curves(log["return"], log["success"], log["epsilon"],
                         f"Tabular Q-learning (seed {SEED})",
                         os.path.join(RESULTS_DIR, "qlearning_training.png"), window=MA_WINDOW)
    last = slice(-100, None)
    p(f"Training: first success at episode {log['success'].index(1) if 1 in log['success'] else None}; "
      f"success rate last 100 episodes = {np.mean(log['success'][last]):.2f}; "
      f"mean return last 100 = {np.mean(log['return'][last]):.2f}; "
      f"timeouts overall = {sum(log['timeout'])}/{N_EPISODES}")

    p("\nGreedy evaluation from default start (2,0) [epsilon=0, no updates]:")
    greedy = greedy_rollout(Q, START)
    p(describe_rollout(greedy))
    plot_trajectory(greedy["path"], f"Greedy Q-table route ({greedy['steps']} steps)",
                    os.path.join(RESULTS_DIR, "qlearning_trajectory.png"), START, GOAL)

    p("\nRandom policy, 100 episodes from (2,0):")
    rnd = random_policy_eval()
    p(json.dumps(rnd, indent=2))

    p("\nComparison table (Task 5):")
    p(f"{'Measure':32s}{'Random policy':>22s}{'Greedy Q-policy':>18s}")
    p(f"{'Goal reached':32s}{rnd['success_frac']*100:>20.0f} %{('yes' if greedy['success'] else 'no'):>18s}")
    ms = f"{rnd['mean_steps_success']:.1f} (mean)" if rnd["mean_steps_success"] else "n/a"
    p(f"{'Steps to goal, if successful':32s}{ms:>22s}{greedy['steps'] if greedy['success'] else 'n/a':>18}")
    p(f"{'Timeouts':32s}{rnd['timeout_frac']*100:>20.0f} %{int(greedy['timeout']):>18d}")
    p(f"{'Invalid moves':32s}{rnd['mean_invalid']:>14.2f} / run{greedy['invalid']:>18d}")

    p("\nAdditional start (0,0) without retraining:")
    extra = greedy_rollout(Q, (0, 0))
    p(describe_rollout(extra))
    plot_trajectory(extra["path"], f"Greedy Q-table route from (0,0)",
                    os.path.join(RESULTS_DIR, "qlearning_trajectory_from_0_0.png"), (0, 0), GOAL)

    p("\nLearned Q-table (visited cells only):")
    p(format_q_table(Q))

    # ---- Optional investigation: no exploration ------------------------------
    p("\n=== Optional: epsilon = 0 throughout (no exploration), same other settings ===")
    no_explore = {}
    for seed in (7, 11, 23):
        Qn, logn = train_q_learning(seed=seed, eps_start=0.0, eps_end=0.0, eps_decay=0)
        gn = greedy_rollout(Qn, START)
        Qe, loge = train_q_learning(seed=seed)
        ge = greedy_rollout(Qe, START)
        no_explore[seed] = {"no_explore": gn, "eps_greedy": ge}
        p(f"seed {seed}: no-exploration -> first success ep "
          f"{logn['success'].index(1) if 1 in logn['success'] else None}, "
          f"train success last 100 = {np.mean(logn['success'][-100:]):.2f}, greedy: "
          f"{'success' if gn['success'] else 'fail'} in {gn['steps']} steps | "
          f"epsilon-greedy -> train success last 100 = {np.mean(loge['success'][-100:]):.2f}, greedy: "
          f"{'success' if ge['success'] else 'fail'} in {ge['steps']} steps")
        if seed == 7:
            plot_training_curves(logn["return"], logn["success"], logn["epsilon"],
                                 "Q-learning with NO exploration (epsilon=0, seed 7)",
                                 os.path.join(RESULTS_DIR, "qlearning_no_exploration.png"), window=MA_WINDOW)

    with open(os.path.join(RESULTS_DIR, "qlearning_results.txt"), "w") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.join(RESULTS_DIR, "summary_qlearning.json"), "w") as f:
        json.dump({"budget": f"{N_EPISODES} episodes", "seed": SEED, "default": greedy, "extra_start": extra,
                   "random": rnd}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
