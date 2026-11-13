"""
Part II - Deep Q-Network for the single-robot warehouse (Task 6).

What changes vs. the Q-table: Q(s, .) is produced by a network
25 one-hot inputs -> 64 ReLU -> 64 ReLU -> 5 linear outputs, trained by
minimising the mean squared TD error on random mini-batches from a replay
buffer, with targets from a periodically-synchronised target network.

What stays the same: the environment, rewards, start, goal, gamma = 0.95,
the 40-step truncation, the terminated/truncated distinction, epsilon-greedy
training and greedy (epsilon = 0, no updates) evaluation.

The trainer below is generic (input/output sizes and the state encoder are
parameters) and is reused unchanged by two_robot_dqn.py.

Run:  python dqn.py
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from plotting import RESULTS_DIR, plot_training_curves, plot_trajectory
from warehouse_env import (ACTION_NAMES, GAMMA, GOAL, N_ACTIONS, N_CELLS, START,
                           WarehouseEnv, one_hot)


@dataclass
class DQNConfig:
    total_steps: int = 12_000
    seed: int = 7
    gamma: float = GAMMA
    lr: float = 1e-3
    buffer_capacity: int = 20_000
    batch_size: int = 64
    learning_starts: int = 500       # env steps collected before the first update
    train_every: int = 4             # one optimizer update every 4 env steps
    target_sync_every: int = 250     # optimizer updates between target copies
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.8      # linear decay over the first 80 % of steps
    hidden: int = 64


# --------------------------------------------------------------- network ---
class QNetwork(nn.Module):
    """in_dim -> 64 ReLU -> 64 ReLU -> n_actions linear outputs (Q-values, no softmax)."""

    def __init__(self, in_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------- replay buffer ---
class ReplayBuffer:
    """Fixed-capacity circular buffer of (s, a, r, s', terminated)."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.term = np.zeros(capacity, dtype=np.float32)   # 1 only for TRUE termination
        self.idx = 0
        self.size = 0

    def push(self, s, a, r, s2, terminated):
        i = self.idx
        self.s[i], self.a[i], self.r[i], self.s2[i], self.term[i] = s, a, r, s2, float(terminated)
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        ids = rng.integers(0, self.size, size=batch_size)   # uniform random batch
        return (torch.from_numpy(self.s[ids]), torch.from_numpy(self.a[ids]),
                torch.from_numpy(self.r[ids]), torch.from_numpy(self.s2[ids]),
                torch.from_numpy(self.term[ids]))

    def __len__(self):
        return self.size


# ------------------------------------------------------------------ loss ---
def dqn_loss(online: nn.Module, target: nn.Module, s, a, r, s2, term, gamma: float = GAMMA):
    """Mean squared TD error.

    q_sa : Q_theta(s_i, a_i) for the RECORDED action a_i (selected with gather).
    y    : r_i + gamma * (1 - d_i) * max_a' Q_target(s'_i, a'), computed under
           torch.no_grad() so no gradient flows into the target.
           d_i = 1 only for true termination; truncated transitions have d_i = 0.
    """
    q_sa = online(s).gather(1, a.long().unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_max = target(s2).max(dim=1).values
        y = r + gamma * (1.0 - term) * next_max
    loss = F.mse_loss(q_sa, y)
    return loss, q_sa, y


# ------------------------------------------------------------- training ---
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def linear_epsilon_steps(step: int, cfg: DQNConfig) -> float:
    decay_steps = cfg.eps_decay_frac * cfg.total_steps
    frac = min(step / decay_steps, 1.0)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def select_action(online: nn.Module, obs: np.ndarray, epsilon: float, n_actions: int,
                  rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    with torch.no_grad():
        q = online(torch.from_numpy(obs).unsqueeze(0)).squeeze(0)
    return int(torch.argmax(q).item())  # exact ties are practically impossible with a network


def train_dqn(env, encode, obs_dim: int, n_actions: int, cfg: DQNConfig, extra_info_keys=(),
              verbose=True):
    """Generic DQN training loop.

    env.reset() -> state ; env.step(a) -> (state, r, terminated, truncated, info)
    encode(state) -> np.float32 vector of length obs_dim
    """
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    online = QNetwork(obs_dim, n_actions, cfg.hidden)
    target = QNetwork(obs_dim, n_actions, cfg.hidden)
    target.load_state_dict(online.state_dict())          # initialise target from online
    target.eval()
    optim = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.buffer_capacity, obs_dim)

    log = {"return": [], "success": [], "epsilon": [], "length": [], "invalid": [], "timeout": [],
           "end_step": [], "loss": [], "loss_step": [], "target_syncs": 0, "updates": 0}
    for k in extra_info_keys:
        log[k] = []

    state = env.reset()
    obs = encode(state)
    ep_ret, ep_inv, ep_extra = 0.0, 0, {k: 0 for k in extra_info_keys}
    ep_eps = linear_epsilon_steps(0, cfg)
    n_updates = 0

    for step in range(cfg.total_steps):
        eps = linear_epsilon_steps(step, cfg)
        a = select_action(online, obs, eps, n_actions, rng)
        next_state, r, terminated, truncated, info = env.step(a)
        next_obs = encode(next_state)                    # next state BEFORE any reset
        buffer.push(obs, a, r, next_obs, terminated)     # truncation is stored as d = 0
        ep_ret += r
        ep_inv += int(info["invalid"])
        for k in extra_info_keys:
            ep_extra[k] += int(info[k])
        obs = next_obs

        # ---- learning ---------------------------------------------------
        if step + 1 >= cfg.learning_starts and (step + 1) % cfg.train_every == 0:
            s, a_b, r_b, s2, d = buffer.sample(cfg.batch_size, rng)
            loss, _, _ = dqn_loss(online, target, s, a_b, r_b, s2, d, cfg.gamma)
            optim.zero_grad()
            loss.backward()
            optim.step()
            n_updates += 1
            if n_updates % 50 == 0:
                log["loss"].append(loss.item())
                log["loss_step"].append(step + 1)
            if n_updates % cfg.target_sync_every == 0:        # target-network sync
                target.load_state_dict(online.state_dict())
                log["target_syncs"] += 1

        # ---- episode bookkeeping ---------------------------------------
        if terminated or truncated:
            success = terminated and not info.get("collision", False)
            log["return"].append(ep_ret)
            log["success"].append(int(success))
            log["epsilon"].append(ep_eps)
            log["length"].append(info["steps"])
            log["invalid"].append(ep_inv)
            log["timeout"].append(int(truncated))
            log["end_step"].append(step + 1)
            for k in extra_info_keys:
                log[k].append(ep_extra[k])
            if verbose and len(log["return"]) % 100 == 0:
                recent = log["success"][-100:]
                print(f"  step {step+1:6d}  episodes {len(log['return']):5d}  eps {eps:.3f}  "
                      f"success(last100) {np.mean(recent):.2f}")
            state = env.reset()
            obs = encode(state)
            ep_ret, ep_inv, ep_extra = 0.0, 0, {k: 0 for k in extra_info_keys}
            ep_eps = linear_epsilon_steps(step + 1, cfg)

    log["updates"] = n_updates
    return online, log


# ------------------------------------------------------------ evaluation ---
def greedy_rollout_dqn(online: nn.Module, start=START):
    env = WarehouseEnv()
    pos = env.reset(start)
    path, actions, invalid, ret = [pos], [], 0, 0.0
    online.eval()
    while True:
        with torch.no_grad():
            q = online(torch.from_numpy(one_hot(pos)).unsqueeze(0)).squeeze(0)
        a = int(torch.argmax(q).item())  # fixed tie-break: first max
        pos, r, terminated, truncated, info = env.step(a)
        path.append(pos)
        actions.append(ACTION_NAMES[a])
        invalid += int(info["invalid"])
        ret += r
        if terminated or truncated:
            break
    return {"path": path, "actions": actions, "success": terminated, "timeout": truncated,
            "steps": env.steps, "invalid": invalid, "return": ret}


def main():
    from qlearning import describe_rollout

    out = []
    p = lambda s="": (print(s), out.append(s))  # noqa: E731
    cfg = DQNConfig()
    p("=== Part II: DQN, single robot ===")
    p(f"config: {json.dumps(asdict(cfg))}")
    p(f"torch {torch.__version__}, numpy {np.__version__}")
    env = WarehouseEnv()
    online, log = train_dqn(env, one_hot, N_CELLS, N_ACTIONS, cfg)
    torch.save(online.state_dict(), os.path.join(RESULTS_DIR, "dqn_single.pt"))

    p(f"episodes: {len(log['return'])}, optimizer updates: {log['updates']}, "
      f"target syncs: {log['target_syncs']}")
    p(f"training success rate last 100 episodes: {np.mean(log['success'][-100:]):.2f}; "
      f"first success at episode {log['success'].index(1) if 1 in log['success'] else None}")
    plot_training_curves(log["return"], log["success"], log["epsilon"],
                         f"DQN single robot ({cfg.total_steps} env steps, seed {cfg.seed})",
                         os.path.join(RESULTS_DIR, "dqn_training.png"), window=50)

    p("\nGreedy DQN evaluation from (2,0):")
    g = greedy_rollout_dqn(online, START)
    p(describe_rollout(g))
    plot_trajectory(g["path"], f"Greedy DQN route ({g['steps']} steps)",
                    os.path.join(RESULTS_DIR, "dqn_trajectory.png"), START, GOAL)

    p("\nGreedy DQN from additional start (0,0), no retraining:")
    e = greedy_rollout_dqn(online, (0, 0))
    p(describe_rollout(e))

    with torch.no_grad():
        q0 = online(torch.from_numpy(one_hot(START)).unsqueeze(0)).squeeze(0).numpy()
    p("\nQ_theta((2,0), .) = " + ", ".join(f"{n}:{v:.2f}" for n, v in zip(ACTION_NAMES, q0)))

    try:
        Q = np.load(os.path.join(RESULTS_DIR, "q_table.npy"))
        p("Q-table  ((2,0), .) = " + ", ".join(f"{n}:{v:.2f}" for n, v in zip(ACTION_NAMES, Q[10])))
    except FileNotFoundError:
        pass

    with open(os.path.join(RESULTS_DIR, "dqn_results.txt"), "w") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.join(RESULTS_DIR, "summary_dqn.json"), "w") as f:
        json.dump({"budget": f"{cfg.total_steps} env steps", "seed": cfg.seed, "default": g,
                   "extra_start": e}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
