"""Plotting helpers shared by the Q-learning, DQN and two-robot scripts."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from warehouse_env import N_COLS, N_ROWS, OBSTACLES  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def moving_average(x, window: int):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return np.arange(len(x)), x
    ma = np.convolve(x, np.ones(window) / window, mode="valid")
    return np.arange(window - 1, len(x)), ma


def plot_training_curves(returns, successes, epsilons, title, path, window=50, x_label="Episode"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x, ma = moving_average(returns, window)
    axes[0].plot(returns, alpha=0.25, color="tab:blue", label="per episode")
    axes[0].plot(x, ma, color="tab:blue", label=f"moving avg ({window} ep)")
    axes[0].set_title("Training return")
    x, ma = moving_average(successes, window)
    axes[1].plot(x, ma, color="tab:green")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title(f"Training success rate (moving avg, {window} ep)")
    axes[2].plot(epsilons, color="tab:orange")
    axes[2].set_title("Epsilon at episode start")
    for ax in axes:
        ax.set_xlabel(x_label)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _draw_grid(ax):
    for r in range(N_ROWS):
        for c in range(N_COLS):
            color = "dimgray" if (r, c) in OBSTACLES else "white"
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, edgecolor="black"))
    ax.set_xlim(-0.5, N_COLS - 0.5)
    ax.set_ylim(N_ROWS - 0.5, -0.5)  # rows increase downwards
    ax.set_xticks(range(N_COLS))
    ax.set_yticks(range(N_ROWS))
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_aspect("equal")


def plot_trajectory(path_cells, title, out_path, start=None, goal=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_grid(ax)
    rs = [p[0] for p in path_cells]
    cs = [p[1] for p in path_cells]
    ax.plot(cs, rs, "-o", color="tab:blue", lw=2)
    for t, (r, c) in enumerate(path_cells):
        ax.annotate(str(t), (c, r), textcoords="offset points", xytext=(6, 6), fontsize=8)
    if start:
        ax.text(start[1], start[0] + 0.3, "S", ha="center", fontsize=12, color="green")
    if goal:
        ax.text(goal[1], goal[0] + 0.3, "G", ha="center", fontsize=12, color="red")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_two_robot_trajectory(states, title, out_path, goals):
    """states: list of (pA, pB) for t = 0..T. Time labels show simultaneous moves."""
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_grid(ax)
    for idx, (color, name, off) in enumerate([("tab:blue", "A", (-0.12, -0.12)), ("tab:red", "B", (0.12, 0.12))]):
        rs = [s[idx][0] + off[0] for s in states]
        cs = [s[idx][1] + off[1] for s in states]
        ax.plot(cs, rs, "-o", color=color, lw=2, label=f"robot {name}", alpha=0.8)
        # one label per visited cell listing every time step spent there
        times: dict = {}
        for t, s in enumerate(states):
            times.setdefault(s[idx], []).append(t)
        for (r, c), ts in times.items():
            txt = ",".join(map(str, ts[:6])) + ("..." if len(ts) > 6 else "")
            ax.annotate(f"{name}:{txt}", (c + off[1], r + off[0]), textcoords="offset points",
                        xytext=(-28, -14 if idx == 0 else 6), fontsize=6.5, color=color)
        g = goals[idx]
        ax.text(g[1], g[0] + 0.38, f"G{name}", ha="center", fontsize=10, color=color)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
