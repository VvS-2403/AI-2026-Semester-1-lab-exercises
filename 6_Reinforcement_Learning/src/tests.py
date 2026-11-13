"""
Tests that run BEFORE any training (Tasks 4, 6 and 7).

  1. Single-robot environment transitions (Task 4 table) + 40-step truncation.
  2. Synthetic Q-learning updates: non-terminal, terminal, truncated (Task 4).
  3. DQN loss on a hand-computable batch; recorded-action selection; no
     gradient into the target network (Task 6).
  4. Two-robot simultaneous-movement / collision / reward rules (Task 7).

Run:  python tests.py      (exits non-zero if any check fails)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

from dqn import QNetwork, dqn_loss
from plotting import RESULTS_DIR
from qlearning import q_update, td_target
from warehouse_env import (ACTION_NAMES, DOWN, LEFT, RIGHT, UP, WAIT, TwoRobotWarehouseEnv,
                           WarehouseEnv, cell_index, encode_joint_action)

LOG: list[str] = []
FAILURES: list[str] = []


def say(s=""):
    print(s)
    LOG.append(s)


def check(name, got, expected, tol=1e-6):
    if isinstance(expected, float):
        ok = abs(got - expected) <= tol
    else:
        ok = got == expected
    say(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got!r}, expected {expected!r}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
def test_single_env():
    say("== 1. Single-robot environment (positions set explicitly, no learner) ==")
    env = WarehouseEnv()
    cases = [
        ((0, 0), UP,    (0, 0), -1.0, False, "boundary -> stay"),
        ((1, 1), RIGHT, (1, 1), -1.0, False, "obstacle (1,2) -> stay"),
        ((2, 1), RIGHT, (2, 2), -0.1, False, "into doorway"),
        ((2, 1), WAIT,  (2, 1), -0.1, False, "wait is valid"),
        ((2, 3), RIGHT, (2, 4), 10.0, True,  "reach goal"),
    ]
    for start, a, exp_pos, exp_r, exp_term, note in cases:
        env.set_position(start)
        pos, r, term, trunc, info = env.step(a)
        say(f" {start} {ACTION_NAMES[a]:>5} ({note})")
        check("next position", pos, exp_pos)
        check("reward", r, exp_r)
        check("terminated", term, exp_term)
        check("truncated", trunc, False)

    say(" 40-step cutoff: wait from (2,0) forty times")
    env.reset()
    for t in range(1, 41):
        pos, r, term, trunc, _ = env.step(WAIT)
        if t == 39:
            check("not truncated after 39 steps", trunc, False)
    check("truncated after 40 steps", trunc, True)
    check("not terminated at cutoff", term, False)
    check("rollout stopped (done flag)", env.done, True)
    try:
        env.step(WAIT)
        check("step after cutoff raises", False, True)
    except AssertionError:
        check("step after cutoff raises", True, True)

    say(" Reaching the goal on step 40 is termination, not truncation")
    env.set_position((2, 3), steps_taken=39)
    _, r, term, trunc, _ = env.step(RIGHT)
    check("terminated", term, True)
    check("truncated", trunc, False)


# ---------------------------------------------------------------------------
def test_q_update():
    say("\n== 2. Synthetic Q-learning updates (alpha=0.2, gamma=0.95, Q(s,a)=0.4, max Q(s',.)=2) ==")
    s, a, s2 = 0, 3, 1

    def fresh():
        Q = np.zeros((25, 5))
        Q[s, a] = 0.4
        Q[s2] = [0.0, 2.0, -1.0, 0.5, 1.0]   # max = 2
        return Q

    # by hand: y = 1 + 0.95*2 = 2.9 ; Q <- 0.4 + 0.2*(2.9 - 0.4) = 0.9
    Q = fresh()
    y, new = q_update(Q, s, a, 1.0, s2, terminated=False, alpha=0.2, gamma=0.95)
    say(" non-terminal r=1:")
    check("target", y, 2.9)
    check("updated Q(s,a)", new, 0.9)

    # by hand: y = 5 ; Q <- 0.4 + 0.2*(5 - 0.4) = 1.32
    Q = fresh()
    y, new = q_update(Q, s, a, 5.0, s2, terminated=True, alpha=0.2, gamma=0.95)
    say(" truly terminal r=5 (next-state value ignored):")
    check("target", y, 5.0)
    check("updated Q(s,a)", new, 1.32)

    # truncation is not termination -> same as non-terminal: y = 2.9, Q = 0.9
    Q = fresh()
    terminated, truncated = False, True
    y, new = q_update(Q, s, a, 1.0, s2, terminated=terminated, alpha=0.2, gamma=0.95)
    say(" non-terminal r=1 with truncation flag set (bootstraps):")
    check("target", y, 2.9)
    check("updated Q(s,a)", new, 0.9)

    say(" Integrated: real env truncation at step 40, update uses env flags")
    env = WarehouseEnv()
    Q = np.zeros((25, 5))
    Q[cell_index((2, 1))] = [0.0, 0.0, 0.0, 2.0, 0.0]   # max Q(s', .) = 2, s' = s for Wait
    env.set_position((2, 1), steps_taken=39)
    pos, r, term, trunc, _ = env.step(WAIT)
    y, _ = q_update(Q, cell_index((2, 1)), WAIT, r, cell_index(pos), term, 0.2, 0.95)
    check("truncated flag from env", trunc, True)
    check("target bootstraps: -0.1 + 0.95*2", y, -0.1 + 0.95 * 2)
    check("td_target(terminated=True) ignores next value", td_target(5.0, 1e6, True), 5.0)


# ---------------------------------------------------------------------------
class ConstQ(nn.Module):
    """Outputs the same learnable Q-vector for every input row (for hand-checkable tests)."""

    def __init__(self, values):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, x):
        return self.w.unsqueeze(0).expand(x.shape[0], -1)


def test_dqn_loss():
    say("\n== 3. DQN loss on a hand-computable batch (gamma=0.95) ==")
    online = ConstQ([0.5, 1.0, 1.5, 2.0, 2.5])           # Q_theta(s, a) = value of action a
    target = ConstQ([0.0, 2.0, 1.0, -1.0, 0.5])          # max_a' Q_target(s', a') = 2
    s = torch.zeros(3, 25)
    s2 = torch.zeros(3, 25)
    a = torch.tensor([3, 0, 1])                          # recorded actions
    r = torch.tensor([1.0, 5.0, 1.0])
    d = torch.tensor([0.0, 1.0, 0.0])                    # non-terminal, terminal, truncated(d=0)
    loss, q_sa, y = dqn_loss(online, target, s, a, r, s2, d, gamma=0.95)

    say(" rows: (non-terminal r=1), (terminal r=5), (truncated r=1)")
    check("y[0] non-terminal = 1 + 0.95*2", round(y[0].item(), 6), 2.9)
    check("y[1] terminal = 5", round(y[1].item(), 6), 5.0)
    check("y[2] truncated bootstraps = 2.9", round(y[2].item(), 6), 2.9)
    check("Q_theta(s,a) uses recorded actions [3,0,1]", [round(v, 4) for v in q_sa.tolist()],
          [2.0, 0.5, 1.0])
    exp_loss = ((2.0 - 2.9) ** 2 + (0.5 - 5.0) ** 2 + (1.0 - 2.9) ** 2) / 3
    check("MSE loss", round(loss.item(), 5), round(exp_loss, 5))
    check("target tensor has no grad", y.requires_grad, False)

    loss.backward()
    exp_grad = [2 * (0.5 - 5.0) / 3, 2 * (1.0 - 2.9) / 3, 0.0, 2 * (2.0 - 2.9) / 3, 0.0]
    check("online grad only on recorded actions", [round(g, 4) for g in online.w.grad.tolist()],
          [round(g, 4) for g in exp_grad])
    check("no gradient reaches target network", target.w.grad, None)

    say(" Same check with the real 25-64-64-5 networks:")
    torch.manual_seed(0)
    on, tg = QNetwork(25, 5), QNetwork(25, 5)
    tg.load_state_dict(on.state_dict())
    s = torch.eye(25)[:4]
    loss, _, _ = dqn_loss(on, tg, s, torch.tensor([0, 1, 2, 3]), torch.ones(4), s, torch.zeros(4))
    loss.backward()
    check("online params receive gradients", all(p.grad is not None for p in on.parameters()), True)
    check("target params receive none", all(p.grad is None for p in tg.parameters()), True)
    check("network output shape (batch, 5) - Q-values, no softmax", tuple(on(s).shape), (4, 5))


# ---------------------------------------------------------------------------
def test_two_robot():
    say("\n== 4. Two-robot simultaneous movement and team reward ==")
    env = TwoRobotWarehouseEnv()

    def do(pA, pB, aA, aB, steps_taken=0):
        env.set_positions(pA, pB, steps_taken)
        return env.step(encode_joint_action(aA, aB))

    cases = [
        # (A old, B old, A act, B act, exp state, exp reward, exp term, exp collision, prediction)
        ((2, 1), (2, 3), RIGHT, LEFT, ((2, 1), (2, 3)), -10.0, True, True,
         "both propose (2,2): same-cell collision, positions kept, -10, terminate"),
        ((2, 1), (2, 2), RIGHT, LEFT, ((2, 1), (2, 2)), -10.0, True, True,
         "A->(2,2), B->(2,1): swap collision, -10, terminate"),
        ((2, 1), (2, 2), RIGHT, RIGHT, ((2, 2), (2, 3)), -0.1, False, False,
         "A follows B into the vacated door cell: allowed, -0.1"),
        ((2, 1), (2, 2), UP, WAIT, ((1, 1), (2, 2)), -0.1, False, False,
         "A steps aside to (1,1), B waits in door: no collision, -0.1"),
        ((2, 1), (2, 2), RIGHT, WAIT, ((2, 1), (2, 2)), -10.0, True, True,
         "A enters waiting B's cell: collision"),
        ((2, 1), (2, 2), WAIT, UP, ((2, 1), (2, 2)), -1.1, False, False,
         "B bumps obstacle (1,2): b=1, reward -0.1-1 = -1.1"),
        ((0, 0), (4, 4), UP, DOWN, ((0, 0), (4, 4)), -2.1, False, False,
         "both hit the boundary: b=2, reward -2.1"),
    ]
    for pA, pB, aA, aB, exp_state, exp_r, exp_term, exp_col, pred in cases:
        state, r, term, trunc, info = do(pA, pB, aA, aB)
        say(f" A{pA} {ACTION_NAMES[aA]:>5} | B{pB} {ACTION_NAMES[aB]:>5}  prediction: {pred}")
        check("state", state, exp_state)
        check("reward", r, exp_r)
        check("terminated", term, exp_term)
        check("collision", info["collision"], exp_col)

    say(" Robot already at its goal (A at (2,4)) stays put and is not re-rewarded")
    state, r, term, _, info = do((2, 3), (1, 1), RIGHT, WAIT)
    check("A arrives: k=1, reward -0.1 + 5", r, 4.9)
    check("not terminated (B not home)", term, False)
    state, r, term, _, info = env.step(encode_joint_action(LEFT, WAIT))
    check("A ignores 'Left' at its goal", state[0], (2, 4))
    check("no repeated arrival reward, no invalid penalty", r, -0.1)
    state, r, term, _, info = env.step(encode_joint_action(UP, WAIT))
    check("A ignores 'Up' at its goal (no invalid count)", (state[0], info["invalid"]), ((2, 4), 0))

    say(" Moving into a parked robot's goal cell is a collision")
    state, r, term, _, info = do((2, 4), (2, 3), WAIT, RIGHT)
    check("B into parked A at (2,4)", (info["collision"], r, state), (True, -10.0, ((2, 4), (2, 3))))

    say(" Both arrive in the same step: k=2, I=1 -> -0.1 + 10 + 5 = 14.9, terminate")
    state, r, term, _, _ = do((2, 3), (2, 1), RIGHT, LEFT)
    check("reward", r, 14.9)
    check("terminated", term, True)

    say(" Second robot arrives later: k=1, I=1 -> -0.1 + 5 + 5 = 9.9")
    state, r, term, _, _ = do((2, 4), (2, 1), WAIT, LEFT)
    check("reward", r, 9.9)
    check("terminated", term, True)

    say(" 40 joint-step truncation")
    env.reset()
    for _ in range(40):
        state, r, term, trunc, _ = env.step(encode_joint_action(WAIT, WAIT))
    check("truncated at 40", (term, trunc), (False, True))

    say(" Joint-action encoding j = 5*a_A + a_B")
    check("encode(Right, Left)", encode_joint_action(RIGHT, LEFT), 17)
    check("divmod(17, 5)", divmod(17, 5), (RIGHT, LEFT))


if __name__ == "__main__":
    test_single_env()
    test_q_update()
    test_dqn_loss()
    test_two_robot()
    say(f"\n{'ALL TESTS PASSED' if not FAILURES else f'{len(FAILURES)} FAILURE(S): {FAILURES}'}")
    with open(os.path.join(RESULTS_DIR, "tests_output.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    sys.exit(1 if FAILURES else 0)
