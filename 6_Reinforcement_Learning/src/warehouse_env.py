"""
Warehouse environments for the RL / Deep RL laboratory (CS F407).

Map (5 x 5, rows increase downwards):

        col 0   1   2   3   4
    row 0   .   .   #   .   .
    row 1   .   .   #   .   .
    row 2   S   .  door .   G
    row 3   .   .   #   .   .
    row 4   .   .   #   .   .

* Obstacles: (0,2), (1,2), (3,2), (4,2).  (2,2) is the only doorway.
* Actions (codes 0..4): Up, Down, Left, Right, Wait.
* A move into an obstacle or beyond the grid leaves the robot in place.

Two environments are provided:

* ``WarehouseEnv``          - single robot, used by Part I (Q-table) and Part II (DQN).
* ``TwoRobotWarehouseEnv``  - two robots, one central controller, joint actions (Part III).

Both separate *termination* (the task really ended) from *truncation*
(the 40-step collection cutoff). Learners must bootstrap on truncation and
must not bootstrap on termination.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Fixed problem specification
# --------------------------------------------------------------------------
N_ROWS = N_COLS = 5
N_CELLS = N_ROWS * N_COLS
OBSTACLES = frozenset({(0, 2), (1, 2), (3, 2), (4, 2)})
START = (2, 0)
GOAL = (2, 4)

ACTION_NAMES = ["Up", "Down", "Left", "Right", "Wait"]
UP, DOWN, LEFT, RIGHT, WAIT = range(5)
N_ACTIONS = 5
DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

MAX_STEPS = 40
GAMMA = 0.95

# Single-robot rewards (mutually exclusive)
R_GOAL = 10.0
R_INVALID = -1.0
R_STEP = -0.1

# Two-robot team reward constants
R_COLLISION = -10.0
R_ARRIVAL = 5.0     # per robot newly reaching its own goal (k)
R_ALL_DONE = 5.0    # bonus when both robots are at their goals (I)


# --------------------------------------------------------------------------
# Grid helpers
# --------------------------------------------------------------------------
def cell_index(pos: tuple[int, int]) -> int:
    """Cell (r, c) -> index 5r + c. Used for the Q-table row and the one-hot."""
    r, c = pos
    return N_COLS * r + c


def index_to_cell(idx: int) -> tuple[int, int]:
    return divmod(idx, N_COLS)


def is_free(pos: tuple[int, int]) -> bool:
    r, c = pos
    return 0 <= r < N_ROWS and 0 <= c < N_COLS and pos not in OBSTACLES


def propose_move(pos: tuple[int, int], action: int) -> tuple[tuple[int, int], bool]:
    """Apply one action from ``pos``.

    Returns (new_position, invalid). An obstacle/boundary move returns the old
    position with invalid=True. Wait is always valid.
    """
    dr, dc = DELTAS[action]
    target = (pos[0] + dr, pos[1] + dc)
    if is_free(target):
        return target, False
    return pos, True


def one_hot(pos: tuple[int, int]) -> np.ndarray:
    v = np.zeros(N_CELLS, dtype=np.float32)
    v[cell_index(pos)] = 1.0
    return v


def free_cells() -> list[tuple[int, int]]:
    return [(r, c) for r in range(N_ROWS) for c in range(N_COLS) if (r, c) not in OBSTACLES]


# --------------------------------------------------------------------------
# Single robot
# --------------------------------------------------------------------------
class WarehouseEnv:
    """Single-robot warehouse MDP.

    State  : the robot position (r, c) (map and goal are fixed, so this is Markov).
    step() : returns (next_state, reward, terminated, truncated, info)
             info = {"invalid": bool, "steps": int}
    """

    def __init__(self, start=START, goal=GOAL, max_steps=MAX_STEPS):
        self.default_start = tuple(start)
        self.goal = tuple(goal)
        self.max_steps = max_steps
        self.pos = self.default_start
        self.steps = 0
        self.done = False

    def reset(self, start=None):
        self.pos = tuple(start) if start is not None else self.default_start
        assert is_free(self.pos), f"start {self.pos} is not a free cell"
        self.steps = 0
        self.done = False
        return self.pos

    # Test mode: place the robot anywhere without training.
    def set_position(self, pos, steps_taken: int = 0):
        assert is_free(tuple(pos)), f"{pos} is not a free cell"
        self.pos = tuple(pos)
        self.steps = steps_taken
        self.done = False
        return self.pos

    def step(self, action: int):
        assert not self.done, "episode is over; call reset()"
        assert 0 <= action < N_ACTIONS
        new_pos, invalid = propose_move(self.pos, action)
        self.pos = new_pos
        self.steps += 1

        terminated = new_pos == self.goal
        if terminated:
            reward = R_GOAL
        elif invalid:
            reward = R_INVALID
        else:
            reward = R_STEP
        # Truncation = collection cutoff only; never set together with termination.
        truncated = (not terminated) and self.steps >= self.max_steps
        self.done = terminated or truncated
        return new_pos, reward, terminated, truncated, {"invalid": invalid, "steps": self.steps}


# --------------------------------------------------------------------------
# Two robots, one controller
# --------------------------------------------------------------------------
START_A, GOAL_A = (2, 0), (2, 4)
START_B, GOAL_B = (2, 4), (2, 0)
N_JOINT_ACTIONS = N_ACTIONS * N_ACTIONS  # 25


def encode_joint_action(a_A: int, a_B: int) -> int:
    return N_ACTIONS * a_A + a_B


def decode_joint_action(j: int) -> tuple[int, int]:
    return divmod(j, N_ACTIONS)


def two_robot_one_hot(state) -> np.ndarray:
    """Concatenate the two length-25 one-hot vectors -> length 50."""
    pA, pB = state
    return np.concatenate([one_hot(pA), one_hot(pB)])


class TwoRobotWarehouseEnv:
    """Central controller for robots A and B.

    State        : ordered pair (pA, pB).
    Joint action : j = 5*a_A + a_B  (25 joint actions).

    Simultaneous movement:
      1. Both proposals are computed from the same old state.
         Obstacle/boundary moves become stays (counted as invalid).
         A robot already at its goal stays regardless of its action
         (and is not counted as invalid).
      2. Collision if the proposed cells are equal (includes entering a cell
         whose robot stays/waits), or if the robots swap old positions.
         Following a robot into the cell it vacates is allowed.
      3. On collision: old positions retained, reward -10, terminate.
         Otherwise reward = -0.1 - b + 5k + 5I, terminate when both arrived.
      4. Truncate unfinished rollouts after 40 joint steps.
    """

    def __init__(self, starts=(START_A, START_B), goals=(GOAL_A, GOAL_B), max_steps=MAX_STEPS):
        self.default_starts = (tuple(starts[0]), tuple(starts[1]))
        self.goals = (tuple(goals[0]), tuple(goals[1]))
        self.max_steps = max_steps
        self.reset()

    @property
    def state(self):
        return (self.pA, self.pB)

    def reset(self, starts=None):
        sA, sB = starts if starts is not None else self.default_starts
        return self.set_positions(sA, sB)

    def set_positions(self, pA, pB, steps_taken: int = 0):
        pA, pB = tuple(pA), tuple(pB)
        assert is_free(pA) and is_free(pB) and pA != pB
        self.pA, self.pB = pA, pB
        self.steps = steps_taken
        self.done = False
        return self.state

    def at_goal(self):
        return (self.pA == self.goals[0], self.pB == self.goals[1])

    def step(self, joint_action: int):
        assert not self.done, "episode is over; call reset()"
        a_A, a_B = decode_joint_action(joint_action)
        old = (self.pA, self.pB)
        was_home = self.at_goal()

        proposals, invalid_count = [], 0
        for i, (pos, act) in enumerate(zip(old, (a_A, a_B))):
            if was_home[i]:
                proposals.append(pos)          # parked at goal
                continue
            new, invalid = propose_move(pos, act)
            invalid_count += int(invalid)
            proposals.append(new)
        nA, nB = proposals

        same_cell = nA == nB
        swap = nA == old[1] and nB == old[0]
        collision = same_cell or swap
        self.steps += 1

        info = {"collision": collision, "same_cell": same_cell, "swap": swap,
                "invalid": invalid_count, "arrived": 0, "steps": self.steps,
                "actions": (ACTION_NAMES[a_A], ACTION_NAMES[a_B])}

        if collision:
            reward = R_COLLISION                # overrides every other term
            terminated = True
        else:
            self.pA, self.pB = nA, nB
            now_home = self.at_goal()
            k = sum(int(now_home[i] and not was_home[i]) for i in range(2))
            all_home = int(all(now_home))
            reward = R_STEP - invalid_count + R_ARRIVAL * k + R_ALL_DONE * all_home
            terminated = bool(all_home)
            info["arrived"] = k

        truncated = (not terminated) and self.steps >= self.max_steps
        self.done = terminated or truncated
        return self.state, reward, terminated, truncated, info
