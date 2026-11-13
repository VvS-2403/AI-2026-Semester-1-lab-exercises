# LLM prompts and corrections

The LLM used was Claude (Claude Code, model Opus 5.5). It was given the lab handout
together with the design written up in the README (Tasks 1, 2 and the Part III
design questions) *before* it wrote any code. The prompts below are the lab's example
prompts adapted to that design. Each one says what states, rewards, stopping rules
and success mean, so the LLM did not decide those itself.

---

## Prompt 1: environment and tabular Q-learning (Task 3)

> Implement tabular Q-learning in Python using NumPy for the warehouse environment.
> Use a 5x5 grid, rows increasing downwards, obstacle set {(0,2),(1,2),(3,2),(4,2)},
> start (2,0) and goal (2,4). Actions are codes 0..4 = Up, Down, Left, Right, Wait.
> An obstacle or boundary move stays in place. Use mutually exclusive rewards:
> +10 on reaching the goal, -1 for an invalid move, -0.1 for another valid action
> (including Wait). Reaching the goal terminates.
>
> The state is the robot position only. Index the Q-table as 25x5 with row 5r+c
> (obstacle rows unused). `step()` must return (next_state, reward, terminated,
> truncated, info) with *separate* termination and truncation flags. Truncate after
> 40 steps. Never set both flags. Bootstrap at truncation, not at goal termination,
> and use the next state from before any reset in every update.
>
> Provide separate functions for: the environment, epsilon-greedy action selection
> (uniform over all 5 actions with probability epsilon, otherwise greedy with random
> tie-breaking), one Q-learning update (return the target and the new value so I can
> test it), training, and greedy evaluation (epsilon = 0, no updates, fixed
> lowest-index tie-break). Use alpha = 0.2, gamma = 0.95, a zero Q-table and seed 7.
> Train for 1,500 episodes and decrease epsilon linearly from 1 to 0.05 over the
> first 1,200 episodes.
>
> Do not use path search or a ready-made RL library. Include a test mode where I can
> set the robot's position (and step counter), choose one action and inspect the
> outcome without training. Record per-episode return, success, epsilon, length,
> invalid moves and timeouts. Plot the training return and success as 50-episode
> moving averages. Print and plot the greedy trajectory. Also evaluate a uniformly
> random policy for 100 episodes with the same map, start, rewards and cutoff.

## Prompt 2: tests (Task 4)

> Write a test script that does not train. It should check:
> (a) the five transitions in the Task 4 table, each from an explicitly set position;
> (b) that 40 Waits from the start give truncated = True and terminated = False,
> and then stop the rollout;
> (c) the synthetic update with Q(s,a)=0.4, r=1, max Q(s',.)=2, alpha=0.2 and
> gamma=0.95 (expected target 2.9, new value 0.9), the terminal case r=5 (target 5,
> new value 1.32), and the truncated case (same as non-terminal).
> Print got/expected for every check and exit non-zero on failure.

## Prompt 3: DQN (Task 6)

> Extend my validated single-robot program to use DQN in PyTorch. Keep the
> environment, rewards, start, goal and discount unchanged. Use a one-hot input of
> length 25 (1 at 5r+c) and two 64-unit ReLU hidden layers. Output five Q-values
> without softmax. Use epsilon-greedy action selection, a replay buffer and a target
> network. Use a replay capacity of 20,000 transitions and random batches of 64.
> Initialise the target network from the online network. Collect 500 steps before
> learning, do one optimizer update every four environment steps, and copy the
> online weights to the target every 250 optimizer updates. Use Adam (lr 0.001) and
> mean squared TD error. Compute detached targets with `torch.no_grad()`. Use a
> reward-only target for termination and bootstrap at truncation (store d = 1 only
> for true termination). Train for 12,000 environment steps with seed 7, and
> decrease epsilon linearly from 1 to 0.05 over the first 80 % of steps. Keep
> training and greedy evaluation separate. Write the trainer so that the input size,
> the output size and the state encoder are parameters, so I can reuse it for two
> robots. Explain where the recorded action is selected when computing the loss.
>
> Also add tests: a batch with a non-terminal row (r=1, max target value 2), a
> terminal row (r=5) and a truncated row (r=1, max 2). Check that Q_theta(s,a) is
> taken for the recorded action, check the loss value, and check that no gradient
> reaches the target network.

## Prompt 4: two robots (Task 7)

> Extend the program to two robots with one central controller. A starts at (2,0)
> with goal (2,4). B starts at (2,4) with goal (2,0). The state is (pA, pB), encoded
> as two concatenated 25-length one-hots (50 inputs). Joint action j = 5*a_A + a_B
> (25 outputs), decoded with `divmod(j, 5)`.
>
> Movement is simultaneous. Compute both proposals from the same old state. An
> obstacle or boundary move becomes a stay and counts as invalid. A robot already at
> its goal stays whatever its action is, and is not counted as invalid. A collision
> happens if the proposed cells are equal or the robots swap old positions. Entering
> a waiting robot's cell is a collision. Following a robot into the cell it vacates
> is allowed. On collision keep the old positions, give reward -10 and terminate.
> Otherwise reward = -0.1 - b + 5k + 5I (b = invalid moves by robots not at their
> goals, k = robots newly at their goal, I = 1 if both are at their goals).
> Terminate when both have arrived and truncate after 40 joint steps.
>
> Reuse the DQN trainer unchanged except for the dimensions. Use 60,000 steps, seed 7
> and the Task 6 settings. Do NOT add a path planner or an action filter that
> prevents collisions. Print a table of both positions after every greedy joint step
> and plot both trajectories with time labels. Add tests for the four Task 7 cases
> and for a robot already at its goal.

---

## Corrections requested and issues found

1. **Environment, not code: PyTorch install.** `pip install torch` into the Windows
   Store Python's user site-packages left a truncated package
   (`No module named 'torch.torch_version'`), because Windows long paths are
   disabled. Fix: a virtual environment at a short path
   (`%USERPROFILE%\.venvs\rl-lab`). The lab code did not change.
2. **Tests:** every check in `tests.py` passed on the first run, so no environment or
   update rule had to be corrected.
3. **Plot readability:** in the two-robot trajectory plot, time labels piled up when a
   robot revisited a cell. Follow-up prompt: *"Merge the time labels so each visited
   cell shows one label listing every time step the robot was there."*
4. **Two-robot training failure:** seed 7 with 60k steps and seed 11 with 60k steps both
   timed out. The tests still passed, so the environment and loss were ruled out.
   Follow-up prompt: *"Print the network's 25 Q-values in the states where B
   oscillates, with A parked at its goal."* This showed overestimated, noisy values
   in B's endgame. Next prompt: *"Add `--seed` and `--steps` options, rerun with a
   larger budget (150k), and add a script that re-evaluates the saved models without
   training."* The 150k run succeeds in 7 joint steps. I did not ask for a planner or
   an action filter.
