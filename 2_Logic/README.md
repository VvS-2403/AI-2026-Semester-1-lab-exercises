# Logic Lab Submission Report

**Student:** Vismay
**Course:** Artificial Intelligence

## 1. Specification of the Planning Problem
- **Initial State ($I$):** `{At(Robot, A), At(Package, A)}`
- **Goal ($G$):** `{At(Package, C)}`
- **Available Actions:**
  - `Move(X, Y)`: Move from connected location X to Y.
  - `PickUp(Package, L)`: Pick up package at location L.
  - `Drop(Package, L)`: Drop package at location L.
- **Preconditions and Effects:**
  - `Move(A, B)`: Preconditions: `{At(Robot, A)}`, Effects: `{At(Robot, B), ¬At(Robot, A)}`
  - `PickUp(Package, A)`: Preconditions: `{At(Robot, A), At(Package, A)}`, Effects: `{Holding(Package), ¬At(Package, A)}`
  - `Drop(Package, C)`: Preconditions: `{At(Robot, C), Holding(Package)}`, Effects: `{At(Package, C), ¬Holding(Package)}`

### Task 0 Questions
**Starting from I, is `PickUp(Package, A)` applicable?**
Yes. Both of its preconditions, `At(Robot, A)` and `At(Package, A)`, are present in the initial state.

**What about `Drop(Package, C)`?**
No. Its preconditions, `At(Robot, C)` and `Holding(Package)`, are not present in the initial state.

## 2. Manually Constructed Plan
| State | Facts |
| --- | --- |
| $S_0$ | `At(Robot, A), At(Package, A)` |
| `PickUp(Package, A)` | |
| $S_1$ | `At(Robot, A), Holding(Package)` |
| `Move(A, B)` | |
| $S_2$ | `At(Robot, B), Holding(Package)` |
| `Move(B, C)` | |
| $S_3$ | `At(Robot, C), Holding(Package)` |
| `Drop(Package, C)` | |
| $S_4$ | `At(Robot, C), At(Package, C)` |

## 3. Prompt Used with the LLM
The prompt used to generate the Python planner can be found in `src/prompts.md`. It clearly specifies the action representation (name, preconditions, effects), application logic, BFS algorithm, and edge cases.

## 4. Generated Python Program
The Python code was implemented in `src/planner.py` with modifications to output to `results/tests_output.txt`.

### "Think About It" Mapping
- **Preconditions (When is an action applicable?):** Implemented in `Action.is_applicable()` which uses set subsets to verify positive preconditions and checks for empty intersection with negative preconditions.
- **Effects (How does the state change?):** Implemented in `Action.apply()` which removes negative effects (set difference) and adds positive effects (set update).
- **Goal (When does planning terminate?):** The `while` loop checks `if goal_state.issubset(current_state)` and terminates early by returning the plan.
- **BFS (How are alternative plans explored?):** A `collections.deque` is used as a queue, popping from the left and appending to the right, exploring states layer by layer.

## 5. Results of Tests
The execution outputs are saved in `results/tests_output.txt`.
- **Test A (Solvable Problem):** A valid plan was found (`PickUp(Package,A) -> Move(A,B) -> Move(B,C) -> Drop(Package,C)`).
- **Test B (Impossible Problem):** Removed the `PickUp` action. The planner correctly reported `No plan found`.
- **Test C (Irrelevant Actions):** Added `Move(C, B)`. The planner correctly ignored it and found the optimal plan, showing it does not get distracted by irrelevant moves.

## 6. Logic and Search
**Question: How logical reasoning and search work together:**
Logical reasoning determines what actions are *possible* in a given state by evaluating preconditions and then generating the accurate successor state using the action's effects. Search determines the *sequence* to try by exploring the tree of possible successor states systematically until it encounters a state that logically satisfies the goal.

## 7. Reflection on the use of the LLM
**Can the LLM Verify Its Own Plan?**
You should trust the independently executed state transitions (the Python program) more than the LLM's explanation. LLMs are prone to hallucination and may generate an explanation that looks plausible but contains logical flaws or incorrectly asserts that a precondition is met when it isn't. The program provides an objective, executable verification of logic.

## 8. Reflection Questions
1. **Why is it useful to specify action preconditions and effects before asking an LLM to write the planner?**
   It forces clarity in the problem specification and gives the LLM rigorous logical constraints, minimizing hallucinations and ensuring the generated code adheres to standard planning formalisms.
2. **Give an example of an error that could occur if the planner failed to check an action's preconditions.**
   The robot could execute `Drop(Package, C)` without ever picking it up, magically teleporting the package to the goal.
3. **Why is a plan that "looks reasonable" not necessarily a valid plan?**
   It might rely on an action being performed in a state where subtle preconditions are missed (e.g., dropping an item when the robot isn't in the right room).
4. **What did the LLM contribute to the implementation?**
   It translated the conceptual framework of logic and BFS into working Python syntax (classes, sets, queues).
5. **What did you have to verify independently?**
   The correctness of the generated plan, especially its robustness in edge cases (impossible problems, irrelevant actions).
6. **In this laboratory, where is logical reasoning being used?**
   Within the `is_applicable` and `apply` methods, ensuring state transitions strictly follow propositional logic.
7. **How is planning related to the search algorithms studied in the previous module?**
   Planning is essentially a search problem where the nodes are logical states and the edges are logically applicable actions. We just apply standard search algorithms (like BFS) over a state space defined by logical rules.
