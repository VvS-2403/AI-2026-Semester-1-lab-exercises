# Search Lab - Warehouse Robot Navigation

**Student Name:** Vismay
**Course:** Artificial Intelligence

## 1. Formulation of the Search Problem

*   **State $S$**: A tuple `(x, y)` representing the current coordinates of the robot on the 2D grid.
*   **Actions $A$**: A set of valid moves `{Up, Down, Left, Right}`. An action is valid if the adjacent cell is within the grid boundaries and is not an obstacle (`#`).
*   **Transition $T$**: Given a state `(x, y)` and an action, the transition function returns the new coordinates `(x', y')`. For example, `(x, y) + Right -> (x+1, y)`.
*   **Initial state $s_0$**: The coordinate where the `S` character is located.
*   **Goal $G$**: The coordinate where the `G` character is located.
*   **Cost $c$**: Every valid movement has a cost of $1$.

**(a) What information is necessary to specify a state?**
Only the current `(x, y)` position of the robot in the grid.

**(b) What makes an action invalid?**
Moving out of the bounds of the grid or moving into a cell containing an obstacle (`#`).

**(c) Is this a deterministic search problem?**
Yes, every action has exactly one predictable outcome.

**(d) What would constitute a solution?**
A sequence of valid actions (or the resulting path of states) from the initial state to the goal state.

---

## 2. Design of the Agent

1.  **State Representation in Python:** A tuple of two integers `(x, y)`.
2.  **Warehouse Representation:** A set of obstacle coordinates `obstacles = {(x1, y1), ...}`, along with the `width` and `height` of the grid.
3.  **Valid Actions Determination:** Checking if `(nx, ny) = (x + dx, y + dy)` is $0 \le nx < width$, $0 \le ny < height$, and `(nx, ny)` not in `obstacles`.
4.  **Goal Recognition:** Checking if the current state equals the goal state tuple `(x, y) == goal`.
5.  **Information Stored in the Frontier:** For A*, a priority queue storing tuples of `(f_score, g_score, state, path)`. The path is a list of states from the start to the current state.
6.  **Path Reconstruction:** The path is maintained within the frontier elements, so when the goal is popped, the path is already fully reconstructed. (Alternatively, a `came_from` dictionary could be used).

---

## 3. Final Python Program

The final implementation of the search algorithms and testing code can be found at:
[src/search.py](file:///c:/Users/Vismay%20VS/Desktop/Artificial%20Intelligence/Lab%20Exercises/1_Search/src/search.py)

---

## 4. Prompts Used with the LLM

The prompt used to generate the base structure of the code is stored here:
[src/prompts.txt](file:///c:/Users/Vismay%20VS/Desktop/Artificial%20Intelligence/Lab%20Exercises/1_Search/src/prompts.txt)

---

## 5. Results of the Tests

All tests were successfully implemented and run. The output of the experiments can be found in:
[results/experiments.txt](file:///c:/Users/Vismay%20VS/Desktop/Artificial%20Intelligence/Lab%20Exercises/1_Search/results/experiments.txt)

*   **Original Warehouse:** Found a path of length 40, expanding 64 states.
*   **Trivial Case:** Found a path of length 1, expanding 2 states.
*   **No Solution:** Correctly reported failure after expanding 9 states (all reachable states in the isolated starting area).
*   **Alternative Paths:** Found the optimal shortest path of length 6 instead of taking longer detours.

---

## 6. BFS vs A* Comparison

*(Tested on the Original Warehouse map)*

| Measure | BFS | A* |
| :--- | :--- | :--- |
| Solution found | Yes | Yes |
| Path length | 40 | 40 |
| States expanded | 64 | 64 |

**(a) Did both algorithms find a solution?** Yes.
**(b) Did they find paths of the same length?** Yes, both algorithms guarantee finding the shortest path (optimal solution), so the length is exactly 40 for both.
**(c) Which algorithm expanded fewer states?** In this specific maze-like map, both expanded the same number of states (64).
**(d) Why might A* expand fewer states?** A* uses the heuristic to guide the search towards the goal. In maps with fewer restrictive walls, A* heavily prioritizes nodes closer to the goal, avoiding the blind, radial expansion that BFS performs. In our specific map, the extensive walls act as large dead-ends which forced A* to eventually explore almost everything anyway, making its performance equivalent to BFS.

---

## 7. Heuristic Investigation

We tested A* on the original warehouse map using different heuristics:

1.  **$h(n) = 0$**: Path length 40, States expanded: 64. (Becomes identical to BFS/Dijkstra)
2.  **$h(n) = \text{Euclidean Distance}$**: Path length 40, States expanded: 64.
3.  **$h(n) = 2 \times \text{Manhattan Distance}$**: Path length 40, States expanded: 67.

**Why is Manhattan distance appropriate?**
Since the robot can only move horizontally and vertically (no diagonals), the Manhattan distance precisely represents the minimum number of steps required to reach the goal if there were no obstacles. It is therefore an *admissible* (never overestimates the true cost) and highly accurate heuristic for grid environments.

**What happens when the heuristic becomes too optimistic or too aggressive?**
*   **Too optimistic (e.g., $h(n) = 0$ or Euclidean)**: The heuristic is still admissible, so optimality is guaranteed. However, because the estimate is lower than the actual cost, the algorithm behaves less aggressively towards the goal and ends up expanding more states (resembling blind search).
*   **Too aggressive ($2 \times$ Manhattan)**: The heuristic is no longer admissible because it overestimates the true cost. This breaks the optimality guarantee of A*. Furthermore, as seen in the experiment (expanding 67 states instead of 64), non-admissible heuristics can lead to unnecessary re-expansions of nodes because a sub-optimal path to a state might be discovered first, and later updated when a better path is found.

---

## 8. Final Reflection

**1. Why is it important to formulate the search problem before writing the search algorithm?**
Formulating the problem abstracts away the domain-specific details into a standardized mathematical model (states, actions, transitions, costs, goals). This separation of concerns allows us to apply generic search algorithms like BFS or A* without hardcoding warehouse-specific logic into the algorithm itself, ensuring correctness and reusability.

**2. In what sense is A* an “informed” search algorithm?**
A* is informed because it uses problem-specific knowledge, encapsulated in the heuristic function $h(n)$, to estimate the remaining cost to the goal. This guides the search toward promising states rather than exploring blindly like BFS or DFS.

**3. Why does the choice of heuristic matter?**
The heuristic determines the efficiency and optimality of A*. An admissible heuristic guarantees finding the shortest path, while a tighter (more accurate) heuristic reduces the number of states expanded. An overly optimistic heuristic (e.g., $h(n)=0$) degrades A* to Uniform Cost Search, and an overestimating heuristic can lose the optimality guarantee and cause unnecessary re-expansions.

**4. What did the LLM contribute to the engineering process?**
The LLM acted as a coding assistant, rapidly translating the formal design and specification into functional Python code. It handled the boilerplate of parsing the grid, setting up the priority queue, and implementing the core A* logic, allowing me to focus on problem formulation, testing, and algorithmic evaluation.

**5. What could go wrong if an engineer simply accepted LLM-generated code without testing it?**
LLM-generated code is a hypothesis and might contain logical bugs, incorrect data structure usage, or edge-case failures. Without testing, the program might fail silently, return suboptimal paths, enter infinite loops, or crash, leading to unreliable and potentially dangerous software in real-world scenarios.
