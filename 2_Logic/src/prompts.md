I want to implement a simple planning agent in Python.
Represent a state as a set of logical propositions.
Each action should contain:
- a name;
- positive preconditions;
- negative preconditions;
- positive effects;
- negative effects.
An action is applicable if all of its preconditions are satisfied by the current state.
When an action is applied:
1. remove its negative effects from the state;
2. add its positive effects to the state.
Use breadth-first search to find a sequence of actions that achieves a specified goal.
The program should also:
- detect when no plan exists;
- print the resulting sequence of actions;
- print the states reached after each action.
Explain the implementation and identify any assumptions you make.
