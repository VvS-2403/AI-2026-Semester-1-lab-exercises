from collections import deque

class Action:
    def __init__(self, name, pos_pre=None, neg_pre=None, pos_eff=None, neg_eff=None):
        self.name = name
        self.pos_pre = pos_pre if pos_pre else set()
        self.neg_pre = neg_pre if neg_pre else set()
        self.pos_eff = pos_eff if pos_eff else set()
        self.neg_eff = neg_eff if neg_eff else set()

    def is_applicable(self, state):
        return self.pos_pre.issubset(state) and len(self.neg_pre.intersection(state)) == 0

    def apply(self, state):
        new_state = state.copy()
        new_state.difference_update(self.neg_eff)
        new_state.update(self.pos_eff)
        return new_state

def bfs_plan(initial_state, goal_state, actions):
    queue = deque([(initial_state, [])])
    visited = []
    
    while queue:
        current_state, plan = queue.popleft()
        
        if goal_state.issubset(current_state):
            print("Plan found!")
            print(f"Initial State: {initial_state}")
            for step, (action, state) in enumerate(plan):
                print(f"Step {step+1}: {action.name}")
                print(f"State: {state}")
            return plan
        
        if current_state not in visited:
            visited.append(current_state)
            
            for action in actions:
                if action.is_applicable(current_state):
                    new_state = action.apply(current_state)
                    new_plan = plan + [(action, new_state)]
                    queue.append((new_state, new_plan))
                    
    print("No plan found")
    return None

def test_a():
    print("\n--- Test A: Solvable Problem ---")
    initial_state = {"At(Robot,A)", "At(Package,A)"}
    goal = {"At(Package,C)"}
    
    actions = [
        Action("Move(A,B)", pos_pre={"At(Robot,A)"}, pos_eff={"At(Robot,B)"}, neg_eff={"At(Robot,A)"}),
        Action("Move(B,C)", pos_pre={"At(Robot,B)"}, pos_eff={"At(Robot,C)"}, neg_eff={"At(Robot,B)"}),
        Action("PickUp(Package,A)", pos_pre={"At(Robot,A)", "At(Package,A)"}, pos_eff={"Holding(Package)"}, neg_eff={"At(Package,A)"}),
        Action("Drop(Package,C)", pos_pre={"At(Robot,C)", "Holding(Package)"}, pos_eff={"At(Package,C)"}, neg_eff={"Holding(Package)"})
    ]
    
    bfs_plan(initial_state, goal, actions)

def test_b():
    print("\n--- Test B: Impossible Problem ---")
    initial_state = {"At(Robot,A)", "At(Package,A)"}
    goal = {"At(Package,C)"}
    
    actions = [
        Action("Move(A,B)", pos_pre={"At(Robot,A)"}, pos_eff={"At(Robot,B)"}, neg_eff={"At(Robot,A)"}),
        Action("Move(B,C)", pos_pre={"At(Robot,B)"}, pos_eff={"At(Robot,C)"}, neg_eff={"At(Robot,B)"}),
        Action("Drop(Package,C)", pos_pre={"At(Robot,C)", "Holding(Package)"}, pos_eff={"At(Package,C)"}, neg_eff={"Holding(Package)"})
    ]
    
    bfs_plan(initial_state, goal, actions)

def test_c():
    print("\n--- Test C: Irrelevant Actions ---")
    initial_state = {"At(Robot,A)", "At(Package,A)"}
    goal = {"At(Package,C)"}
    
    actions = [
        Action("Move(A,B)", pos_pre={"At(Robot,A)"}, pos_eff={"At(Robot,B)"}, neg_eff={"At(Robot,A)"}),
        Action("Move(B,C)", pos_pre={"At(Robot,B)"}, pos_eff={"At(Robot,C)"}, neg_eff={"At(Robot,B)"}),
        Action("PickUp(Package,A)", pos_pre={"At(Robot,A)", "At(Package,A)"}, pos_eff={"Holding(Package)"}, neg_eff={"At(Package,A)"}),
        Action("Drop(Package,C)", pos_pre={"At(Robot,C)", "Holding(Package)"}, pos_eff={"At(Package,C)"}, neg_eff={"Holding(Package)"}),
        Action("Move(C,B)", pos_pre={"At(Robot,C)"}, pos_eff={"At(Robot,B)"}, neg_eff={"At(Robot,C)"}),
    ]
    
    bfs_plan(initial_state, goal, actions)

if __name__ == "__main__":
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    with open('results/tests_output.txt', 'w') as f:
        import sys
        sys.stdout = f
        test_a()
        test_b()
        test_c()
