import heapq
from collections import deque

def parse_map(map_str):
    grid = map_str.strip().split('\n')
    start = None
    goal = None
    obstacles = set()
    for y, row in enumerate(grid):
        for x, char in enumerate(row):
            if char == 'S':
                start = (x, y)
            elif char == 'G':
                goal = (x, y)
            elif char == '#':
                obstacles.add((x, y))
    return start, goal, obstacles, len(grid[0]), len(grid)

def get_neighbors(state, obstacles, width, height):
    x, y = state
    neighbors = []
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles:
            neighbors.append((nx, ny))
    return neighbors

def bfs(map_str):
    start, goal, obstacles, width, height = parse_map(map_str)
    
    if start is None or goal is None:
        return None, 0, 0
    
    frontier = deque([(start, [start])])
    visited = {start}
    expanded = 0
    
    while frontier:
        state, path = frontier.popleft()
        expanded += 1
        
        if state == goal:
            return path, len(path) - 1, expanded
            
        for neighbor in get_neighbors(state, obstacles, width, height):
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append((neighbor, path + [neighbor]))
                
    return None, 0, expanded

def a_star(map_str, heuristic_func):
    start, goal, obstacles, width, height = parse_map(map_str)
    
    if start is None or goal is None:
        return None, 0, 0
        
    frontier = [(heuristic_func(start, goal), 0, start, [start])]
    visited = {}
    expanded = 0
    
    while frontier:
        f, g, state, path = heapq.heappop(frontier)
        
        if state in visited and visited[state] <= g:
            continue
            
        visited[state] = g
        expanded += 1
        
        if state == goal:
            return path, len(path) - 1, expanded
            
        for neighbor in get_neighbors(state, obstacles, width, height):
            new_g = g + 1
            if neighbor not in visited or new_g < visited[neighbor]:
                new_f = new_g + heuristic_func(neighbor, goal)
                heapq.heappush(frontier, (new_f, new_g, neighbor, path + [neighbor]))
                
    return None, 0, expanded

def h_manhattan(state, goal):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

def h_zero(state, goal):
    return 0

def h_euclidean(state, goal):
    return ((state[0] - goal[0])**2 + (state[1] - goal[1])**2)**0.5

def h_double_manhattan(state, goal):
    return 2 * h_manhattan(state, goal)

if __name__ == '__main__':
    map_original = """
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################
    """
    map_trivial = """
#####
#SG##
#####
    """
    map_no_solution = """
#######
#S....#
###.###
#...#G#
#######
    """
    map_alternative = """
#######
#S....#
#.....#
#....G#
#######
    """

    results = []

    def run_test(name, map_str, algo, h=None):
        if algo == 'bfs':
            path, length, exp = bfs(map_str)
        else:
            path, length, exp = a_star(map_str, h)
        res = f"Test: {name}\nPath found: {path is not None}\nLength: {length}\nStates expanded: {exp}\n"
        results.append(res)
        return path, length, exp

    # Task 3
    run_test("Original Map A* (Manhattan)", map_original, 'a_star', h_manhattan)
    run_test("Trivial Map A*", map_trivial, 'a_star', h_manhattan)
    run_test("No Solution Map A*", map_no_solution, 'a_star', h_manhattan)
    run_test("Alternative Map A*", map_alternative, 'a_star', h_manhattan)
    
    # Task 5: BFS vs A*
    run_test("Original Map BFS", map_original, 'bfs')
    
    # Task 6: Heuristic Investigation
    run_test("Original Map A* (h=0)", map_original, 'a_star', h_zero)
    run_test("Original Map A* (Euclidean)", map_original, 'a_star', h_euclidean)
    run_test("Original Map A* (2*Manhattan)", map_original, 'a_star', h_double_manhattan)

    with open('../results/experiments.txt', 'w') as f:
        f.write("\n".join(results))
    print("Experiments run successfully.")
