from collections import deque

warehouse_map = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################"
]

def solve_warehouse(grid):
    """
    Solves the warehouse navigation problem using Breadth-First Search (BFS).
    BFS is appropriate here because all moves have equal cost (1 step),
    and BFS guarantees finding the shortest path in such an unweighted grid.
    """
    start = None
    goal = None
    rows = len(grid)
    cols = len(grid[0])
    
    # Find start (S) and goal (G) positions
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 'S':
                start = (r, c)
            elif grid[r][c] == 'G':
                goal = (r, c)
                
    if not start or not goal:
        return "Start or Goal not found"
        
    # BFS initialization
    queue = deque([(start, [])])
    visited = set()
    visited.add(start)
    
    # Directions: (row_delta, col_delta, move_name)
    directions = [(-1, 0, 'Up'), (1, 0, 'Down'), (0, -1, 'Left'), (0, 1, 'Right')]
    
    while queue:
        (r, c), path = queue.popleft()
        
        if (r, c) == goal:
            return path
            
        for dr, dc, move in directions:
            nr, nc = r + dr, c + dc
            # Check bounds and obstacles
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '#' and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [move]))
                
    return "No path exists"

if __name__ == "__main__":
    print("Warehouse Map:")
    for row in warehouse_map:
        print(row)
    
    print("\nExplanation of Search Algorithm:")
    print("The agent uses Breadth-First Search (BFS) to find the path.")
    print("BFS systematically explores the warehouse layer by layer, guaranteeing the shortest path in an unweighted grid without getting stuck in infinite loops.")
    
    print("\nSearching for path...")
    path = solve_warehouse(warehouse_map)
    print(f"\nResult Path ({len(path)} steps):")
    print(path)
