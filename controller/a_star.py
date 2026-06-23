import heapq
import matplotlib.pyplot as plt

#(up,down,left,right)
# 1. Your complete 5x5 labyrinth map
maze = [
    [[1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 1]],
    [[0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 1, 0], [1, 1, 0, 0], [0, 1, 0, 1]],
    [[1, 0, 1, 1], [0, 0, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 1]],
    [[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 1]],
    [[0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1]],
    [[0, 1, 1, 0], [0, 1, 0, 1], [1, 1, 1, 0], [1, 1, 0, 0], [0, 1, 0, 1]],
]

# 2. The A* Algorithm Function
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    open_list = []
    heapq.heappush(open_list, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    
    while open_list:
        _, current = heapq.heappop(open_list)
        
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        directions = [
            ((current[0] - 1, current[1]), 0),  # UP
            ((current[0] + 1, current[1]), 1),  # DOWN
            ((current[0], current[1] - 1), 2),  # LEFT
            ((current[0], current[1] + 1), 3)   # RIGHT
        ]
        
        for neighbor, wall_idx in directions:
            r, c = neighbor
            if 0 <= r < rows and 0 <= c < cols:
                if maze[current[0]][current[1]][wall_idx] == 1:
                    continue
                
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    if neighbor not in [i[1] for i in open_list]:
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))
    return None

def get_path_with_directions(path):
    if not path:
        return []
        
    directed_path = []
    # The starting cell doesn't have a previous action heading into it
    directed_path.append((path[0], "start"))
    
    for i in range(1, len(path)):
        prev_r, prev_c = path[i-1]
        curr_r, curr_c = path[i]
        
        # Determine movement delta
        dr = curr_r - prev_r
        dc = curr_c - prev_c
        
        if dr == -1:
            direction = "up"
        elif dr == 1:
            direction = "down"
        elif dc == -1:
            direction = "left"
        elif dc == 1:
            direction = "right"
        else:
            direction = "unknown"
            
        directed_path.append((path[i], direction))
        
    return directed_path

# 3. Matplotlib Rendering Function
def draw_maze(maze, path=None, start=(0,0), end=(4,1)):
    rows = len(maze)
    cols = len(maze[0])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Draw walls based on cell configuration
    # Grid coordinates mapping: 
    # row index increases DOWN (y axis needs to be inverted or carefully planned)
    # We will treat row as Y (going down) and col as X (going right)
    for r in range(rows):
        for c in range(cols):
            cell_walls = maze[r][c]
            
            # Boundaries of the current cell square
            x_left, x_right = c, c + 1
            y_top, y_bottom = rows - r, rows - (r + 1)
            
            # UP Wall (index 0)
            if cell_walls[0]:
                ax.plot([x_left, x_right], [y_top, y_top], color='red', linewidth=3)
            # DOWN Wall (index 1)
            if cell_walls[1]:
                ax.plot([x_left, x_right], [y_bottom, y_bottom], color='red', linewidth=3)
            # LEFT Wall (index 2)
            if cell_walls[2]:
                ax.plot([x_left, x_left], [y_top, y_bottom], color='red', linewidth=3)
            # RIGHT Wall (index 3)
            if cell_walls[3]:
                ax.plot([x_right, x_right], [y_top, y_bottom], color='red', linewidth=3)
                
    # Mark Start and End positions (placing markers in the center of the cell)
    ax.plot(start[1] + 0.5, rows - start[0] - 0.5, 'go', markersize=12, label='Start')
    ax.plot(end[1] + 0.5, rows - end[0] - 0.5, 'ro', markersize=12, label='Goal')

    # Draw the Path if found
    if path:
        path_x = [c + 0.5 for (r, c) in path]
        path_y = [rows - r - 0.5 for (r, c) in path]
        ax.plot(path_x, path_y, color='blue', linewidth=4, linestyle='-', label='A* Path')

    # Formatting the plot
    ax.set_xlim(-0.5, cols + 0.5)
    ax.set_ylim(-0.5, rows + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("Labyrinth Map with A* Calculated Path", fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    # --- Execution ---
    start_pos = (5, 2)
    goal_pos = (0, 0)
    calculated_path = astar(maze, start_pos, goal_pos)

    draw_maze(maze, path=calculated_path, start=start_pos, end=goal_pos)

    final_path_with_directions = get_path_with_directions(calculated_path)
    for step in final_path_with_directions:
        print(f"Cell: {step[0]} -> Headed: {step[1].upper()}")

