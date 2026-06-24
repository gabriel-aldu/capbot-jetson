import heapq

#(up,down,left,right)
# 1. Your complete 5x6 labyrinth map
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

