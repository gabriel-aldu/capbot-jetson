import pygame
import math
import sys
import heapq

# --- 1. Conversion Scale & Layout ---
PIXELS_PER_CM = 4  # 4 pixels equals 1 centimeter

# Labyrinth Definition (6 rows x 5 columns)
MAZE = [
    [[1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 1]],
    [[0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 1, 0], [1, 1, 0, 0], [0, 1, 0, 1]],
    [[1, 0, 1, 1], [0, 0, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 1]],
    [[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 1]],
    [[0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1]],
    [[0, 1, 1, 0], [0, 1, 0, 1], [1, 1, 1, 0], [1, 1, 0, 0], [0, 1, 0, 1]],
]

CELL_SIZE_CM = 30                        # Each cell is 30cm x 30cm
CELL_SIZE = CELL_SIZE_CM * PIXELS_PER_CM # 120 pixels

OFFSET = 50
ROWS = len(MAZE)
COLS = len(MAZE[0])
SCREEN_SIZE = (COLS * CELL_SIZE + OFFSET * 2, ROWS * CELL_SIZE + OFFSET * 2)

# --- Robot Dimensions ---
ROBOT_LENGTH_CM = 20
ROBOT_WIDTH_CM = 16
ROBOT_LENGTH = ROBOT_LENGTH_CM * PIXELS_PER_CM  # 80 pixels
ROBOT_WIDTH = ROBOT_WIDTH_CM * PIXELS_PER_CM    # 64 pixels

ROBOT_SPEED_CMS = 35                            # 35 cm per second forward speed
LINEAR_SPEED = ROBOT_SPEED_CMS * PIXELS_PER_CM  # 140 pixels/sec
ANGULAR_SPEED = 3.0                             # Radians per second

# --- 2. A* Pathfinding (for background guide) ---
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_astar_path(maze, start, end):
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
        
        directions = [((current[0]-1, current[1]), 0), ((current[0]+1, current[1]), 1),
                      ((current[0], current[1]-1), 2), ((current[0], current[1]+1), 3)]
        
        for neighbor, wall_idx in directions:
            r, c = neighbor
            if 0 <= r < ROWS and 0 <= c < COLS:
                if maze[current[0]][current[1]][wall_idx] == 1:
                    continue
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                    if neighbor not in [i[1] for i in open_list]:
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))
    return []

# --- 3. Collision Helper ---
def closest_point_on_segment(p, a, b):
    ap = (p[0] - a[0], p[1] - a[1])
    ab = (b[0] - a[0], b[1] - a[1])
    ab2 = ab[0]**2 + ab[1]**2
    if ab2 == 0: return a
    t = max(0, min(1, (ap[0] * ab[0] + ap[1] * ab[1]) / ab2))
    return (a[0] + t * ab[0], a[1] + t * ab[1])

# --- 4. Two-Wheeled Box Robot Class ---
class TwoWheeledRobot:
    def __init__(self, start_cell):
        self.theta = 0.0  # Heading angle in radians
        self.length = ROBOT_LENGTH
        self.width = ROBOT_WIDTH
        
        # Standard differential steering rotates around its wheel axle midpoint.
        # Place the physical geometric center in the middle of the starting cell,
        # then offset the wheel axle position 1/4 length from the back (which is 1/4 length behind center).
        center_x = OFFSET + (start_cell[1] + 0.5) * CELL_SIZE
        center_y = OFFSET + (start_cell[0] + 0.5) * CELL_SIZE
        
        self.x = center_x - (self.length / 4) * math.cos(self.theta)
        self.y = center_y - (self.length / 4) * math.sin(self.theta)
        
        # Bounding circle radius optimized for a clean box sliding clearance
        self.collision_radius = 36 

    def update(self, dt, keys, wall_segments):
        # Handle Input (Differential steering simulation around wheel axle)
        v = 0
        omega = 0
        if keys[pygame.K_w]: v += LINEAR_SPEED
        if keys[pygame.K_s]: v -= LINEAR_SPEED
        if keys[pygame.K_a]: omega -= ANGULAR_SPEED
        if keys[pygame.K_d]: omega += ANGULAR_SPEED

        # Update pose variables at the kinematic center (the axle)
        self.theta += omega * dt
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt

        # Calculate current geometric center for physical collision boundaries
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)
        center_x = self.x + (self.length / 4) * cos_t
        center_y = self.y + (self.length / 4) * sin_t

        # Smooth Sliding Collision Detection centered on physical body
        for seg_a, seg_b in wall_segments:
            closest = closest_point_on_segment((center_x, center_y), seg_a, seg_b)
            dist_x = center_x - closest[0]
            dist_y = center_y - closest[1]
            distance = math.hypot(dist_x, dist_y)

            if distance < self.collision_radius:
                overlap = self.collision_radius - distance
                if distance == 0:
                    continue
                center_x += (dist_x / distance) * overlap
                center_y += (dist_y / distance) * overlap

        # Re-derive kinematics position (axle tracking) based on collision adjusted body position
        self.x = center_x - (self.length / 4) * cos_t
        self.y = center_y - (self.length / 4) * sin_t

    def draw(self, surface):
        half_l = self.length / 2
        half_w = self.width / 2

        # Direction unit vectors
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)
        
        # Calculate shifted geometric center (1/4 of total length forward from axle)
        center_x = self.x + (self.length / 4) * cos_t
        center_y = self.y + (self.length / 4) * sin_t
        
        # Calculate rotated rectangle corners relative to geometric center
        corners = [
            (center_x + half_l * cos_t - half_w * sin_t, center_y + half_l * sin_t + half_w * cos_t),
            (center_x + half_l * cos_t + half_w * sin_t, center_y + half_l * sin_t - half_w * cos_t),
            (center_x - half_l * cos_t + half_w * sin_t, center_y - half_l * sin_t - half_w * cos_t),
            (center_x - half_l * cos_t - half_w * sin_t, center_y - half_l * sin_t + half_w * cos_t)
        ]

        # Draw Left & Right Wheels mounted exactly on the axle axis (self.x, self.y)
        wheel_width = 12
        wheel_length = 26
        for side in [-1, 1]:  # -1 = Right side, 1 = Left side
            wx = self.x - (half_w * side) * sin_t
            wy = self.y + (half_w * side) * cos_t
            
            w_start = (wx - (wheel_length / 2) * cos_t, wy - (wheel_length / 2) * sin_t)
            w_end = (wx + (wheel_length / 2) * cos_t, wy + (wheel_length / 2) * sin_t)
            pygame.draw.line(surface, (40, 40, 40), w_start, w_end, wheel_width)

        # Draw Rotated Box Base Chassis
        pygame.draw.polygon(surface, (30, 144, 255), corners)     # Main body fill
        pygame.draw.polygon(surface, (0, 105, 180), corners, 3)   # Outer border

        # Draw Direction Indicator Vector (From Axle center through Front bumper center)
        front_mid_x = center_x + half_l * cos_t
        front_mid_y = center_y + half_l * sin_t
        pygame.draw.line(surface, (255, 255, 255), (self.x, self.y), (front_mid_x, front_mid_y), 3)
        pygame.draw.circle(surface, (255, 69, 0), (int(front_mid_x), int(front_mid_y)), 6)


# --- 5. Main Simulator Execution ---
def main():
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption("Two-Wheeled Rectangular Robot Simulator")
    clock = pygame.time.Clock()

    start_cell = (0, 0)
    goal_cell = (4, 1)

    # Pre-calculate A* Path for visual guidance
    astar_coords = get_astar_path(MAZE, start_cell, goal_cell)
    robot = TwoWheeledRobot(start_cell)

    # Process maze structure into line segments
    wall_segments = []
    for r in range(ROWS):
        for c in range(COLS):
            x_l, x_r = OFFSET + c * CELL_SIZE, OFFSET + (c + 1) * CELL_SIZE
            y_t, y_b = OFFSET + r * CELL_SIZE, OFFSET + (r + 1) * CELL_SIZE
            
            if MAZE[r][c][0]: wall_segments.append(((x_l, y_t), (x_r, y_t)))  # UP
            if MAZE[r][c][1]: wall_segments.append(((x_l, y_b), (x_r, y_b)))  # DOWN
            if MAZE[r][c][2]: wall_segments.append(((x_l, y_t), (x_l, y_b)))  # LEFT
            if MAZE[r][c][3]: wall_segments.append(((x_r, y_t), (x_r, y_b)))  # RIGHT

    # Simulation Loop
    while True:
        dt = clock.tick(60) / 1000.0  # Time step delta

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Update Robot position and handle wall interactions
        keys = pygame.key.get_pressed()
        robot.update(dt, keys, wall_segments)

        # Background Fill
        screen.fill((245, 245, 245))

        # Draw Goal Area Target
        goal_x = OFFSET + (goal_cell[1] + 0.5) * CELL_SIZE
        goal_y = OFFSET + (goal_cell[0] + 0.5) * CELL_SIZE
        pygame.draw.circle(screen, (255, 200, 200), (int(goal_x), int(goal_y)), 30)
        pygame.draw.circle(screen, (220, 20, 60), (int(goal_x), int(goal_y)), 12)

        # Draw A* Guided Pathway Track
        if len(astar_coords) > 1:
            path_points = [
                (OFFSET + (c + 0.5) * CELL_SIZE, OFFSET + (r + 0.5) * CELL_SIZE)
                for r, c in astar_coords
            ]
            pygame.draw.lines(screen, (180, 230, 180), False, path_points, 8)

        # Draw Labyrinth Map Walls
        for seg_a, seg_b in wall_segments:
            pygame.draw.line(screen, (200, 30, 30), seg_a, seg_b, 5)

        # Draw Box Robot
        robot.draw(screen)

        pygame.display.flip()

if __name__ == "__main__":
    main()