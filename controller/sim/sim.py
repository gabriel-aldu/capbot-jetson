import pygame
import math
import sys
from a_star import astar, get_path_with_directions

# --- Configuration Constants ---
PIXELS_PER_CM = 4
MAZE = [
    [[1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 1]],
    [[0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 1, 0], [1, 1, 0, 0], [0, 1, 0, 1]],
    [[1, 0, 1, 1], [0, 0, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 1]],
    [[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 1]],
    [[0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1]],
    [[0, 1, 1, 0], [0, 1, 0, 1], [1, 1, 1, 0], [1, 1, 0, 0], [0, 1, 0, 1]],
]
CELL_SIZE = 30 * PIXELS_PER_CM
OFFSET = 50
ROWS, COLS = len(MAZE), len(MAZE[0])
SCREEN_SIZE = (COLS * CELL_SIZE + OFFSET * 2, ROWS * CELL_SIZE + OFFSET * 2)

ROBOT_LENGTH = 20 * PIXELS_PER_CM
ROBOT_WIDTH = 16 * PIXELS_PER_CM
LINEAR_SPEED = 35 * PIXELS_PER_CM
ANGULAR_SPEED = 3.0


def closest_point_on_segment(p, a, b):
    ap, ab = (p[0] - a[0], p[1] - a[1]), (b[0] - a[0], b[1] - a[1])
    ab2 = ab[0]**2 + ab[1]**2
    if ab2 == 0: return a
    t = max(0, min(1, (ap[0] * ab[0] + ap[1] * ab[1]) / ab2))
    return (a[0] + t * ab[0], a[1] + t * ab[1])

class RobotEnvironment:
    def __init__(self, start_cell=(0, 0), goal_cell=(4, 1)):
        pygame.init()
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("Robot Environment (Continuous Inputs)")
        self.clock = pygame.time.Clock()
        
        self.start_cell = start_cell
        self.goal_cell = goal_cell
        self.astar_coords = astar(MAZE, start_cell, goal_cell)
        
        # Track velocities internally to pass back via step state
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        
        # Holder for high-resolution line follower path visualization
        self.pixel_path_points = []
        
        self.wall_segments = []
        for r in range(ROWS):
            for c in range(COLS):
                x_l, x_r = OFFSET + c * CELL_SIZE, OFFSET + (c + 1) * CELL_SIZE
                y_t, y_b = OFFSET + r * CELL_SIZE, OFFSET + (r + 1) * CELL_SIZE
                if MAZE[r][c][0]: self.wall_segments.append(((x_l, y_t), (x_r, y_t)))
                if MAZE[r][c][1]: self.wall_segments.append(((x_l, y_b), (x_r, y_b)))
                if MAZE[r][c][2]: self.wall_segments.append(((x_l, y_t), (x_l, y_b)))
                if MAZE[r][c][3]: self.wall_segments.append(((x_r, y_t), (x_r, y_b)))
        
        self.reset()

    def get_path(self):
        direction_path = get_path_with_directions(self.astar_coords)
        return direction_path

    def set_continuous_path(self, line_path):
        """ Receives centimeter path points and converts them to screen pixels """
        self.pixel_path_points = []
        for pt_x, pt_y, _ in line_path:
            px = pt_x * PIXELS_PER_CM + (OFFSET + 0.5 * CELL_SIZE)
            py = pt_y * PIXELS_PER_CM + (OFFSET + 0.5 * CELL_SIZE)
            self.pixel_path_points.append((int(px), int(py)))
    
    def reset(self):
        center_x = OFFSET + (self.start_cell[1] + 0.5) * CELL_SIZE
        center_y = OFFSET + (self.start_cell[0] + 0.5) * CELL_SIZE
        self.theta = 0.0
        self.x = center_x - (ROBOT_LENGTH / 4) * math.cos(self.theta)
        self.y = center_y - (ROBOT_LENGTH / 4) * math.sin(self.theta)
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.collision_radius = 36
        return self.get_state()

    def get_state(self):
        return {
            "x": self.x, 
            "y": self.y, 
            "theta": self.theta,
            "linear_vel": self.linear_vel,
            "angular_vel": self.angular_vel
        }

    def step(self, action, dt=1/60.0):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        linear_throttle = max(-1.0, min(1.0, float(action[0])))
        angular_throttle = max(-1.0, min(1.0, float(action[1])))

        self.linear_vel = linear_throttle * LINEAR_SPEED
        self.angular_vel = angular_throttle * ANGULAR_SPEED

        self.theta += self.angular_vel * dt
        self.x += self.linear_vel * math.cos(self.theta) * dt
        self.y += self.linear_vel * math.sin(self.theta) * dt

        cos_t, sin_t = math.cos(self.theta), math.sin(self.theta)
        center_x = self.x + (ROBOT_LENGTH / 4) * cos_t
        center_y = self.y + (ROBOT_LENGTH / 4) * sin_t

        for seg_a, seg_b in self.wall_segments:
            closest = closest_point_on_segment((center_x, center_y), seg_a, seg_b)
            dist_x, dist_y = center_x - closest[0], center_y - closest[1]
            distance = math.hypot(dist_x, dist_y)

            if distance < self.collision_radius and distance > 0:
                overlap = self.collision_radius - distance
                center_x += (dist_x / distance) * overlap
                center_y += (dist_y / distance) * overlap

        self.x = center_x - (ROBOT_LENGTH / 4) * cos_t
        self.y = center_y - (ROBOT_LENGTH / 4) * sin_t

        goal_x = OFFSET + (self.goal_cell[1] + 0.5) * CELL_SIZE
        goal_y = OFFSET + (self.goal_cell[0] + 0.5) * CELL_SIZE
        done = math.hypot(center_x - goal_x, center_y - goal_y) < 20

        return self.get_state(), done

    def render(self):
        self.screen.fill((245, 245, 245))

        # Goal Target
        goal_x = OFFSET + (self.goal_cell[1] + 0.5) * CELL_SIZE
        goal_y = OFFSET + (self.goal_cell[0] + 0.5) * CELL_SIZE
        pygame.draw.circle(self.screen, (255, 200, 200), (int(goal_x), int(goal_y)), 30)
        pygame.draw.circle(self.screen, (220, 20, 60), (int(goal_x), int(goal_y)), 12)

        # Draw A* Guide Track (Thick green behind)
        if len(self.astar_coords) > 1:
            path_points = [(OFFSET + (c + 0.5) * CELL_SIZE, OFFSET + (r + 0.5) * CELL_SIZE) for r, c in self.astar_coords]
            pygame.draw.lines(self.screen, (210, 245, 210), False, path_points, 8)

        # Draw Continuous Line Follower Path Track (Purple line)
        if len(self.pixel_path_points) > 1:
            pygame.draw.lines(self.screen, (148, 0, 211), False, self.pixel_path_points, 3)

        # Draw Walls
        for seg_a, seg_b in self.wall_segments:
            pygame.draw.line(self.screen, (200, 30, 30), seg_a, seg_b, 5)

        # Draw Robot Chassis
        cos_t, sin_t = math.cos(self.theta), math.sin(self.theta)
        center_x = self.x + (ROBOT_LENGTH / 4) * cos_t
        center_y = self.y + (ROBOT_LENGTH / 4) * sin_t
        half_l, half_w = ROBOT_LENGTH / 2, ROBOT_WIDTH / 2
        
        corners = [
            (center_x + half_l * cos_t - half_w * sin_t, center_y + half_l * sin_t + half_w * cos_t),
            (center_x + half_l * cos_t + half_w * sin_t, center_y + half_l * sin_t - half_w * cos_t),
            (center_x - half_l * cos_t + half_w * sin_t, center_y - half_l * sin_t - half_w * cos_t),
            (center_x - half_l * cos_t - half_w * sin_t, center_y - half_l * sin_t + half_w * cos_t)
        ]

        for side in [-1, 1]:
            wx = self.x - (half_w * side) * sin_t
            wy = self.y + (half_w * side) * cos_t
            w_start = (wx - 13 * cos_t, wy - 13 * sin_t)
            w_end = (wx + 13 * cos_t, wy + 13 * sin_t)
            pygame.draw.line(self.screen, (40, 40, 40), w_start, w_end, 12)

        pygame.draw.polygon(self.screen, (30, 144, 255), corners)
        pygame.draw.polygon(self.screen, (0, 105, 180), corners, 3)

        front_mid_x = center_x + half_l * cos_t
        front_mid_y = center_y + half_l * sin_t
        pygame.draw.line(self.screen, (255, 255, 255), (self.x, self.y), (front_mid_x, front_mid_y), 3)
        pygame.draw.circle(self.screen, (255, 69, 0), (int(front_mid_x), int(front_mid_y)), 6)

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()