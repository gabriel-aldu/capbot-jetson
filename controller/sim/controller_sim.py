import pygame
import time
from sim import RobotEnvironment, OFFSET, CELL_SIZE, PIXELS_PER_CM, LINEAR_SPEED, ANGULAR_SPEED
import math

def get_line_path(steps, steps_per_cell=30):
    if not steps:
        return []
    waypoints = [(col * 30.0, row * 30.0) for (row, col), _ in steps]
    path = []
    for i in range(len(waypoints) - 1):
        x_start, y_start = waypoints[i]
        x_end, y_end = waypoints[i+1]
        dx = x_end - x_start
        dy = y_end - y_start
        segment_theta = math.atan2(dy, dx)
        for step in range(steps_per_cell):
            t = step / steps_per_cell
            interp_x = x_start + t * dx
            interp_y = y_start + t * dy
            if i == 0 and step == 0:
                current_theta = 0.0
            else:
                current_theta = segment_theta
            path.append((round(interp_x, 2), round(interp_y, 2), round(current_theta, 3)))
    final_x, final_y = waypoints[-1]
    path.append((final_x, final_y, path[-1][2]))
    return path


class HighLevelControl:
    def __init__(self, path, lookahead_dist=12.0):
        self.path = path
        self.lookahead_dist = lookahead_dist
        self.target_idx = 0

    def reset(self):
        self.target_idx = 0

    def get_control(self, robot_x_cm, robot_y_cm, robot_theta):
        if not self.path:
            return 0.0, 0.0

        while self.target_idx < len(self.path) - 1:
            target_x, target_y, _ = self.path[self.target_idx]
            dist_to_point = math.hypot(target_x - robot_x_cm, target_y - robot_y_cm)
            if dist_to_point > self.lookahead_dist:
                break
            self.target_idx += 1

        target_x, target_y, _ = self.path[self.target_idx]
        angle_to_target = math.atan2(target_y - robot_y_cm, target_x - robot_x_cm)
        heading_error = angle_to_target - robot_theta
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        kp_angular = 3.0
        angular_vel_cmd = kp_angular * heading_error

        final_x, final_y, _ = self.path[-1]
        dist_to_goal = math.hypot(final_x - robot_x_cm, final_y - robot_y_cm)

        if dist_to_goal < 4.0:
            linear_vel_cmd = 0.0
            angular_vel_cmd = 0.0
        else:
            linear_vel_cmd = max(0.1, 1.0 - (abs(heading_error) / (math.pi / 2)))

        return linear_vel_cmd, angular_vel_cmd


def main():
    env = RobotEnvironment(start_cell=(5,2), goal_cell=(5, 0))
    print("Environment running with autonomous HighLevelControl.")
    
    # 1. Generate line path
    astar_path = env.get_path()
    line_path = get_line_path(astar_path, steps_per_cell=40)
    
    # 2. Give the path to the environment to draw it correctly
    env.set_continuous_path(line_path)
    
    controller = HighLevelControl(line_path, lookahead_dist=14.0)

    running = True
    counter = 0
    while running:
        state = env.get_state()
        
        robot_x_cm = (state["x"] - (OFFSET + 0.5 * CELL_SIZE)) / PIXELS_PER_CM
        robot_y_cm = (state["y"] - (OFFSET + 0.5 * CELL_SIZE)) / PIXELS_PER_CM
        robot_theta = state["theta"]

        linear_vel, angular_vel = controller.get_control(robot_x_cm, robot_y_cm, robot_theta)

        target_linear = max(-1.0, min(1.0, linear_vel))
        target_angular = max(-1.0, min(1.0, angular_vel))
        
        action = (target_linear, target_angular)
        
        next_state, done = env.step(action)

        # Convert state and action to SI units (meters, radians, m/s, rad/s)
        x_m = robot_x_cm / 100.0
        y_m = robot_y_cm / 100.0
        linear_vel_mps = state["linear_vel"] / PIXELS_PER_CM / 100.0
        angular_vel_radps = state["angular_vel"]
        linear_action_mps = target_linear * LINEAR_SPEED / PIXELS_PER_CM / 100.0
        angular_action_radps = target_angular * ANGULAR_SPEED

        counter += 1
        if counter == 10:
            print(
                f"State: x={x_m:.3f} m, y={y_m:.3f} m, theta={robot_theta:.3f} rad, "
                f"v={linear_vel_mps:.3f} m/s, w={angular_vel_radps:.3f} rad/s | "
                f"Action: v_cmd={linear_action_mps:.3f} m/s, w_cmd={angular_action_radps:.3f} rad/s",
                flush=True,
            )
            counter = 0

        if done:
            print("Goal Reached! Resetting...")
            time.sleep(1)
            env.reset()
            controller.reset()

        env.render()

    env.close()

if __name__ == "__main__":
    main()