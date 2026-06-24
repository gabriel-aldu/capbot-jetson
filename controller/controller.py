import asyncio
import logging
import math

from core.bus import Ev, bus
from controller.a_star import astar, get_path_with_directions, maze as MAZE
from protocol.udp_frame import SETPOINT_X_POS, SETPOINT_Y_POS

log = logging.getLogger(__name__)

# El host fija la meta por UDP (Ev.CMD_SETPOINT): x -> fila, y -> columna.
# Hasta que llegue el primer setpoint, usamos esta celda como meta inicial.
_DEFAULT_START_CELL = (5, 2)
_DEFAULT_GOAL_CELL = (0, 0)


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

        
        angular_pos_cmd = angle_to_target

        final_x, final_y, _ = self.path[-1]
        dist_to_goal = math.hypot(final_x - robot_x_cm, final_y - robot_y_cm)

        if dist_to_goal < 4.0:
            linear_vel_cmd = 0.0
            angular_pos_cmd = 0.0
        else:
            linear_vel_cmd = max(0.1, 1.0 - (abs(heading_error) / (math.pi / 2)))

        return linear_vel_cmd, angular_pos_cmd


async def run_controller(stop_event: asyncio.Event) -> None:
    """Sigue el path A* con odometria del ESP32 y emite Ev.CMD_VEL.

    Reemplaza al puente ROS2 (ros_bridge.py, removido): antes un nodo ROS
    corria HighLevelControl y publicaba el resultado a 'to_bridge'; ahora
    corre directo en el loop asyncio del servicio y el resultado va al bus,
    desde donde hw.esp32_link lo manda al ESP32 como frame VEL_CMD.
    """
    cells = astar(MAZE, _DEFAULT_START_CELL, _DEFAULT_GOAL_CELL)
    path = get_line_path(get_path_with_directions(cells), steps_per_cell=40)
    hlc = HighLevelControl(path, lookahead_dist=14.0)
    seq = 0

    def _on_telemetry(data) -> None:
        nonlocal seq
        if not isinstance(data, dict):
            return
        odo = data.get("odo")
        if not isinstance(odo, dict):
            return
        try:
            x_cm = float(odo.get("x", 0.0))
            y_cm = float(odo.get("y", 0.0))
            theta = float(odo.get("a", 0.0))
        except (TypeError, ValueError):
            return

        linear_vel_cmd, angular_pos_cmd = hlc.get_control(x_cm, y_cm, theta)
        seq += 1
        bus.emit(Ev.CMD_VEL, {"linear": linear_vel_cmd, "angular": angular_pos_cmd, "seq": seq})

    bus.on(Ev.TELEMETRY, _on_telemetry)
    try:
        await stop_event.wait()
    finally:
        bus.off(Ev.TELEMETRY, _on_telemetry)
