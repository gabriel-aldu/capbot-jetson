"""Controlador de navegación autónoma.

Reemplaza a nav2 + gui_bridge_node del stack ROS2:
  1. El host manda un goal (x, y, yaw en metros, frame del mapa) por el WS de
     navegación (net/nav_server.py) -> Ev.NAV_GOAL.
  2. Se planifica con A* sobre el mapa de ocupación (controller/a_star.py),
     el mismo .pgm/.yaml que muestra el host.
  3. Un lazo a CFG.nav.control_rate_hz sigue el camino con pure pursuit usando
     la pose que el ESP32 estima on-board (encoders+IMU) y que core/odometry.py
     reexpresa en el frame del mapa, emitiendo Ev.CMD_VEL {linear, angular} del
     chasis; hw/esp32_link.py hace la cinemática diferencial y manda VEL_CMD
     (rad/s por rueda) al ESP32.
  4. El progreso se reporta como Ev.NAV_STATUS (accepted/active/succeeded/...),
     que nav_server difunde al host con el mismo JSON que usaba ROS2.

Al aceptar un goal se manda MODE_CMD(1) al ESP32 (AUTONOMOUS_NAV) para que el
clic en el mapa baste por sí solo; el host puede volver a manual con su switch
de modo de siempre.
"""
# PY36: sin `from __future__ import annotations`; genéricos desde typing.
import asyncio
import logging
import math

from typing import List, Optional, Tuple

from config import CFG, AVAILABLE_MAPS
from core.bus import Ev, bus
from core.occupancy_map import load_map
from core.state import state
from controller.a_star import AStarPlanner

log = logging.getLogger(__name__)

# Fases de un goal activo
_PH_FOLLOW = "follow"   # siguiendo el camino
_PH_ALIGN = "align"     # en el punto: rotando al yaw final

# Con error de rumbo mayor a esto se rota en el lugar (sin avance).
_TURN_IN_PLACE_RAD = 1.2


def _wrap(a):
    # type: (float) -> float
    return math.atan2(math.sin(a), math.cos(a))


class NavController:
    def __init__(self, planner):
        # type: (AStarPlanner) -> None
        self._planner = planner
        self._path = []          # type: List[Tuple[float, float]]
        self._target_idx = 0
        self._goal = None        # type: Optional[Tuple[float, float, float]]
        self._phase = _PH_FOLLOW
        self._seq = 0
        self._active = False

    # ------------------------------------------------------------
    # Eventos
    # ------------------------------------------------------------
    def attach(self):
        # type: () -> None
        bus.on(Ev.NAV_GOAL, self._on_goal)
        bus.on(Ev.NAV_CANCEL, self._on_cancel)
        bus.on(Ev.STOP_MOTORS, self._on_stop)
        bus.on(Ev.ESP32_OFFLINE, self._on_esp32_offline)

    def _on_goal(self, data):
        # type: (dict) -> None
        try:
            gx = float(data["x"])
            gy = float(data["y"])
            gyaw = float(data.get("yaw", 0.0))
        except (KeyError, TypeError, ValueError):
            return

        if state.emergency_active:
            log.warning("Goal rechazado: emergencia activa")
            self._status("rejected")
            return
        if not state.pose_valid:
            log.warning("Goal rechazado: sin pose (ESP32 sin telemetría)")
            self._status("rejected")
            return

        path = self._planner.plan((state.pose_x, state.pose_y), (gx, gy))
        if not path:
            log.warning("Goal rechazado: sin ruta a x=%.2f y=%.2f", gx, gy)
            self._status("rejected")
            return

        self._path = path
        self._target_idx = 0
        self._goal = (gx, gy, gyaw)
        self._phase = _PH_FOLLOW
        self._active = True
        log.info("Goal aceptado: x=%.2f y=%.2f yaw=%.2f (%d waypoints)",
                 gx, gy, gyaw, len(path))
        self._status("accepted")
        # AUTONOMOUS_NAV en el ESP32: sin esto VEL_CMD se ignora y el robot
        # no se movería hasta que el usuario cambie el modo a mano en el host.
        bus.emit(Ev.CMD_MODE, {"mode": 1, "seq": 0})
        self._status("active")

    def _on_cancel(self, _data):
        # type: (object) -> None
        if not self._active:
            return
        self._finish("canceled")

    def _on_stop(self, _data):
        # type: (object) -> None
        # Emergencia u host offline: el freno ya salió por otro lado; sólo
        # abortamos el goal para no seguir empujando VEL_CMD.
        if self._active:
            self._finish("aborted", send_zero=False)

    def _on_esp32_offline(self, _reason):
        # type: (object) -> None
        if self._active:
            self._finish("aborted", send_zero=False)

    # ------------------------------------------------------------
    # Lazo de control
    # ------------------------------------------------------------
    async def run(self, stop_event):
        # type: (asyncio.Event) -> None
        period = 1.0 / CFG.nav.control_rate_hz
        status_div = max(1, int(CFG.nav.control_rate_hz / 2))  # status a ~2 Hz
        tick = 0
        while not stop_event.is_set():
            try:
                await asyncio.sleep(period)
            except asyncio.CancelledError:
                return
            if not self._active:
                continue
            if not state.pose_valid:
                continue

            tick += 1
            x, y, yaw = state.pose_x, state.pose_y, state.pose_yaw

            if self._planner.is_blocked(x, y):
                log.warning("Pose actual sobre celda bloqueada (pared/inflado); "
                            "abortando navegacion")
                self._finish("aborted")
                continue
            if self._phase == _PH_FOLLOW and not self._target_segment_clear(x, y):
                if not self._try_replan(x, y):
                    log.warning("Camino hacia el target cruza una pared y no hay "
                                "ruta alternativa; abortando navegacion")
                    self._finish("aborted")
                    continue

            v, w, remaining = self._step(x, y, yaw)
            if self._active:
                self._emit_vel(v, w)
                if tick % status_div == 0:
                    self._status("active", remaining)

    def _step(self, x, y, yaw):
        # type: (float, float, float) -> Tuple[float, float, float]
        """Un paso de control. Devuelve (v, w, distancia_restante)."""
        gx, gy, gyaw = self._goal
        dist_goal = math.hypot(gx - x, gy - y)

        if self._phase == _PH_FOLLOW:
            if dist_goal < CFG.nav.goal_tolerance_m:
                self._phase = _PH_ALIGN
                return 0.0, 0.0, dist_goal
            v, w = self._pure_pursuit(x, y, yaw)
            return v, w, dist_goal

        # _PH_ALIGN: rotar en el lugar hacia el yaw final del goal.
        yaw_err = _wrap(gyaw - yaw)
        if abs(yaw_err) < CFG.nav.yaw_tolerance_rad:
            self._finish("succeeded")
            return 0.0, 0.0, 0.0
        w = max(-CFG.robot.max_angular_speed,
                min(CFG.robot.max_angular_speed, CFG.nav.k_heading * yaw_err))
        return 0.0, w, dist_goal

    def _target_segment_clear(self, x, y):
        # type: (float, float) -> bool
        """Chequeo de seguridad: el tramo recto hacia el waypoint que persigue
        pure pursuit no debe cruzar una celda bloqueada. Detecta el caso en
        que la deriva de odometria movio al robot fuera del camino planeado
        y el siguiente tramo ahora corta una pared."""
        if self._target_idx >= len(self._path):
            return True
        tx, ty = self._path[self._target_idx]
        return self._planner.segment_clear((x, y), (tx, ty))

    def _try_replan(self, x, y):
        # type: (float, float) -> bool
        gx, gy, _ = self._goal
        path = self._planner.plan((x, y), (gx, gy))
        if not path:
            return False
        self._path = path
        self._target_idx = 0
        return True

    def _pure_pursuit(self, x, y, yaw):
        # type: (float, float, float) -> Tuple[float, float]
        # Avanzar el índice de target hasta el primer waypoint fuera del
        # radio de lookahead (el último nunca se salta).
        lookahead = CFG.nav.lookahead_m
        while self._target_idx < len(self._path) - 1:
            tx, ty = self._path[self._target_idx]
            if math.hypot(tx - x, ty - y) > lookahead:
                break
            self._target_idx += 1

        tx, ty = self._path[self._target_idx]
        heading_err = _wrap(math.atan2(ty - y, tx - x) - yaw)

        w = CFG.nav.k_heading * heading_err
        w = max(-CFG.robot.max_angular_speed, min(CFG.robot.max_angular_speed, w))

        if abs(heading_err) > _TURN_IN_PLACE_RAD:
            return 0.0, w  # muy desalineado: girar en el lugar primero

        # Reducir avance con el error de rumbo (perfil lineal simple).
        scale = max(0.2, 1.0 - abs(heading_err) / (math.pi / 2.0))
        v = min(CFG.nav.cruise_speed, CFG.robot.max_linear_speed) * scale
        return v, w

    # ------------------------------------------------------------
    def _finish(self, result, send_zero=True):
        # type: (str, bool) -> None
        self._active = False
        self._path = []
        self._goal = None
        if send_zero:
            self._emit_vel(0.0, 0.0)
        log.info("Navegación terminada: %s", result)
        self._status(result)

    def _emit_vel(self, v, w):
        # type: (float, float) -> None
        self._seq += 1
        bus.emit(Ev.CMD_VEL, {"linear": v, "angular": w, "seq": self._seq})

    @staticmethod
    def _status(st, distance_remaining=None):
        # type: (str, Optional[float]) -> None
        msg = {"state": st}
        if distance_remaining is not None:
            msg["distance_remaining"] = round(distance_remaining, 3)
        bus.emit(Ev.NAV_STATUS, msg)


async def run_controller(stop_event):
    # type: (asyncio.Event) -> None
    """Carga el mapa activo, arma el planificador y corre el lazo de control."""
    entry = AVAILABLE_MAPS.get(CFG.nav.map_name)
    if entry is None:
        log.error("Mapa '%s' no existe en AVAILABLE_MAPS; navegación deshabilitada",
                  CFG.nav.map_name)
        await stop_event.wait()
        return
    try:
        occ = load_map(entry[0], entry[1])
    except (OSError, ValueError) as exc:
        log.error("No se pudo cargar el mapa '%s': %s; navegación deshabilitada",
                  CFG.nav.map_name, exc)
        await stop_event.wait()
        return

    planner = AStarPlanner(
        occ,
        occupied_below=CFG.nav.occupied_below,
        inflation_radius_m=CFG.nav.inflation_radius_m,
        center_bias_radius_m=CFG.nav.center_bias_radius_m,
        center_bias_weight=CFG.nav.center_bias_weight,
        planning_resolution_m=CFG.nav.planning_resolution_m,
    )
    log.info("Planner listo: mapa '%s' %dx%d @ %.3f m/px -> rejilla %dx%d @ %.3f m/celda",
             CFG.nav.map_name, occ.width, occ.height, occ.resolution,
             planner.grid_width, planner.grid_height, planner.grid_resolution)

    controller = NavController(planner)
    controller.attach()
    await controller.run(stop_event)
