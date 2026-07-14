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

Obstáculos detectados por la DNN (controller/obstacle_tracker.py): el
controlador mantiene DOS planners:
  * `_planner` (completo: paredes + celdas de obstáculo) para planificar y
    para el chequeo de segmento del lazo — así una celda bloqueada dispara
    replanificación ("replan first") y detiene al robot antes de entrar.
  * `_base_planner` (sólo paredes) para el chequeo de aborto por holgura
    (una botella cerca no debe abortar como si fuera una pared) y para
    decidir si un bloqueo es POR OBSTÁCULO: si el planner completo no
    encuentra ruta pero el de paredes sí, el robot entra en fase de ESPERA
    (nav_status "waiting", velocidad 0) en vez de abortar, y reanuda solo
    cuando el tracker libera la celda (intercambio de planner) — salvo que
    el host cancele el goal.
"""
# PY36: sin `from __future__ import annotations`; genéricos desde typing.
import asyncio
import logging
import math

from typing import List, Optional, Tuple

from config import CFG, AVAILABLE_MAPS
from core import maze_walls
from core.bus import Ev, bus
from core.occupancy_map import load_map
from core.state import state
from controller.a_star import AStarPlanner
from controller.obstacle_tracker import ObstacleTracker
from controller.wall_editor import WallEditor

log = logging.getLogger(__name__)

# Fases de un goal activo
_PH_FOLLOW = "follow"   # siguiendo el camino
_PH_ALIGN = "align"     # en el punto: rotando al yaw final
_PH_WAIT = "wait"       # detenido: obstáculo bloquea toda ruta al goal

# Con error de rumbo mayor a esto se rota en el lugar (sin avance).
_TURN_IN_PLACE_RAD = 0.7

# En fase de espera, reintentar la planificación cada tanto (además del
# reintento inmediato con cada intercambio de planner del tracker).
_WAIT_REPLAN_PERIOD_S = 2.0


def _wrap(a):
    # type: (float) -> float
    return math.atan2(math.sin(a), math.cos(a))


class NavController:
    def __init__(self, planner, has_tracker=False):
        # type: (AStarPlanner, bool) -> None
        self._planner = planner        # completo (paredes + obstáculos DNN)
        self._base_planner = planner   # sólo paredes (seguridad / "¿es obstáculo?")
        # Con tracker, el planner completo lo entrega set_obstacle_planner();
        # set_planner() (WallEditor) sólo actualiza el de paredes.
        self._has_tracker = has_tracker
        self._path = []          # type: List[Tuple[float, float]]
        self._corners = set()    # índices de waypoints de pivote (giro brusco)
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
            # Sin ruta en el planner completo: si con SOLO paredes sí hay
            # ruta, el bloqueo es por un obstáculo detectado -> se acepta el
            # goal, se avanza por la ruta de paredes hasta donde se pueda y
            # el chequeo de segmento del lazo detiene al robot a esperar.
            if self._base_planner is not self._planner:
                path = self._base_planner.plan(
                    (state.pose_x, state.pose_y), (gx, gy))
            if not path:
                log.warning("Goal rechazado: sin ruta a x=%.2f y=%.2f", gx, gy)
                self._status("rejected")
                return
            log.info("Ruta al goal bloqueada por obstáculo: el robot se "
                     "acercará y esperará a que se libere")

        self._set_path(path)
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

    def set_planner(self, planner):
        # type: (AStarPlanner) -> None
        """Intercambia el planner de PAREDES (el mapa cambió por edición de
        paredes, ver controller/wall_editor.py). Sin tracker de obstáculos es
        también el planner completo y se replanifica de inmediato: una pared
        quitada puede abrir una ruta más corta, y una pared nueva sobre el
        camino actual se esquiva sin esperar a que el chequeo de seguridad
        del lazo la detecte. Si ya no hay ruta se conserva el camino viejo y
        el chequeo del lazo aborta al acercarse. CON tracker, el planner
        completo (paredes + obstáculos) llega enseguida por
        set_obstacle_planner() — el tracker escucha Ev.WALLS_CHANGED — y la
        replanificación ocurre ahí."""
        self._base_planner = planner
        if self._has_tracker:
            return
        self._planner = planner
        self._replan_after_swap("cambio de paredes")

    def set_obstacle_planner(self, planner):
        # type: (AStarPlanner) -> None
        """Intercambia el planner COMPLETO (paredes + celdas de obstáculo),
        entregado por controller/obstacle_tracker.py con cada cambio. Si hay
        goal activo se replanifica al instante: un obstáculo nuevo sobre el
        camino se esquiva si hay alternativa ("replan first") y una celda
        liberada saca al robot de la fase de espera."""
        self._planner = planner
        self._replan_after_swap("cambio de obstáculos")

    def _replan_after_swap(self, reason):
        # type: (str) -> None
        if not (self._active and self._goal is not None and state.pose_valid):
            return
        if self._try_replan(state.pose_x, state.pose_y):
            log.info("Replanificado por %s (%d waypoints)",
                     reason, len(self._path))
        elif self._phase != _PH_WAIT:
            log.warning("Sin ruta al goal tras %s; se mantiene el camino "
                        "anterior bajo vigilancia", reason)

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
        replan_div = max(1, int(CFG.nav.control_rate_hz * _WAIT_REPLAN_PERIOD_S))
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

            # Holgura contra la pared REAL (planner de paredes: una celda de
            # obstáculo cerca no debe abortar — para eso está la espera).
            if self._base_planner.clearance(x, y) < CFG.nav.abort_clearance_m:
                log.warning("Pose actual a menos de %.2f m de una pared "
                            "(o fuera del mapa); abortando navegacion",
                            CFG.nav.abort_clearance_m)
                self._finish("aborted")
                continue

            if self._phase == _PH_WAIT:
                # Detenido esperando que el obstáculo se retire. La reanudación
                # normal llega con el intercambio de planner del tracker
                # (set_obstacle_planner -> _try_replan); este reintento
                # periódico es sólo un respaldo barato.
                if tick % replan_div == 0 and self._try_replan(x, y):
                    continue  # _try_replan ya volvió a FOLLOW y avisó
                if tick % status_div == 0:
                    self._emit_vel(0.0, 0.0)
                    self._status("waiting")
                continue

            if self._phase == _PH_FOLLOW and not self._target_segment_clear(x, y):
                if not self._try_replan(x, y):
                    if self._begin_wait(x, y):
                        continue
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
        # Contra la pared REAL (no el inflado): con la banda libre de ~6 cm
        # que deja el inflado, chequear contra el inflado dispara replans en
        # cadena ante cualquier desvío chico de la pose.
        return self._planner.segment_clear((x, y), (tx, ty),
                                           min_clearance_m=CFG.nav.segment_clearance_m)

    def _try_replan(self, x, y):
        # type: (float, float) -> bool
        gx, gy, _ = self._goal
        path = self._planner.plan((x, y), (gx, gy))
        if not path:
            return False
        self._set_path(path)
        if self._phase == _PH_WAIT:
            # El obstáculo se liberó (o una edición de paredes abrió ruta):
            # reanudar el seguimiento donde quedó.
            self._phase = _PH_FOLLOW
            log.info("Ruta disponible de nuevo; reanudando navegación")
            self._status("active")
        return True

    def _begin_wait(self, x, y):
        # type: (float, float) -> bool
        """Entra en fase de espera si el bloqueo actual es POR OBSTÁCULO:
        sin ruta en el planner completo pero CON ruta en el de sólo paredes
        (si tampoco la hay con sólo paredes, el goal es inalcanzable de
        verdad y el llamador aborta). El robot queda detenido con el goal
        activo; un cancel del host lo aborta como siempre."""
        if self._base_planner is self._planner:
            return False  # sin tracker no hay celdas de obstáculo
        gx, gy, _ = self._goal
        if self._base_planner.plan((x, y), (gx, gy)) is None:
            return False
        self._phase = _PH_WAIT
        self._emit_vel(0.0, 0.0)
        log.info("Obstáculo bloquea toda ruta al goal; esperando a que se "
                 "retire (cancel del host para abortar)")
        self._status("waiting")
        return True

    def _set_path(self, path):
        # type: (List[Tuple[float, float]]) -> None
        """Fija el camino y marca los waypoints de pivote: donde el camino
        gira más que pivot_turn_min_rad (esquinas que el planner colocó en
        el bolsillo de la intersección). En esos waypoints pure pursuit usa
        corner_capture_m en vez del lookahead: el robot llega hasta el punto
        y recién ahí el error de rumbo salta ~90° y el umbral de giro en el
        lugar (_TURN_IN_PLACE_RAD) lo hace pivotear sin avance, en vez de
        recortar la esquina en arco (el barrido diagonal de 15.6 cm del
        chasis no cabe en el pasillo de 30 cm)."""
        self._path = path
        self._target_idx = 0
        corners = set()
        thr = CFG.nav.pivot_turn_min_rad
        for i in range(1, len(path) - 1):
            v0x = path[i][0] - path[i - 1][0]
            v0y = path[i][1] - path[i - 1][1]
            v1x = path[i + 1][0] - path[i][0]
            v1y = path[i + 1][1] - path[i][1]
            turn = abs(math.atan2(v0x * v1y - v0y * v1x, v0x * v1x + v0y * v1y))
            if turn > thr:
                corners.add(i)
        self._corners = corners

    def _pure_pursuit(self, x, y, yaw):
        # type: (float, float, float) -> Tuple[float, float]
        # Avanzar el índice de target hasta el primer waypoint fuera del
        # radio de captura (el último nunca se salta). Los waypoints de
        # pivote usan corner_capture_m: hay que LLEGAR a la esquina antes
        # de doblar, no recortarla desde un lookahead antes.
        lookahead = CFG.nav.lookahead_m
        capture = CFG.nav.corner_capture_m
        while self._target_idx < len(self._path) - 1:
            tx, ty = self._path[self._target_idx]
            reach = capture if self._target_idx in self._corners else lookahead
            if math.hypot(tx - x, ty - y) > reach:
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
        self._corners = set()
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


def _build_planner(occ):
    # type: (object) -> AStarPlanner
    return AStarPlanner(
        occ,
        occupied_below=CFG.nav.occupied_below,
        inflation_radius_m=CFG.nav.inflation_radius_m,
        center_bias_radius_m=CFG.nav.center_bias_radius_m,
        center_bias_weight=CFG.nav.center_bias_weight,
        planning_resolution_m=CFG.nav.planning_resolution_m,
    )


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

    # Mapas con rejilla de paredes editables (hoy sólo "maze"): el planner se
    # construye desde el render paredes->píxeles, no desde el PGM directo, y
    # el WallEditor lo reconstruye/intercambia con cada edición del host.
    editor = None
    occ_base = occ  # PGM crudo (el ObstacleTracker re-renderiza desde aquí)
    grid = maze_walls.grid_for_map(CFG.nav.map_name)
    if grid is not None:
        # PY36: get_event_loop() dentro de una corrutina devuelve el loop
        # activo (main.py hizo set_event_loop antes de arrancar).
        editor = WallEditor(occ, grid, CFG.nav.map_name, _build_planner,
                            asyncio.get_event_loop())
        occ = editor.render_occ()

    planner = _build_planner(occ)
    log.info("Planner listo: mapa '%s' %dx%d @ %.3f m/px -> rejilla %dx%d @ %.3f m/celda%s",
             CFG.nav.map_name, occ.width, occ.height, occ.resolution,
             planner.grid_width, planner.grid_height, planner.grid_resolution,
             " (paredes editables)" if editor else "")

    # Celdas bloqueadas por la DNN: sólo mapas con rejilla (hoy "maze") y con
    # la percepción habilitada. El tracker estampa las celdas sobre el render
    # de paredes y entrega el planner completo vía set_obstacle_planner().
    tracker = None
    if editor is not None and CFG.perception.enabled:
        tracker = ObstacleTracker(occ_base, grid, CFG.nav.map_name,
                                  _build_planner, asyncio.get_event_loop(),
                                  editor.walls_snapshot, occ)
        log.info("Tracker de obstáculos DNN activo (celdas de %.2f m)",
                 grid.cell_m)

    controller = NavController(planner, has_tracker=tracker is not None)
    controller.attach()
    if editor is not None:
        editor.set_controller(controller)
        editor.attach()
    if tracker is not None:
        tracker.set_controller(controller)
        tracker.attach()
    await controller.run(stop_event)
