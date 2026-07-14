"""Editor runtime de paredes del maze (secciones de 30 cm).

Consume los Ev.WALL_EDIT que net/nav_server.py recibe del host
({action: add|remove|reset, o, i, j}), y para cada edición válida:

  1. Actualiza el conjunto de paredes interiores (core/maze_walls.py).
  2. Re-renderiza los píxeles del mapa desde el PGM base + paredes.
  3. Reconstruye el AStarPlanner desde el mapa editado (en un executor para
     no bloquear el lazo de control a 20 Hz) y lo intercambia en el
     NavController, que replanifica solo si hay un goal activo.
  4. Persiste el conjunto en CFG.nav.walls_state_path (el PGM original nunca
     se toca en disco; borrar el JSON restaura el laberinto original).
  5. Publica el estado completo (Ev.WALLS_CHANGED -> broadcast WS) con la
     conectividad del grafo de celdas: agregar una pared que desconecta el
     laberinto se PERMITE, pero el host muestra la advertencia.

El perímetro es fijo: quitarlo dejaría al robot salir del área mapeada.
"""
# PY36: sin `from __future__ import annotations`; genéricos desde typing.
import asyncio
import json
import logging
import os

from typing import Callable, Optional, Set  # PY36

from config import CFG
from core import maze_walls as mw
from core.bus import Ev, bus
from core.occupancy_map import OccupancyMap
from core.state import state

log = logging.getLogger(__name__)


class WallEditor:
    def __init__(self, occ_base, grid, map_name, build_planner, loop):
        # type: (OccupancyMap, mw.MazeGrid, str, Callable[[OccupancyMap], object], asyncio.AbstractEventLoop) -> None
        self._occ_base = occ_base
        self._grid = grid
        self._map_name = map_name
        self._build_planner = build_planner
        self._loop = loop
        self._controller = None  # se fija con set_controller()
        # PY36: Lock captura el loop al construirse; lo pasamos explícito.
        # (En 3.10+ el parámetro `loop` no existe: ahí se enlaza solo.)
        try:
            self._lock = asyncio.Lock(loop=loop)
        except TypeError:
            self._lock = asyncio.Lock()

        # Paredes del PGM original (para "reset") y estado actual (JSON si hay).
        self._original = frozenset(mw.detect_walls(
            occ_base, grid, occupied_below=CFG.nav.occupied_below))
        self._walls = set(self._original)  # type: Set[mw.Segment]
        loaded = self._load_state()
        if loaded is not None:
            self._walls = loaded
            log.info("Paredes editadas cargadas de %s: %d interiores "
                     "(original: %d)", CFG.nav.walls_state_path,
                     len(self._walls), len(self._original))

    # ------------------------------------------------------------
    # API para run_controller
    # ------------------------------------------------------------
    def render_occ(self):
        # type: () -> OccupancyMap
        """Mapa base con el conjunto de paredes actual aplicado."""
        return self._render(self._walls)

    def set_controller(self, controller):
        # type: (object) -> None
        self._controller = controller

    def attach(self):
        # type: () -> None
        bus.on(Ev.WALL_EDIT, self._on_edit)
        self.publish_state(emit=False)

    def publish_state(self, emit=True):
        # type: (bool) -> None
        """Refresca state.walls_state (lo que nav_server manda al conectar) y
        opcionalmente lo difunde a los clientes ya conectados."""
        robot_xy = (state.pose_x, state.pose_y) if state.pose_valid else None
        connected, unreachable = mw.connectivity_info(
            self._grid, self._walls, robot_xy)
        st = {
            "map": self._map_name,
            "walls": mw.walls_to_list(self._walls),
            "connected": connected,
            "unreachable": unreachable,
        }
        state.walls_state = st
        if emit:
            bus.emit(Ev.WALLS_CHANGED, st)

    # ------------------------------------------------------------
    # Edición
    # ------------------------------------------------------------
    async def _on_edit(self, data):
        # type: (object) -> None
        if not isinstance(data, dict):
            return
        async with self._lock:
            action = data.get("action")
            if action == "reset":
                new_walls = set(self._original)
                if new_walls == self._walls:
                    # No-op: confirmar y re-sincronizar al host sin rebuild.
                    self._result(True, action)
                    self.publish_state()
                    return
            elif action in ("add", "remove"):
                seg = mw.parse_segment(data, self._grid)
                if seg is None:
                    self._result(False, action, "segmento inválido")
                    return
                if not mw.is_interior(self._grid, seg):
                    self._result(False, action, "la pared del perímetro es fija")
                    return
                new_walls = set(self._walls)
                if action == "add":
                    if seg in new_walls:
                        self._result(False, action, "ya hay una pared ahí")
                        return
                    new_walls.add(seg)
                else:
                    if seg not in new_walls:
                        self._result(False, action, "no hay pared ahí")
                        return
                    new_walls.discard(seg)
            else:
                self._result(False, str(action), "acción desconocida")
                return

            # Render + rebuild del planner fuera del loop (en la Jetson tarda
            # algunos cientos de ms; el lazo de control sigue corriendo con
            # el planner viejo mientras tanto).
            try:
                planner = await self._loop.run_in_executor(
                    None, self._render_and_build, new_walls)
            except Exception:
                log.exception("Error reconstruyendo el planner tras editar pared")
                self._result(False, action, "error interno reconstruyendo el mapa")
                return

            self._walls = new_walls
            if self._controller is not None:
                self._controller.set_planner(planner)
            self._save_state()
            self._result(True, action)
            self.publish_state()
            log.info("Paredes actualizadas (%s): %d interiores", action,
                     len(self._walls))

    def _render(self, walls):
        # type: (Set[mw.Segment]) -> OccupancyMap
        base = self._occ_base
        return OccupancyMap(
            width=base.width,
            height=base.height,
            resolution=base.resolution,
            origin_x=base.origin_x,
            origin_y=base.origin_y,
            pixels=mw.render_walls(base, self._grid, walls),
        )

    def _render_and_build(self, walls):
        # type: (Set[mw.Segment]) -> object
        return self._build_planner(self._render(walls))

    @staticmethod
    def _result(ok, action, reason=""):
        # type: (bool, str, str) -> None
        msg = {"ok": ok, "action": action}
        if reason:
            msg["reason"] = reason
            log.warning("Edición de pared rechazada (%s): %s", action, reason)
        bus.emit(Ev.WALL_RESULT, msg)

    # ------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------
    def _load_state(self):
        # type: () -> Optional[Set[mw.Segment]]
        path = CFG.nav.walls_state_path
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("map") != self._map_name:
            log.warning("%s ignorado: no corresponde al mapa '%s'",
                        path, self._map_name)
            return None
        return mw.parse_walls(data.get("walls"), self._grid)

    def _save_state(self):
        # type: () -> None
        path = CFG.nav.walls_state_path
        payload = {"map": self._map_name, "walls": mw.walls_to_list(self._walls)}
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=1)
            os.replace(tmp, path)
        except OSError:
            log.exception("No se pudo persistir el estado de paredes en %s", path)
