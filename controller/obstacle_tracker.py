"""Rastreador de celdas bloqueadas por obstáculos detectados por la DNN.

Consume los Ev.DETECTIONS de perception/detector.py (puntos ya proyectados
al frame del mapa) y mantiene el conjunto de celdas de 30 cm ocupadas por
objetos, con histéresis en ambos sentidos:

  * BLOQUEO: una celda se marca tras verse ocupada de forma continua
    mark_debounce_s (filtra falsos positivos de un frame).
  * LIBERACIÓN: una celda bloqueada se libera tras clear_debounce_s continuos
    SIN detecciones que caigan en ella, y sólo mientras está "a la vista"
    (dentro del FOV de la cámara, en rango y con línea de vista sin paredes).
    Si la cámara no la está mirando, la celda persiste — el robot en espera
    queda mirando al objeto, así que en la práctica la liberación ocurre al
    retirarlo.

Con cada cambio (y con cada edición de paredes, Ev.WALLS_CHANGED) se
reconstruye el planner COMPLETO (paredes + celdas de obstáculo estampadas
como ocupadas) en un executor —mismo patrón que controller/wall_editor.py—
y se intercambia con NavController.set_obstacle_planner(), que replanifica
al instante ("replan first"); si no queda ruta, el controlador entra en
espera hasta que la celda se libere. El estado se difunde al host
(Ev.OBSTACLES_CHANGED -> WS nav {"type":"obstacles"}) para pintar las celdas.

Las celdas de obstáculo NO se persisten: son objetos físicos transitorios.
"""
# PY36: sin `from __future__ import annotations`; genéricos desde typing.
import asyncio
import logging
import math
import time

from typing import Callable, Dict, Set, Tuple  # PY36

from config import CFG
from core import maze_walls as mw
from core.bus import Ev, bus
from core.occupancy_map import OccupancyMap
from core.state import state

log = logging.getLogger(__name__)

Cell = Tuple[int, int]


def _wrap(a):
    # type: (float) -> float
    return math.atan2(math.sin(a), math.cos(a))


class ObstacleTracker:
    def __init__(self, occ_base, grid, map_name, build_planner, loop,
                 walls_provider, walls_occ):
        # type: (OccupancyMap, mw.MazeGrid, str, Callable[[OccupancyMap], object], asyncio.AbstractEventLoop, Callable[[], Set[mw.Segment]], OccupancyMap) -> None
        self._occ_base = occ_base          # PGM base (sin render de paredes)
        self._grid = grid
        self._map_name = map_name
        self._build_planner = build_planner
        self._loop = loop
        # Snapshot del conjunto de paredes actual (WallEditor.walls_snapshot):
        # se toma en el loop y se renderiza en el executor sin carreras.
        self._walls_provider = walls_provider
        # Mapa con SOLO paredes renderizadas (sin obstáculos): línea de vista.
        self._walls_occ = walls_occ
        self._controller = None            # se fija con set_controller()

        self._blocked = set()              # type: Set[Cell]
        self._pending = {}                 # type: Dict[Cell, float]
        self._clear_since = {}             # type: Dict[Cell, float]
        self._dirty = False
        self._rebuilding = False

    # ------------------------------------------------------------
    # API para run_controller
    # ------------------------------------------------------------
    def set_controller(self, controller):
        # type: (object) -> None
        self._controller = controller

    def attach(self):
        # type: () -> None
        bus.on(Ev.DETECTIONS, self._on_detections)
        bus.on(Ev.WALLS_CHANGED, self._on_walls_changed)
        self.publish_state(emit=False)

    def publish_state(self, emit=True):
        # type: (bool) -> None
        st = {
            "map": self._map_name,
            "cells": [[i, j] for (i, j) in sorted(self._blocked)],
        }
        state.obstacles_state = st
        if emit:
            bus.emit(Ev.OBSTACLES_CHANGED, st)

    # ------------------------------------------------------------
    # Detecciones -> máquina de estados por celda
    # ------------------------------------------------------------
    def _on_detections(self, data):
        # type: (object) -> None
        if not isinstance(data, dict):
            return
        pose = data.get("pose")
        if not isinstance(pose, dict):
            # Sin pose no se puede ni mapear puntos ni acreditar ausencias.
            return
        p = CFG.perception
        grid = self._grid
        now = float(data.get("stamp") or time.time())
        px, py, pyaw = float(pose["x"]), float(pose["y"]), float(pose["yaw"])
        robot_cell = grid.cell_of(px, py)

        seen = set()  # type: Set[Cell]
        for pt in data.get("points") or []:
            try:
                ox, oy = float(pt["x"]), float(pt["y"])
            except (KeyError, TypeError, ValueError):
                continue
            # Fuera de la rejilla (cell_of haría clamp): ignorar.
            if not (grid.x0 <= ox <= grid.x0 + grid.cols * grid.cell_m):
                continue
            if not (grid.y0 <= oy <= grid.y0 + grid.rows * grid.cell_m):
                continue
            cell = grid.cell_of(ox, oy)
            # La celda que ocupa el propio robot nunca se bloquea (una
            # detección ahí es error de proyección o el objeto ya toca al
            # robot; bloquearla rompería la planificación desde adentro).
            if cell == robot_cell:
                continue
            seen.add(cell)

        changed = False

        # Bloqueo con debounce de confirmación (presencia continua).
        for cell in seen:
            if cell in self._blocked:
                self._clear_since.pop(cell, None)
                continue
            t0 = self._pending.get(cell)
            if t0 is None:
                self._pending[cell] = now
            elif now - t0 >= p.mark_debounce_s:
                del self._pending[cell]
                self._blocked.add(cell)
                changed = True
                log.info("Obstáculo detectado: celda (%d, %d) bloqueada",
                         cell[0], cell[1])
        # Candidatas que dejaron de verse antes de confirmar: descartar.
        for cell in list(self._pending):
            if cell not in seen:
                del self._pending[cell]

        # Liberación: ausencia continua mientras la celda está a la vista.
        for cell in list(self._blocked):
            if cell in seen:
                continue
            if self._cell_visible(px, py, pyaw, cell):
                t0 = self._clear_since.get(cell)
                if t0 is None:
                    self._clear_since[cell] = now
                elif now - t0 >= p.clear_debounce_s:
                    self._blocked.discard(cell)
                    self._clear_since.pop(cell, None)
                    changed = True
                    log.info("Obstáculo retirado: celda (%d, %d) liberada",
                             cell[0], cell[1])
            else:
                # No visible: la ausencia no acredita nada; reiniciar timer.
                self._clear_since.pop(cell, None)

        if changed:
            self._schedule_rebuild()

    def _cell_visible(self, px, py, pyaw, cell):
        # type: (float, float, float, Cell) -> bool
        """True si el centro de la celda está dentro del cono del FOV, en
        rango de detección y con línea de vista libre de paredes."""
        p = CFG.perception
        grid = self._grid
        cx = grid.x0 + (cell[0] + 0.5) * grid.cell_m
        cy = grid.y0 + (cell[1] + 0.5) * grid.cell_m
        dx, dy = cx - px, cy - py
        r = math.hypot(dx, dy)
        if r > p.max_range_m:
            return False
        half = math.radians(p.cam_hfov_deg) / 2.0 - math.radians(p.fov_margin_deg)
        if abs(_wrap(math.atan2(dy, dx) - pyaw)) > half:
            return False
        return self._los_free(px, py, cx, cy)

    def _los_free(self, ax, ay, bx, by):
        # type: (float, float, float, float) -> bool
        """Bresenham sobre los píxeles del mapa de paredes: True si el rayo
        no cruza ningún pixel ocupado (paredes de 1 px incluidas)."""
        occ = self._walls_occ
        x0f, y0f = occ.world_to_pixel(ax, ay)
        x1f, y1f = occ.world_to_pixel(bx, by)
        x0, y0 = int(x0f), int(y0f)
        x1, y1 = int(x1f), int(y1f)
        w, h = occ.width, occ.height
        px = occ.pixels
        thr = CFG.nav.occupied_below
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            if 0 <= x < w and 0 <= y < h and px[y * w + x] < thr:
                return False
            if x == x1 and y == y1:
                return True
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    # ------------------------------------------------------------
    # Rebuild del planner (paredes + obstáculos)
    # ------------------------------------------------------------
    def _on_walls_changed(self, _data):
        # type: (object) -> None
        # Cambió el conjunto de paredes: re-renderizar la base y re-estampar
        # los obstáculos (con el tracker activo, NavController delega en
        # nosotros el intercambio del planner completo; ver set_planner).
        self._schedule_rebuild()

    def _schedule_rebuild(self):
        # type: () -> None
        self._dirty = True
        if self._rebuilding:
            return
        self._rebuilding = True
        asyncio.ensure_future(self._rebuild_loop(), loop=self._loop)

    async def _rebuild_loop(self):
        # type: () -> None
        try:
            while self._dirty:
                self._dirty = False
                walls = set(self._walls_provider())
                cells = sorted(self._blocked)
                try:
                    walls_occ, planner = await self._loop.run_in_executor(
                        None, self._render_and_build, walls, cells)
                except Exception:
                    log.exception("Error reconstruyendo el planner con obstáculos")
                    return
                self._walls_occ = walls_occ
                if self._controller is not None:
                    self._controller.set_obstacle_planner(planner)
                self.publish_state()
                log.info("Planner con obstáculos actualizado: %d celda(s) "
                         "bloqueada(s)", len(cells))
        finally:
            self._rebuilding = False

    def _render_and_build(self, walls, cells):
        # type: (Set[mw.Segment], list) -> Tuple[OccupancyMap, object]
        base = self._occ_base
        wall_pixels = mw.render_walls(base, self._grid, walls)
        walls_occ = OccupancyMap(
            width=base.width, height=base.height, resolution=base.resolution,
            origin_x=base.origin_x, origin_y=base.origin_y, pixels=wall_pixels)

        buf = bytearray(wall_pixels)
        for cell in cells:
            self._stamp_cell(buf, walls_occ, cell)
        full_occ = OccupancyMap(
            width=base.width, height=base.height, resolution=base.resolution,
            origin_x=base.origin_x, origin_y=base.origin_y, pixels=bytes(buf))
        return walls_occ, self._build_planner(full_occ)

    def _stamp_cell(self, buf, occ, cell):
        # type: (bytearray, OccupancyMap, Cell) -> None
        """Rellena como ocupados los píxeles de la celda (i, j) de 30 cm."""
        grid = self._grid
        i, j = cell
        wx0 = grid.x0 + i * grid.cell_m
        wy0 = grid.y0 + j * grid.cell_m
        # Esquina superior-izquierda en píxeles = (x0, y0+cell) del mundo.
        px0, py0 = occ.world_to_pixel(wx0, wy0 + grid.cell_m)
        px1, py1 = occ.world_to_pixel(wx0 + grid.cell_m, wy0)
        c0 = max(0, int(math.floor(px0)))
        c1 = min(occ.width - 1, int(math.ceil(px1)))
        r0 = max(0, int(math.floor(py0)))
        r1 = min(occ.height - 1, int(math.ceil(py1)))
        if c1 < c0 or r1 < r0:
            return
        w = occ.width
        n = c1 - c0 + 1
        row_val = b"\x00" * n
        for r in range(r0, r1 + 1):
            buf[r * w + c0:r * w + c0 + n] = row_val
