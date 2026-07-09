"""Planificador global sobre el mapa de ocupación (reemplaza al planner de nav2).

Pipeline:
  1. El PGM se binariza: celda bloqueada si pixel < CFG.nav.occupied_below
     (cubre ocupado=0 y desconocido=205; libre=254/255).
  2. Los obstáculos se inflan por el radio del robot (nav.inflation_radius_m)
     para poder tratarlo como un punto.
  3. A* 8-conexo (sin cortar esquinas) sobre la rejilla inflada.
  4. El camino de celdas se acorta por línea-de-vista (greedy) y se devuelve
     como waypoints (x, y) en metros, frame del mapa.

Todo en stdlib (sin numpy) porque los mapas usados son chicos
(decenas de miles de celdas).
"""
# PY36: sin `from __future__ import annotations`; genéricos desde typing.
import heapq
import logging
import math

from typing import List, Optional, Tuple

from core.occupancy_map import OccupancyMap

log = logging.getLogger(__name__)

_SQRT2 = math.sqrt(2.0)


class GridPlanner:
    def __init__(self, occ, occupied_below=220, inflation_radius_m=0.10):
        # type: (OccupancyMap, int, float) -> None
        self._occ = occ
        self._w = occ.width
        self._h = occ.height
        # blocked[row*w + col] = True si la celda (inflada) no es transitable.
        self._blocked = self._build_blocked(occupied_below, inflation_radius_m)

    # ------------------------------------------------------------
    # Construcción de la rejilla
    # ------------------------------------------------------------
    def _build_blocked(self, occupied_below, inflation_radius_m):
        # type: (int, float) -> List[bool]
        w, h, occ = self._w, self._h, self._occ
        raw = [occ.pixels[i] < occupied_below for i in range(w * h)]

        r_cells = int(math.ceil(inflation_radius_m / occ.resolution))
        if r_cells <= 0:
            return raw

        # Disco de offsets para el inflado (euclídeo).
        offsets = []
        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                if dr * dr + dc * dc <= r_cells * r_cells:
                    offsets.append((dr, dc))

        blocked = raw[:]
        for row in range(h):
            base = row * w
            for col in range(w):
                if not raw[base + col]:
                    continue
                for dr, dc in offsets:
                    rr, cc = row + dr, col + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        blocked[rr * w + cc] = True
        return blocked

    def _is_free(self, col, row):
        # type: (int, int) -> bool
        if not (0 <= col < self._w and 0 <= row < self._h):
            return False
        return not self._blocked[row * self._w + col]

    # ------------------------------------------------------------
    # Mundo <-> celda
    # ------------------------------------------------------------
    def world_to_cell(self, x, y):
        # type: (float, float) -> Tuple[int, int]
        px, py = self._occ.world_to_pixel(x, y)
        return int(px), int(py)

    def cell_to_world(self, col, row):
        # type: (int, int) -> Tuple[float, float]
        # Centro de la celda.
        return self._occ.pixel_to_world(col + 0.5, row + 0.5)

    def _nearest_free(self, col, row, max_radius=8):
        # type: (int, int, int) -> Optional[Tuple[int, int]]
        """Celda libre más cercana (para poses/goals que caen en el inflado)."""
        if self._is_free(col, row):
            return (col, row)
        best = None
        best_d2 = None
        for r in range(1, max_radius + 1):
            for dr in range(-r, r + 1):
                for dc in range(-r, r + 1):
                    if max(abs(dr), abs(dc)) != r:
                        continue  # sólo el anillo de radio r
                    cc, rr = col + dc, row + dr
                    if self._is_free(cc, rr):
                        d2 = dr * dr + dc * dc
                        if best_d2 is None or d2 < best_d2:
                            best, best_d2 = (cc, rr), d2
            if best is not None:
                return best
        return None

    # ------------------------------------------------------------
    # A*
    # ------------------------------------------------------------
    def plan(self, start_xy, goal_xy):
        # type: (Tuple[float, float], Tuple[float, float]) -> Optional[List[Tuple[float, float]]]
        """Camino de (x,y) mundo en metros, o None si no hay ruta."""
        start = self._nearest_free(*self.world_to_cell(*start_xy))
        goal = self._nearest_free(*self.world_to_cell(*goal_xy))
        if start is None or goal is None:
            log.warning("plan: start u goal fuera del área transitable")
            return None

        cells = self._astar(start, goal)
        if cells is None:
            return None
        cells = self._shortcut(cells)
        path = [self.cell_to_world(c, r) for c, r in cells]
        # El último waypoint es el goal pedido (no el centro de celda) para
        # que la tolerancia de llegada se mida contra el clic real del host.
        path[-1] = (float(goal_xy[0]), float(goal_xy[1]))
        return path

    def _astar(self, start, goal):
        # type: (Tuple[int, int], Tuple[int, int]) -> Optional[List[Tuple[int, int]]]
        def h(cell):
            # Distancia octile (admisible en rejilla 8-conexa).
            dx = abs(cell[0] - goal[0])
            dy = abs(cell[1] - goal[1])
            return (dx + dy) + (_SQRT2 - 2.0) * min(dx, dy)

        open_heap = [(h(start), 0.0, start)]
        came_from = {}
        g_score = {start: 0.0}
        closed = set()

        while open_heap:
            _, g, current = heapq.heappop(open_heap)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            if current in closed:
                continue
            closed.add(current)

            col, row = current
            for dc in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if dc == 0 and dr == 0:
                        continue
                    nc, nr = col + dc, row + dr
                    if not self._is_free(nc, nr):
                        continue
                    if dc != 0 and dr != 0:
                        # No cortar esquinas: ambos ortogonales deben estar libres.
                        if not (self._is_free(col + dc, row) and self._is_free(col, row + dr)):
                            continue
                        step = _SQRT2
                    else:
                        step = 1.0
                    neighbor = (nc, nr)
                    tentative = g + step
                    if neighbor not in g_score or tentative < g_score[neighbor]:
                        g_score[neighbor] = tentative
                        came_from[neighbor] = current
                        heapq.heappush(open_heap, (tentative + h(neighbor), tentative, neighbor))
        return None

    # ------------------------------------------------------------
    # Suavizado por línea de vista
    # ------------------------------------------------------------
    def _shortcut(self, cells):
        # type: (List[Tuple[int, int]]) -> List[Tuple[int, int]]
        if len(cells) <= 2:
            return cells
        out = [cells[0]]
        i = 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1 and not self._line_free(cells[i], cells[j]):
                j -= 1
            out.append(cells[j])
            i = j
        return out

    def _line_free(self, a, b):
        # type: (Tuple[int, int], Tuple[int, int]) -> bool
        """Bresenham supercover: todas las celdas tocadas deben estar libres."""
        c0, r0 = a
        c1, r1 = b
        dc = abs(c1 - c0)
        dr = abs(r1 - r0)
        sc = 1 if c1 > c0 else -1
        sr = 1 if r1 > r0 else -1
        err = dc - dr
        c, r = c0, r0
        while True:
            if not self._is_free(c, r):
                return False
            if c == c1 and r == r1:
                return True
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c += sc
            if e2 < dc:
                err += dc
                r += sr
            # Evitar pasar en diagonal entre dos celdas bloqueadas.
            if e2 > -dr and e2 < dc:
                if not (self._is_free(c - sc, r) or self._is_free(c, r - sr)):
                    return False
