"""Planificador de trayectoria sobre celdas de laberinto.

El laberinto físico de este proyecto está armado sobre una rejilla regular
(ver controller/a_star.py, el prototipo con el maze escrito a mano como
matriz de paredes). En vez de correr A* píxel a píxel sobre el PGM entero
--decenas/cientos de miles de celdas en mapas de alta resolución, varios
segundos por goal en la Jetson-- esta clase reconstruye esa misma rejilla
automáticamente a partir del mapa:

  1. n_rows x n_cols según el tamaño físico de celda (nav.maze_cell_size_m)
     y el tamaño del mapa.
  2. Para cada par de celdas vecinas se muestrea una banda de píxeles
     alrededor del borde compartido (no una única línea exacta: las paredes
     del PGM suelen tener 1-2px de grosor y el borde calculado por
     redondeo puede caer a un par de píxeles de distancia): si la fracción
     ocupada máxima en esa banda supera wall_frac, la arista del grafo
     queda bloqueada.
  3. A* corre sobre ese grafo de decenas de nodos (4-conexo, igual que
     controller/a_star.py), no sobre la imagen: de segundos a microsegundos
     por goal.
  4. Los waypoints resultantes son centros de celda (naturalmente
     centrados en el pasillo, sin necesitar inflado ni sesgo de costo) más
     el goal exacto como último punto.

Los chequeos de seguridad en tiempo real (is_blocked/segment_clear) miran
directamente los píxeles crudos del PGM, sin ningún precálculo caro.
"""
# PY36: sin `from __future__ import annotations`; genéricos desde typing.
import heapq
import logging

from bisect import bisect_right
from typing import List, Optional, Tuple

from core.occupancy_map import OccupancyMap

log = logging.getLogger(__name__)


class MazePlanner:
    def __init__(self, occ, cell_size_m=0.3, occupied_below=220, wall_frac=0.2):
        # type: (OccupancyMap, float, int, float) -> None
        self._occ = occ
        self._occupied_below = occupied_below

        self._n_rows = max(1, int(round(occ.height * occ.resolution / cell_size_m)))
        self._n_cols = max(1, int(round(occ.width * occ.resolution / cell_size_m)))

        # Límites de píxel de cada fila/columna de celdas (n+1 cortes que
        # cubren toda la imagen, calculados desde 0 en cada paso para no
        # arrastrar redondeo acumulado).
        self._row_px = [int(round(r * occ.height / self._n_rows))
                         for r in range(self._n_rows + 1)]
        self._col_px = [int(round(c * occ.width / self._n_cols))
                         for c in range(self._n_cols + 1)]

        # Tolerancia de búsqueda alrededor del límite calculado: las
        # paredes del PGM son de 1-2px y el límite redondeado puede caer a
        # un par de píxeles de la línea real.
        self._margin = max(2, int(round(0.01 / occ.resolution)))

        # vwalls[r][c]: pared entre celda (r,c) y (r,c+1).
        # hwalls[r][c]: pared entre celda (r,c) y (r+1,c).
        self._vwalls = self._build_vwalls(wall_frac)
        self._hwalls = self._build_hwalls(wall_frac)

    # ------------------------------------------------------------
    # Construcción de la rejilla de paredes
    # ------------------------------------------------------------
    def _blocked_frac(self, row0, row1, col0, col1):
        # type: (int, int, int, int) -> float
        occ = self._occ
        w = occ.width
        total = 0
        blocked = 0
        for r in range(row0, row1):
            base = r * w
            for c in range(col0, col1):
                total += 1
                if occ.pixels[base + c] < self._occupied_below:
                    blocked += 1
        return (blocked / total) if total else 1.0

    def _build_vwalls(self, wall_frac):
        # type: (float) -> List[List[bool]]
        w = self._occ.width
        margin = self._margin
        vwalls = []
        for r in range(self._n_rows):
            row0, row1 = self._row_px[r], self._row_px[r + 1]
            walls = []
            for c in range(self._n_cols - 1):
                boundary = self._col_px[c + 1]
                col0 = max(0, boundary - margin)
                col1 = min(w, boundary + margin + 1)
                best = 0.0
                for col in range(col0, col1):
                    f = self._blocked_frac(row0, row1, col, col + 1)
                    if f > best:
                        best = f
                walls.append(best > wall_frac)
            vwalls.append(walls)
        return vwalls

    def _build_hwalls(self, wall_frac):
        # type: (float) -> List[List[bool]]
        h = self._occ.height
        margin = self._margin
        hwalls = []
        for r in range(self._n_rows - 1):
            boundary = self._row_px[r + 1]
            row0 = max(0, boundary - margin)
            row1 = min(h, boundary + margin + 1)
            walls = []
            for c in range(self._n_cols):
                col0, col1 = self._col_px[c], self._col_px[c + 1]
                best = 0.0
                for row in range(row0, row1):
                    f = self._blocked_frac(row, row + 1, col0, col1)
                    if f > best:
                        best = f
                walls.append(best > wall_frac)
            hwalls.append(walls)
        return hwalls

    # ------------------------------------------------------------
    # Mundo <-> celda
    # ------------------------------------------------------------
    def world_to_cell(self, x, y):
        # type: (float, float) -> Tuple[int, int]
        px, py = self._occ.world_to_pixel(x, y)
        col = self._locate(px, self._col_px, self._n_cols)
        row = self._locate(py, self._row_px, self._n_rows)
        return row, col

    @staticmethod
    def _locate(v, bounds, n):
        # type: (float, List[int], int) -> int
        idx = bisect_right(bounds, v) - 1
        return max(0, min(n - 1, idx))

    def cell_to_world(self, row, col):
        # type: (int, int) -> Tuple[float, float]
        px = (self._col_px[col] + self._col_px[col + 1]) / 2.0
        py = (self._row_px[row] + self._row_px[row + 1]) / 2.0
        return self._occ.pixel_to_world(px, py)

    # ------------------------------------------------------------
    # Seguridad en tiempo real: mira el PGM crudo, sin precálculo.
    # ------------------------------------------------------------
    def is_blocked(self, x, y):
        # type: (float, float) -> bool
        px, py = self._occ.world_to_pixel(x, y)
        col, row = int(px), int(py)
        if not (0 <= col < self._occ.width and 0 <= row < self._occ.height):
            return True
        return self._occ.pixel_at(col, row) < self._occupied_below

    def segment_clear(self, a_xy, b_xy):
        # type: (Tuple[float, float], Tuple[float, float]) -> bool
        ax, ay = self._occ.world_to_pixel(*a_xy)
        bx, by = self._occ.world_to_pixel(*b_xy)
        return self._line_free(int(ax), int(ay), int(bx), int(by))

    def _pixel_free(self, col, row):
        # type: (int, int) -> bool
        if not (0 <= col < self._occ.width and 0 <= row < self._occ.height):
            return False
        return self._occ.pixel_at(col, row) >= self._occupied_below

    def _line_free(self, c0, r0, c1, r1):
        # type: (int, int, int, int) -> bool
        """Bresenham: todos los píxeles tocados deben estar libres."""
        dc = abs(c1 - c0)
        dr = abs(r1 - r0)
        sc = 1 if c1 > c0 else -1
        sr = 1 if r1 > r0 else -1
        err = dc - dr
        c, r = c0, r0
        while True:
            if not self._pixel_free(c, r):
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

    # ------------------------------------------------------------
    # A* sobre el grafo de celdas
    # ------------------------------------------------------------
    def plan(self, start_xy, goal_xy):
        # type: (Tuple[float, float], Tuple[float, float]) -> Optional[List[Tuple[float, float]]]
        """Camino de (x,y) mundo en metros, o None si no hay ruta."""
        start = self.world_to_cell(*start_xy)
        goal = self.world_to_cell(*goal_xy)

        cells = self._astar(start, goal)
        if cells is None:
            log.warning("plan: sin ruta de celda %s a %s", start, goal)
            return None

        cells = self._compress_collinear(cells)
        path = [self.cell_to_world(r, c) for r, c in cells]
        # El último waypoint es el goal pedido (no el centro de celda) para
        # que la tolerancia de llegada se mida contra el clic real del host.
        path[-1] = (float(goal_xy[0]), float(goal_xy[1]))
        return path

    def _neighbors(self, cell):
        # type: (Tuple[int, int]) -> List[Tuple[int, int]]
        r, c = cell
        out = []
        if c + 1 < self._n_cols and not self._vwalls[r][c]:
            out.append((r, c + 1))
        if c - 1 >= 0 and not self._vwalls[r][c - 1]:
            out.append((r, c - 1))
        if r + 1 < self._n_rows and not self._hwalls[r][c]:
            out.append((r + 1, c))
        if r - 1 >= 0 and not self._hwalls[r - 1][c]:
            out.append((r - 1, c))
        return out

    def _astar(self, start, goal):
        # type: (Tuple[int, int], Tuple[int, int]) -> Optional[List[Tuple[int, int]]]
        def h(cell):
            return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

        open_heap = [(h(start), start)]
        came_from = {}
        g_score = {start: 0}
        closed = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            if current in closed:
                continue
            closed.add(current)
            for neighbor in self._neighbors(current):
                tentative = g_score[current] + 1
                if neighbor not in g_score or tentative < g_score[neighbor]:
                    g_score[neighbor] = tentative
                    came_from[neighbor] = current
                    heapq.heappush(open_heap, (tentative + h(neighbor), neighbor))
        return None

    @staticmethod
    def _compress_collinear(cells):
        # type: (List[Tuple[int, int]]) -> List[Tuple[int, int]]
        """Colapsa tramos rectos consecutivos a sus dos extremos (menos
        waypoints redundantes para pure pursuit; no cambia la ruta)."""
        if len(cells) <= 2:
            return cells
        out = [cells[0]]
        prev_dir = None
        for i in range(1, len(cells)):
            d = (cells[i][0] - cells[i - 1][0], cells[i][1] - cells[i - 1][1])
            if d == prev_dir:
                out[-1] = cells[i]
            else:
                out.append(cells[i])
                prev_dir = d
        return out
