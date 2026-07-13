"""Planificador global A* eficiente sobre el mapa de ocupación.

Reemplazo de controller/planner.py (GridPlanner). Mismo contrato público
(plan / is_blocked / segment_clear / world_to_cell / cell_to_world) para que
controller.py, el host y el ESP32 no necesiten cambios, pero con otro
pipeline interno:

  1. El PGM se SUBMUESTREA a una rejilla de planificación de
     nav.planning_resolution_m por celda (min-pooling: si cualquier pixel
     del bloque está ocupado/desconocido, la celda queda ocupada —
     conservador, no borra paredes de 1 px). Con el maze @ 0.003 m/px y
     0.015 m/celda la rejilla baja de 501x601 a ~101x121 (25x menos celdas).
  2. El inflado NO se hace estampando un disco por cada celda ocupada
     (O(ocupadas × radio²), el cuello de botella de GridPlanner); se calcula
     UNA transformada de distancia chamfer (dos pasadas, O(w×h)) y de ese
     campo salen las dos cosas a la vez:
       - blocked: dist <= inflation_radius_m
       - gradiente suave de inflado: costo extra que decae cuadráticamente
         de center_bias_weight a 0 entre el borde del inflado y
         center_bias_radius_m más allá, empujando a A* al centro del pasillo.
  3. A* 8-conexo (sin cortar esquinas) con el paso ponderado por el
     gradiente, y atajo por línea de vista (igual que GridPlanner).

Todo en stdlib (sin numpy), PY36-compatible, igual que el resto del repo.
"""
# PY36: sin `from __future__ import annotations`; genéricos desde typing.
import heapq
import logging
import math

from typing import List, Optional, Tuple

from core.occupancy_map import OccupancyMap

log = logging.getLogger(__name__)

_SQRT2 = math.sqrt(2.0)
_INF = float("inf")

# Detección de esquinas para pivote (ver _pivot_corners):
_PIVOT_WINDOW = 3          # celdas a cada lado para medir el giro local
_PIVOT_REGION_RAD = 0.52   # ~30°: la celda pertenece a una zona de giro
_PIVOT_NET_RAD = 0.79      # ~45°: giro neto que amerita pivotear en el lugar


class AStarPlanner:
    def __init__(self, occ, occupied_below=220, inflation_radius_m=0.10,
                 center_bias_radius_m=0.0, center_bias_weight=0.0,
                 planning_resolution_m=0.0):
        # type: (OccupancyMap, int, float, float, float, float) -> None
        self._occ = occ
        # Factor de submuestreo (>=1). planning_resolution_m <= resolución
        # nativa => sin submuestreo (p.ej. test_map_small @ 0.025 m/px).
        f = 1
        if planning_resolution_m > 0:
            f = max(1, int(round(planning_resolution_m / occ.resolution)))
        self._f = f
        self.grid_resolution = occ.resolution * f
        self.grid_width = (occ.width + f - 1) // f
        self.grid_height = (occ.height + f - 1) // f
        self._w = self.grid_width
        self._h = self.grid_height

        raw = self._downsample_blocked(occupied_below)
        # dist[i] = distancia (m, aprox chamfer) al pixel ocupado más cercano.
        self._dist = self._distance_transform(raw)
        # blocked[row*w + col] = True si la celda (inflada) no es transitable.
        # cost[row*w + col] = costo extra (>=0) por pasar cerca del inflado.
        self._blocked, self._cost = self._masks_from_distance(
            inflation_radius_m, center_bias_radius_m, center_bias_weight)

    # ------------------------------------------------------------
    # Construcción de la rejilla
    # ------------------------------------------------------------
    def _downsample_blocked(self, occupied_below):
        # type: (int) -> List[bool]
        """Rejilla cruda submuestreada: celda ocupada si CUALQUIER pixel de
        su bloque f×f es < occupied_below (ocupado=0 o desconocido=205)."""
        w, h, f = self._w, self._h, self._f
        occ = self._occ
        src_w = occ.width
        px = occ.pixels
        blocked = [False] * (w * h)
        for row in range(occ.height):
            grow = (row // f) * w
            base = row * src_w
            for col, v in enumerate(px[base:base + src_w]):
                if v < occupied_below:
                    blocked[grow + col // f] = True
        return blocked

    def _distance_transform(self, raw):
        # type: (List[bool]) -> List[float]
        """Transformada de distancia chamfer (pesos 1/√2, dos pasadas).

        Aproxima la distancia euclídea de cada celda a la celda ocupada más
        cercana en O(w×h), en vez del disco de offsets O(ocupadas × radio²)
        de GridPlanner. Devuelve metros."""
        w, h = self._w, self._h
        dist = [0.0 if b else _INF for b in raw]

        # Pasada hacia adelante (arriba-izquierda -> abajo-derecha).
        for row in range(h):
            base = row * w
            up = base - w
            for col in range(w):
                i = base + col
                d = dist[i]
                if d == 0.0:
                    continue
                if col > 0:
                    v = dist[i - 1] + 1.0
                    if v < d:
                        d = v
                if row > 0:
                    v = dist[up + col] + 1.0
                    if v < d:
                        d = v
                    if col > 0:
                        v = dist[up + col - 1] + _SQRT2
                        if v < d:
                            d = v
                    if col < w - 1:
                        v = dist[up + col + 1] + _SQRT2
                        if v < d:
                            d = v
                dist[i] = d

        # Pasada hacia atrás (abajo-derecha -> arriba-izquierda).
        for row in range(h - 1, -1, -1):
            base = row * w
            down = base + w
            for col in range(w - 1, -1, -1):
                i = base + col
                d = dist[i]
                if d == 0.0:
                    continue
                if col < w - 1:
                    v = dist[i + 1] + 1.0
                    if v < d:
                        d = v
                if row < h - 1:
                    v = dist[down + col] + 1.0
                    if v < d:
                        d = v
                    if col < w - 1:
                        v = dist[down + col + 1] + _SQRT2
                        if v < d:
                            d = v
                    if col > 0:
                        v = dist[down + col - 1] + _SQRT2
                        if v < d:
                            d = v
                dist[i] = d

        res = self.grid_resolution
        return [d * res if d < _INF else _INF for d in dist]

    def _masks_from_distance(self, inflation_radius_m, bias_radius_m, weight):
        # type: (float, float, float) -> Tuple[List[bool], List[float]]
        """Del campo de distancia salen el inflado y el gradiente suave:

          dist <= inflado                     -> bloqueada
          inflado < dist < inflado + bias     -> costo = weight * frac²
          dist >= inflado + bias              -> costo 0

        frac decae linealmente de 1 (borde del inflado) a 0 (fin del bias),
        así el costo es un gradiente cuadrático suave como el center_bias
        de GridPlanner, pero derivado del mismo campo (gratis)."""
        n = self._w * self._h
        blocked = [False] * n
        cost = [0.0] * n
        use_bias = bias_radius_m > 0.0 and weight > 0.0
        outer = inflation_radius_m + bias_radius_m
        dist = self._dist
        for i in range(n):
            d = dist[i]
            if d <= inflation_radius_m:
                blocked[i] = True
            elif use_bias and d < outer:
                frac = (outer - d) / bias_radius_m
                cost[i] = weight * frac * frac
        return blocked, cost

    def _is_free(self, col, row):
        # type: (int, int) -> bool
        if not (0 <= col < self._w and 0 <= row < self._h):
            return False
        return not self._blocked[row * self._w + col]

    # ------------------------------------------------------------
    # Mundo <-> celda (de planificación)
    # ------------------------------------------------------------
    def world_to_cell(self, x, y):
        # type: (float, float) -> Tuple[int, int]
        px, py = self._occ.world_to_pixel(x, y)
        return int(math.floor(px / self._f)), int(math.floor(py / self._f))

    def cell_to_world(self, col, row):
        # type: (int, int) -> Tuple[float, float]
        # Centro de la celda de planificación.
        f = self._f
        return self._occ.pixel_to_world((col + 0.5) * f, (row + 0.5) * f)

    def is_blocked(self, x, y):
        # type: (float, float) -> bool
        """True si (x,y) mundo cae en pared/inflado o fuera del mapa."""
        col, row = self.world_to_cell(x, y)
        return not self._is_free(col, row)

    def clearance(self, x, y):
        # type: (float, float) -> float
        """Distancia (m) desde (x,y) mundo a la pared REAL más cercana
        (no al inflado); -1 si cae fuera del mapa. Para los chequeos de
        seguridad del controlador: con pasillos donde la banda libre tras
        inflar mide ~6 cm, abortar apenas la pose toca el inflado hace la
        navegación frágil; contra la pared real el margen es el físico."""
        col, row = self.world_to_cell(x, y)
        if not (0 <= col < self._w and 0 <= row < self._h):
            return -1.0
        return self._dist[row * self._w + col]

    def segment_clear(self, a_xy, b_xy, min_clearance_m=None):
        # type: (Tuple[float, float], Tuple[float, float], Optional[float]) -> bool
        """True si el segmento recto entre dos puntos del mundo no cruza pared.

        Sin min_clearance_m se chequea contra la rejilla inflada (igual que
        el A*). Con min_clearance_m se exige esa distancia a la pared REAL
        en cada celda tocada: más laxo que el inflado, pensado para el
        chequeo de seguridad del lazo de control (ver clearance())."""
        a = self.world_to_cell(*a_xy)
        b = self.world_to_cell(*b_xy)
        return self._line_free(a, b, min_dist=min_clearance_m)

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
        cells = self._pivot_corners(cells)
        cells = self._shortcut(cells)
        path = [self.cell_to_world(c, r) for c, r in cells]
        # El último waypoint es el goal pedido (no el centro de celda) para
        # que la tolerancia de llegada se mida contra el clic real del host —
        # pero sólo si el clic cayó en zona transitable; si cayó en pared o
        # inflado se conserva el centro de la celda libre más cercana (antes
        # el tramo final llevaba al robot directo contra la pared).
        if self._is_free(*self.world_to_cell(*goal_xy)):
            path[-1] = (float(goal_xy[0]), float(goal_xy[1]))
        return path

    # ------------------------------------------------------------
    # Esquinas con pivote
    # ------------------------------------------------------------
    def _pivot_corners(self, cells):
        # type: (List[Tuple[int, int]]) -> List[Tuple[int, int]]
        """Reemplaza cada curva cerrada del camino por UN waypoint de pivote
        en el "bolsillo" de la intersección (la celda con máxima distancia a
        pared, ver _pocket_cell).

        Motivo: el robot es un cuadrado de 22 cm -> al rotar barre una
        semidiagonal de 15.6 cm, MÁS que la media holgura del pasillo
        (15 cm). Girar en arco dentro del pasillo roza inevitablemente la
        esquina interior aunque el camino tenga la holgura máxima posible
        (13.5 cm). En el bolsillo de la intersección (~17 cm de holgura) la
        rotación sí cabe. El controlador detecta estos waypoints por el
        ángulo del camino y rota en el lugar ahí (pivot_turn_min_rad /
        corner_capture_m en NavConfig)."""
        n = len(cells)
        w = _PIVOT_WINDOW
        if n < 2 * w + 2:
            return cells

        def _turn(i):
            # Giro local: ángulo entre la dirección de llegada y de salida
            # medido con una ventana de w celdas (suaviza el "escalonado"
            # diagonal de la rejilla 8-conexa, que no es un giro real).
            vix = cells[i][0] - cells[i - w][0]
            viy = cells[i][1] - cells[i - w][1]
            vox = cells[i + w][0] - cells[i][0]
            voy = cells[i + w][1] - cells[i][1]
            return abs(math.atan2(vix * voy - viy * vox, vix * vox + viy * voy))

        regions = []
        i = w
        while i < n - w:
            if _turn(i) > _PIVOT_REGION_RAD:
                j = i
                while j + 1 < n - w and _turn(j + 1) > _PIVOT_REGION_RAD:
                    j += 1
                # Giro neto de toda la zona (una S corta suma ~0: no pivotear).
                vix = cells[i][0] - cells[i - w][0]
                viy = cells[i][1] - cells[i - w][1]
                vox = cells[min(j + w, n - 1)][0] - cells[j][0]
                voy = cells[min(j + w, n - 1)][1] - cells[j][1]
                net = abs(math.atan2(vix * voy - viy * vox, vix * vox + viy * voy))
                if net >= _PIVOT_NET_RAD:
                    regions.append((i, j))
                i = j + 1
            else:
                i += 1

        out = list(cells)
        for s, e in reversed(regions):  # de atrás hacia adelante: índices estables
            pocket = self._pocket_cell(cells[(s + e) // 2])
            a, b = cells[s - 1], cells[e + 1]
            if (pocket != a and pocket != b
                    and self._line_free(a, pocket) and self._line_free(pocket, b)):
                out[s:e + 1] = [pocket]
            # si no hay bolsillo alcanzable en línea recta se conserva el
            # arco original de A* (mejor curva apretada que atravesar pared)
        return out

    def _pocket_cell(self, cell, max_steps=4):
        # type: (Tuple[int, int], int) -> Tuple[int, int]
        """Sube por el gradiente del campo de distancia hasta un máximo
        local: el punto con más holgura cerca de la esquina (en una
        intersección en L queda desplazado en diagonal hacia la esquina
        interior, donde el barrido de rotación del robot cabe entero)."""
        col, row = cell
        w = self._w
        for _ in range(max_steps):
            d0 = self._dist[row * w + col]
            best = None
            for dc in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if dc == 0 and dr == 0:
                        continue
                    nc, nr = col + dc, row + dr
                    if self._is_free(nc, nr) and self._dist[nr * w + nc] > d0:
                        d0 = self._dist[nr * w + nc]
                        best = (nc, nr)
            if best is None:
                break
            col, row = best
        return (col, row)

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
                    # Gradiente de inflado (ver _masks_from_distance): promedio
                    # entre la celda de salida y la de llegada, para que A*
                    # prefiera rutas por el centro del pasillo.
                    w = self._w
                    edge_cost = 0.5 * (self._cost[row * w + col] + self._cost[nr * w + nc])
                    tentative = g + step * (1.0 + edge_cost)
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
            # max_cost=0.0: el atajo sólo se toma si queda fuera de la zona
            # del gradiente de inflado; si no, se conservan los waypoints
            # intermedios de A* que ya pasan por el centro.
            while j > i + 1 and not self._line_free(cells[i], cells[j], max_cost=0.0):
                j -= 1
            out.append(cells[j])
            i = j
        return out

    def _line_free(self, a, b, max_cost=None, min_dist=None):
        # type: (Tuple[int, int], Tuple[int, int], Optional[float], Optional[float]) -> bool
        """Bresenham supercover: todas las celdas tocadas deben estar libres.

        Si max_cost no es None, además exige que el gradiente de inflado
        (ver _masks_from_distance) de cada celda tocada no lo supere.
        Si min_dist no es None, en vez de la rejilla inflada se exige esa
        distancia (m) a la pared real en cada celda tocada.
        """
        w = self._w

        def _ok(col, row):
            if not (0 <= col < self._w and 0 <= row < self._h):
                return False
            if min_dist is not None:
                return self._dist[row * w + col] >= min_dist
            if self._blocked[row * w + col]:
                return False
            if max_cost is not None and self._cost[row * w + col] > max_cost:
                return False
            return True

        c0, r0 = a
        c1, r1 = b
        dc = abs(c1 - c0)
        dr = abs(r1 - r0)
        sc = 1 if c1 > c0 else -1
        sr = 1 if r1 > r0 else -1
        err = dc - dr
        c, r = c0, r0
        while True:
            if not _ok(c, r):
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
                if not (_ok(c - sc, r) or _ok(c, r - sr)):
                    return False
