"""Modelo editable de paredes del laberinto (rejilla de celdas de 30 cm).

El maze físico es una rejilla de 5 columnas x 6 filas de celdas de 30x30 cm
cuyas paredes son secciones de 30 cm que viven EXACTAMENTE sobre las líneas
de la rejilla. En el PGM (0.003 m/px) cada pared es una línea de 1 px sobre
la columna/fila de píxeles de su línea de rejilla, así que el conjunto de
paredes se puede DETECTAR desde los píxeles y, a la inversa, los píxeles se
pueden RE-RENDERIZAR desde el conjunto de paredes. Con eso se puede agregar
o quitar una sección de pared en runtime: el host edita/dibuja y la Jetson
reconstruye el planner A* desde el mapa re-renderizado.

Este archivo se comparte VERBATIM entre capbot-jetson y capbot-host (misma
convención que core/occupancy_map.py): cualquier cambio debe copiarse al
otro repo.

Identificación de un segmento de pared (o, i, j):
  * ("v", i, j): pared vertical sobre la línea x = x0 + i*cell (i en 0..cols),
    cubriendo la celda-fila j (de y0 + j*cell a y0 + (j+1)*cell).
    Para i interior separa las celdas (i-1, j) y (i, j).
  * ("h", i, j): pared horizontal sobre la línea y = y0 + j*cell (j en 0..rows),
    cubriendo la celda-columna i (de x0 + i*cell a x0 + (i+1)*cell).
    Para j interior separa las celdas (i, j-1) y (i, j).

Sólo los segmentos INTERIORES son editables: el perímetro es fijo (quitarlo
dejaría al robot salir del área mapeada).
"""
# PY36: sin `from __future__ import annotations`; genéricos desde typing.
import math

from typing import Dict, List, Optional, Set, Tuple

Segment = Tuple[str, int, int]


class MazeGrid:
    """Geometría de la rejilla del laberinto en el frame del mapa."""

    def __init__(self, cols, rows, cell_m, x0, y0):
        # type: (int, int, float, float, float) -> None
        self.cols = cols        # celdas en x
        self.rows = rows        # celdas en y
        self.cell_m = cell_m    # lado de la celda (m)
        self.x0 = x0            # esquina inferior-izquierda de la rejilla
        self.y0 = y0

    def cell_of(self, x, y):
        # type: (float, float) -> Tuple[int, int]
        """Celda (ci, cj) que contiene el punto mundo, con clamp a la rejilla."""
        ci = int(math.floor((x - self.x0) / self.cell_m))
        cj = int(math.floor((y - self.y0) / self.cell_m))
        return (max(0, min(self.cols - 1, ci)), max(0, min(self.rows - 1, cj)))


# Rejillas conocidas, por nombre de mapa (mismo nombre que AVAILABLE_MAPS).
# "small" no tiene rejilla: es un rectángulo de prueba sin paredes interiores
# y no admite edición. Derivado del maze físico real: celdas de 30 cm,
# 5x6, con el frame map centrado en el centro del laberinto.
GRIDS = {
    "maze": MazeGrid(cols=5, rows=6, cell_m=0.30, x0=-0.75, y0=-0.90),
}  # type: Dict[str, MazeGrid]


def grid_for_map(map_name):
    # type: (str) -> Optional[MazeGrid]
    return GRIDS.get(map_name)


# ------------------------------------------------------------
# Validación / (de)serialización de segmentos
# ------------------------------------------------------------
def in_bounds(grid, seg):
    # type: (MazeGrid, Segment) -> bool
    o, i, j = seg
    if o == "v":
        return 0 <= i <= grid.cols and 0 <= j < grid.rows
    if o == "h":
        return 0 <= i < grid.cols and 0 <= j <= grid.rows
    return False


def is_interior(grid, seg):
    # type: (MazeGrid, Segment) -> bool
    """True si el segmento es editable (no pertenece al perímetro)."""
    if not in_bounds(grid, seg):
        return False
    o, i, j = seg
    if o == "v":
        return 1 <= i <= grid.cols - 1
    return 1 <= j <= grid.rows - 1


def parse_segment(data, grid):
    # type: (dict, MazeGrid) -> Optional[Segment]
    """Segmento desde un dict {'o','i','j'} (payload JSON); None si inválido."""
    try:
        seg = (str(data["o"]), int(data["i"]), int(data["j"]))
    except (KeyError, TypeError, ValueError):
        return None
    return seg if in_bounds(grid, seg) else None


def parse_walls(items, grid):
    # type: (list, MazeGrid) -> Set[Segment]
    """Set de segmentos interiores desde una lista JSON [["v",1,0], ...].
    Ignora silenciosamente entradas inválidas o del perímetro."""
    walls = set()  # type: Set[Segment]
    if not isinstance(items, list):
        return walls
    for it in items:
        if not (isinstance(it, (list, tuple)) and len(it) == 3):
            continue
        try:
            seg = (str(it[0]), int(it[1]), int(it[2]))
        except (TypeError, ValueError):
            continue
        if is_interior(grid, seg):
            walls.add(seg)
    return walls


def walls_to_list(walls):
    # type: (Set[Segment]) -> List[List]
    """Lista JSON-serializable, ordenada (determinista para persistir/difundir)."""
    return [[o, i, j] for o, i, j in sorted(walls)]


# ------------------------------------------------------------
# Píxeles <-> segmentos (requiere el OccupancyMap base del maze)
# ------------------------------------------------------------
def _vline_col(occ, grid, i):
    # type: (object, MazeGrid, int) -> int
    """Columna de píxeles de la línea vertical i de la rejilla."""
    return int((grid.x0 + i * grid.cell_m - occ.origin_x) / occ.resolution)


def _hline_row(occ, grid, j):
    # type: (object, MazeGrid, int) -> int
    """Fila de píxeles de la línea horizontal j de la rejilla."""
    return int(occ.height - (grid.y0 + j * grid.cell_m - occ.origin_y) / occ.resolution)


def _segment_pixels(occ, grid, seg):
    # type: (object, MazeGrid, Segment) -> List[int]
    """Índices (row*width+col) de la línea de 1 px del segmento, con extremos."""
    o, i, j = seg
    w, h = occ.width, occ.height
    out = []  # type: List[int]
    if o == "v":
        col = _vline_col(occ, grid, i)
        if not (0 <= col < w):
            return out
        r0 = _hline_row(occ, grid, j + 1)   # extremo superior (fila menor)
        r1 = _hline_row(occ, grid, j)       # extremo inferior (fila mayor)
        for r in range(max(0, r0), min(h - 1, r1) + 1):
            out.append(r * w + col)
    else:
        row = _hline_row(occ, grid, j)
        if not (0 <= row < h):
            return out
        c0 = _vline_col(occ, grid, i)
        c1 = _vline_col(occ, grid, i + 1)
        for c in range(max(0, c0), min(w - 1, c1) + 1):
            out.append(row * w + c)
    return out


def detect_walls(occ, grid, occupied_below=220):
    # type: (object, MazeGrid, int) -> Set[Segment]
    """Paredes INTERIORES presentes en los píxeles del mapa.

    Un segmento cuenta como presente si más de la mitad de su tramo interior
    (excluyendo ~20% en cada extremo, donde se cruzan paredes perpendiculares)
    está ocupado. Con el PGM limpio del maze (paredes de 1 px exactamente
    sobre la rejilla) esto es una detección exacta."""
    walls = set()  # type: Set[Segment]
    px = occ.pixels
    for o, ni, nj in (("v", grid.cols - 1, grid.rows), ("h", grid.cols, grid.rows - 1)):
        i0, j0 = (1, 0) if o == "v" else (0, 1)
        for i in range(i0, i0 + ni):
            for j in range(j0, j0 + nj):
                idxs = _segment_pixels(occ, grid, (o, i, j))
                if not idxs:
                    continue
                m = max(1, len(idxs) // 5)
                inner = idxs[m:-m]
                occ_n = sum(1 for k in inner if px[k] < occupied_below)
                if occ_n * 2 > len(inner):
                    walls.add((o, i, j))
    return walls


def render_walls(occ, grid, walls, free=255, occupied=0):
    # type: (object, MazeGrid, Set[Segment], int, int) -> bytes
    """Píxeles del mapa con el conjunto `walls` aplicado sobre el mapa base.

    Determinista: (1) limpia TODAS las líneas de la rejilla del mapa base,
    (2) estampa el perímetro completo (fijo), (3) estampa cada pared interior
    de `walls`. Así quitar una pared no deja huecos en las perpendiculares
    que comparten esquina, y agregar/quitar es idempotente."""
    px = bytearray(occ.pixels)

    for i in range(grid.cols + 1):
        for j in range(grid.rows):
            for k in _segment_pixels(occ, grid, ("v", i, j)):
                px[k] = free
    for j in range(grid.rows + 1):
        for i in range(grid.cols):
            for k in _segment_pixels(occ, grid, ("h", i, j)):
                px[k] = free

    perimeter = (
        [("v", 0, j) for j in range(grid.rows)]
        + [("v", grid.cols, j) for j in range(grid.rows)]
        + [("h", i, 0) for i in range(grid.cols)]
        + [("h", i, grid.rows) for i in range(grid.cols)]
    )
    for seg in perimeter:
        for k in _segment_pixels(occ, grid, seg):
            px[k] = occupied
    for seg in walls:
        if is_interior(grid, seg):
            for k in _segment_pixels(occ, grid, seg):
                px[k] = occupied
    return bytes(px)


# ------------------------------------------------------------
# Selección de segmento desde un clic (UI del host)
# ------------------------------------------------------------
def pick_segment(grid, x, y, tol_m):
    # type: (MazeGrid, float, float, float) -> Optional[Segment]
    """Segmento de rejilla más cercano al punto mundo (x, y), o None si el
    punto está a más de tol_m de toda línea o fuera de la rejilla."""
    if not (grid.x0 - tol_m <= x <= grid.x0 + grid.cols * grid.cell_m + tol_m):
        return None
    if not (grid.y0 - tol_m <= y <= grid.y0 + grid.rows * grid.cell_m + tol_m):
        return None

    fi = (x - grid.x0) / grid.cell_m
    fj = (y - grid.y0) / grid.cell_m
    i = max(0, min(grid.cols, int(round(fi))))   # línea vertical más cercana
    j = max(0, min(grid.rows, int(round(fj))))   # línea horizontal más cercana
    dxv = abs(x - (grid.x0 + i * grid.cell_m))
    dyh = abs(y - (grid.y0 + j * grid.cell_m))
    if min(dxv, dyh) > tol_m:
        return None

    if dxv <= dyh:
        cj = max(0, min(grid.rows - 1, int(math.floor(fj))))
        return ("v", i, cj)
    ci = max(0, min(grid.cols - 1, int(math.floor(fi))))
    return ("h", ci, j)


def segment_endpoints_world(grid, seg):
    # type: (MazeGrid, Segment) -> Tuple[Tuple[float, float], Tuple[float, float]]
    """Extremos del segmento en coordenadas del mundo (para dibujarlo)."""
    o, i, j = seg
    if o == "v":
        x = grid.x0 + i * grid.cell_m
        return ((x, grid.y0 + j * grid.cell_m),
                (x, grid.y0 + (j + 1) * grid.cell_m))
    y = grid.y0 + j * grid.cell_m
    return ((grid.x0 + i * grid.cell_m, y),
            (grid.x0 + (i + 1) * grid.cell_m, y))


# ------------------------------------------------------------
# Conectividad del grafo de celdas
# ------------------------------------------------------------
def components(grid, walls):
    # type: (MazeGrid, Set[Segment]) -> List[Set[Tuple[int, int]]]
    """Componentes conexas del grafo de celdas 5x6 según las paredes
    interiores presentes (el perímetro no afecta: siempre está)."""
    seen = set()  # type: Set[Tuple[int, int]]
    comps = []    # type: List[Set[Tuple[int, int]]]
    for sc in range(grid.cols):
        for sr in range(grid.rows):
            if (sc, sr) in seen:
                continue
            comp = set()  # type: Set[Tuple[int, int]]
            stack = [(sc, sr)]
            seen.add((sc, sr))
            while stack:
                ci, cj = stack.pop()
                comp.add((ci, cj))
                nbrs = (
                    (ci + 1, cj, ("v", ci + 1, cj)),
                    (ci - 1, cj, ("v", ci, cj)),
                    (ci, cj + 1, ("h", ci, cj + 1)),
                    (ci, cj - 1, ("h", ci, cj)),
                )
                for nc, nr, wall in nbrs:
                    if not (0 <= nc < grid.cols and 0 <= nr < grid.rows):
                        continue
                    if wall in walls or (nc, nr) in seen:
                        continue
                    seen.add((nc, nr))
                    stack.append((nc, nr))
            comps.append(comp)
    return comps


def connectivity_info(grid, walls, robot_xy=None):
    # type: (MazeGrid, Set[Segment], Optional[Tuple[float, float]]) -> Tuple[bool, int]
    """(conectado, celdas_inalcanzables). Inalcanzable = fuera de la componente
    de la celda del robot (o de la componente más grande si no hay pose)."""
    comps = components(grid, walls)
    if len(comps) <= 1:
        return True, 0
    if robot_xy is not None:
        cell = grid.cell_of(robot_xy[0], robot_xy[1])
        ref = next((c for c in comps if cell in c), max(comps, key=len))
    else:
        ref = max(comps, key=len)
    return False, grid.cols * grid.rows - len(ref)
