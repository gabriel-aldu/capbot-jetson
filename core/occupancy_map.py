"""Carga de mapas de ocupación (.pgm + .yaml de ROS) y transformaciones.

Copia PY36-compatible de capbot-host/core/occupancy_map.py (misma convención
de nav2_map_server) para que Jetson y host planifiquen/dibujen sobre el MISMO
mapa y frame de coordenadas:
  * El .pgm (P5 binario) es la rejilla: 255 = libre, 0 = ocupado (negate:0),
    205 = desconocido.
  * El .yaml da `resolution` (m/px) y `origin` [ox, oy, theta]: esquina
    INFERIOR-IZQUIERDA de la imagen en coordenadas del mundo (frame map).

Transformaciones (py invertida: la fila 0 del PGM es la parte superior):
    px = (x - ox) / res
    py = H - (y - oy) / res
"""
# PY36: sin `from __future__ import annotations`; genéricos desde typing.
from dataclasses import dataclass
from typing import Dict, List, Tuple


def _read_pgm_tokens(raw, count):
    # type: (bytes, int) -> Tuple[List[bytes], int]
    """Lee `count` tokens ASCII de la cabecera PGM saltando comentarios."""
    tokens = []  # type: List[bytes]
    i = 0
    n = len(raw)
    while len(tokens) < count:
        while i < n and raw[i:i + 1].isspace():
            i += 1
        if i < n and raw[i:i + 1] == b"#":
            while i < n and raw[i:i + 1] not in (b"\n", b"\r"):
                i += 1
            continue
        start = i
        while i < n and not raw[i:i + 1].isspace():
            i += 1
        if start == i:
            break
        tokens.append(raw[start:i])
    return tokens, i


@dataclass
class OccupancyMap:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    pixels: bytes  # width*height bytes, fila 0 = arriba; valor en [0,255]

    # ---- Transformaciones mundo <-> píxel ----
    def world_to_pixel(self, x, y):
        # type: (float, float) -> Tuple[float, float]
        px = (x - self.origin_x) / self.resolution
        py = self.height - (y - self.origin_y) / self.resolution
        return px, py

    def pixel_to_world(self, px, py):
        # type: (float, float) -> Tuple[float, float]
        x = self.origin_x + px * self.resolution
        y = self.origin_y + (self.height - py) * self.resolution
        return x, y

    def pixel_at(self, col, row):
        # type: (int, int) -> int
        return self.pixels[row * self.width + col]


def _coerce(val):
    try:
        return float(val)
    except ValueError:
        return val


def _parse_yaml(path):
    # type: (str) -> Dict
    """Parser mínimo del map.yaml de ROS (sin PyYAML). Igual que en el host."""
    out = {}  # type: Dict
    pending_key = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("- ") or stripped == "-":
                if pending_key is not None:
                    out.setdefault(pending_key, []).append(
                        _coerce(stripped[1:].strip()))
                continue
            if ":" not in stripped:
                continue
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            if not val:
                pending_key = key
                out[key] = []
                continue
            pending_key = None
            if val.startswith("[") and val.endswith("]"):
                items = [v.strip() for v in val[1:-1].split(",") if v.strip()]
                out[key] = [_coerce(v) for v in items]
            else:
                out[key] = _coerce(val)
    return out


def load_map(pgm_path, yaml_path):
    # type: (str, str) -> OccupancyMap
    """Carga un mapa de ocupación desde su .pgm (P5) y su .yaml asociado."""
    meta = _parse_yaml(yaml_path)
    resolution = float(meta.get("resolution", 0.05))
    origin = meta.get("origin", [0.0, 0.0, 0.0])
    origin_x = float(origin[0]) if len(origin) > 0 else 0.0
    origin_y = float(origin[1]) if len(origin) > 1 else 0.0

    with open(pgm_path, "rb") as f:
        raw = f.read()

    tokens, offset = _read_pgm_tokens(raw, 4)
    if len(tokens) < 4 or tokens[0] != b"P5":
        raise ValueError("PGM no soportado (se esperaba binario P5): {}".format(pgm_path))
    width = int(tokens[1])
    height = int(tokens[2])
    data_start = offset + 1
    expected = width * height
    pixels = raw[data_start:data_start + expected]
    if len(pixels) < expected:
        raise ValueError(
            "PGM truncado: {} bytes, se esperaban {}".format(len(pixels), expected))

    return OccupancyMap(
        width=width,
        height=height,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        pixels=pixels,
    )
