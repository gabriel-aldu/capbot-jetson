"""Proyección de un pixel de contacto con el piso a distancia métrica.

Adaptación en METROS del GroundPlaneMapper de capbot-identification-test
/test_stream.py (allá está en cm). Modelo: pinhole ideal (sin distorsión),
piso plano y cámara a altura fija. El borde inferior de la caja de una
detección se asume como el punto donde el objeto toca el piso; retroproyectar
ese pixel sobre el plano del piso da la distancia hacia adelante / lateral.
"""
# PY36: sin `from __future__ import annotations`; genéricos desde typing.
import math

from typing import Optional, Tuple


class GroundPlaneMapper(object):
    def __init__(self, img_w, img_h, height_m, pitch_deg=0.0,
                 hfov_deg=62.2, min_ground_m=None):
        # type: (int, int, float, float, float, Optional[float]) -> None
        self.cx = img_w / 2.0
        self.cy = img_h / 2.0
        self.fx = (img_w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        self.fy = self.fx                      # píxeles cuadrados asumidos
        self.height = height_m
        if min_ground_m is not None and min_ground_m > 0:
            # La fila inferior de la imagen ve el piso a min_ground_m, i.e.
            # min_ground = H / tan(pitch + beta_bottom); despejar pitch.
            beta = math.atan2(img_h - self.cy, self.fy)
            self.pitch = math.atan2(height_m, min_ground_m) - beta
        else:
            self.pitch = math.radians(pitch_deg)

    def locate(self, u, v):
        # type: (float, float) -> Optional[Tuple[float, float]]
        """Pixel (u, v) de un punto de contacto con el piso ->
        (adelante_m, lateral_m) con lateral + a la DERECHA de la cámara, o
        None cuando el rayo apunta al horizonte o más arriba (no toca piso)."""
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        s, c = math.sin(self.pitch), math.cos(self.pitch)
        down = y * c + s                       # componente del rayo hacia el piso
        if down <= 1e-6:
            return None
        t = self.height / down
        forward = t * (c - y * s)              # sobre el piso, hacia adelante
        lateral = t * x                        # + derecha / - izquierda
        return forward, lateral
