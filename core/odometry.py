"""Pose del robot a partir de la odometría on-board del ESP32.

El ESP32 (lib/Sensors/Odometry.cpp) fusiona encoders + IMU (filtro
complementario gyro-Z) y manda su propia estimación de pose/velocidad en el
bloque "odo" de cada telemetría (ver SensorHub::buildPayload): x, y, a (yaw en
grados), v (m/s), w (yaw rate en grados/s). La Jetson ya NO reintegra desde
vel_left_cps/vel_right_cps (eso quedó obsoleto y perdía la corrección de la
IMU); sólo reexpresa esa pose en el frame del mapa.

El ESP32 arranca su odometría siempre en (0, 0, 0) (Odometry::reset() en su
setup()) y nunca se le resetea desde la Jetson. Por eso, al conectar (o tras
un reset()), esta clase fija como referencia la primera muestra "odo" recibida
y le aplica una transformación rígida (rotación + traslación) para que esa
referencia coincida con la pose inicial configurada (config/CLI), y las
muestras siguientes seguidas de esa misma transformación.

Sin corrección externa (ArUco/EKF) la pose deriva con el tiempo; la pose
inicial se fija por config/CLI y debe corresponder a dónde se coloca
físicamente el robot en el mapa.

Publica cada actualización como Ev.POSE y la refleja en `state.pose_*` para
consumidores síncronos (nav_server, controller).
"""
# PY36: sin `from __future__ import annotations`.
import logging
import math
import time

from config import CFG
from core.bus import Ev, bus
from core.state import state

log = logging.getLogger(__name__)


class Odometry:
    def __init__(self):
        self._map_x0 = CFG.nav.initial_x
        self._map_y0 = CFG.nav.initial_y
        self._map_yaw0 = CFG.nav.initial_yaw
        # Primera muestra "odo" del ESP32 tras el (re)arranque; sirve de
        # referencia para la transformación rígida. None = todavía sin fijar.
        self._ref_fx = None    # type: float
        self._ref_fy = None    # type: float
        self._ref_ftheta = None  # type: float

    def attach(self):
        # type: () -> None
        self.reset(CFG.nav.initial_x, CFG.nav.initial_y, CFG.nav.initial_yaw)
        bus.on(Ev.TELEMETRY, self._on_telemetry)

    def detach(self):
        # type: () -> None
        bus.off(Ev.TELEMETRY, self._on_telemetry)

    def reset(self, x, y, yaw):
        # type: (float, float, float) -> None
        self._map_x0 = float(x)
        self._map_y0 = float(y)
        self._map_yaw0 = float(yaw)
        # La próxima telemetría recibida vuelve a fijar la referencia.
        self._ref_fx = None
        self._ref_fy = None
        self._ref_ftheta = None
        state.pose_valid = False
        log.info("Odometría reiniciada a x=%.3f y=%.3f yaw=%.3f", x, y, yaw)

    # ------------------------------------------------------------
    def _on_telemetry(self, data):
        # type: (dict) -> None
        if not isinstance(data, dict):
            return
        odo = data.get("odo")
        if not isinstance(odo, dict):
            return
        try:
            fx = float(odo["x"])
            fy = float(odo["y"])
            ftheta = math.radians(float(odo["a"]))
            v = float(odo["v"])
            w = math.radians(float(odo["w"]))
        except (KeyError, TypeError, ValueError):
            return

        if self._ref_fx is None:
            self._ref_fx = fx
            self._ref_fy = fy
            self._ref_ftheta = ftheta

        # Transformación rígida: delta en el frame del ESP32 (desde la
        # referencia) rotado y trasladado al frame del mapa.
        dx = fx - self._ref_fx
        dy = fy - self._ref_fy
        dtheta = ftheta - self._ref_ftheta

        yaw0 = self._map_yaw0
        cos0 = math.cos(yaw0)
        sin0 = math.sin(yaw0)
        x = self._map_x0 + dx * cos0 - dy * sin0
        y = self._map_y0 + dx * sin0 + dy * cos0
        yaw = math.atan2(math.sin(yaw0 + dtheta), math.cos(yaw0 + dtheta))

        now = time.time()
        state.pose_x = x
        state.pose_y = y
        state.pose_yaw = yaw
        state.pose_v = v
        state.pose_w = w
        state.pose_stamp = now
        state.pose_valid = True

        bus.emit(Ev.POSE, {
            "x": x,
            "y": y,
            "yaw": yaw,
            "v": v,
            "w": w,
            "stamp": now,
            "valid": True,
        })


# Singleton: main.py llama attach() al arrancar; nav_server puede usar reset().
odometry = Odometry()
