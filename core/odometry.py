"""Pose del robot integrada en la Jetson a partir de los encoders del ESP32.

El firmware ya NO calcula pose ni odometría on-board: el refactor que le quitó
IMU/Odometry/ToF (ver lib/Sensors/SensorHub.cpp) dejó la telemetría con sólo
`u.vel_left_cps`/`u.vel_right_cps` (cuentas/seg de cada rueda) y sin ningún
bloque "odo". Por eso la integración de pose se hace acá, igual que hace
`esp32_serial_bridge.py` en el stack ROS: cinemática diferencial a partir de
las velocidades de rueda, usando `CFG.robot.wheel_radius/wheel_separation/
wheel_cpr` (deben calzar con `Cfg::WHEEL_CPR` del firmware).

La pose arranca en la pose inicial configurada (config/CLI) y queda válida
de inmediato al llamar `attach()`/`reset()` — no hace falta esperar telemetría
del ESP32 para saber dónde está el robot, porque la posición inicial la fija
quien lanza `main.py`, no el hardware. Cada telemetría siguiente integra el
movimiento desde ahí; sin corrección externa la pose deriva con el tiempo.

Publica cada actualización como Ev.POSE y la refleja en `state.pose_*` para
consumidores síncronos (nav_server, controller).
"""
# PY36: sin `from __future__ import annotations`.
import logging
import math
import time

from typing import Optional  # PY36: añadido

from config import CFG
from core.bus import Ev, bus
from core.state import state

log = logging.getLogger(__name__)

_TWO_PI = 2.0 * math.pi

# Hueco de tiempo entre telemetrías por encima del cual no integramos ese
# tramo (reconexión del serial, freeze, etc.): un dt grande con una velocidad
# instantánea produciría un salto de pose irreal.
_MAX_INTEGRATION_GAP_S = 0.5


class Odometry:
    def __init__(self):
        self._x = CFG.nav.initial_x
        self._y = CFG.nav.initial_y
        self._yaw = CFG.nav.initial_yaw
        self._last_ts = None  # type: Optional[float]

    def attach(self):
        # type: () -> None
        self.reset(CFG.nav.initial_x, CFG.nav.initial_y, CFG.nav.initial_yaw)
        bus.on(Ev.TELEMETRY, self._on_telemetry)

    def detach(self):
        # type: () -> None
        bus.off(Ev.TELEMETRY, self._on_telemetry)

    def reset(self, x, y, yaw):
        # type: (float, float, float) -> None
        self._x = float(x)
        self._y = float(y)
        self._yaw = float(yaw)
        # La próxima telemetría vuelve a fijar el reloj de integración, para
        # no integrar un dt gigante acumulado mientras no llegaba nada.
        self._last_ts = None

        now = time.time()
        state.pose_x = self._x
        state.pose_y = self._y
        state.pose_yaw = self._yaw
        state.pose_v = 0.0
        state.pose_w = 0.0
        state.pose_stamp = now
        # Válida de inmediato: la pose inicial la define quien lanza el
        # servicio, no el ESP32 (que ya no reporta pose propia).
        state.pose_valid = True
        log.info("Odometría reiniciada a x=%.3f y=%.3f yaw=%.3f", x, y, yaw)

        bus.emit(Ev.POSE, {
            "x": self._x,
            "y": self._y,
            "yaw": self._yaw,
            "v": 0.0,
            "w": 0.0,
            "stamp": now,
            "valid": True,
        })

    # ------------------------------------------------------------
    def _on_telemetry(self, data):
        # type: (dict) -> None
        if not isinstance(data, dict):
            return
        u = data.get("u")
        if not isinstance(u, dict):
            return
        try:
            cps_left = float(u["vel_left_cps"])
            cps_right = float(u["vel_right_cps"])
        except (KeyError, TypeError, ValueError):
            return

        now = time.time()
        if self._last_ts is None:
            # Primera muestra tras (re)conectar: sólo fija el reloj de
            # integración, sin mover la pose (dt desconocido).
            self._last_ts = now
            return
        dt = now - self._last_ts
        self._last_ts = now
        if dt <= 0.0 or dt > _MAX_INTEGRATION_GAP_S:
            return

        rb = CFG.robot
        rad_per_count = _TWO_PI / rb.wheel_cpr
        v_left = cps_left * rad_per_count * rb.wheel_radius    # m/s
        v_right = cps_right * rad_per_count * rb.wheel_radius

        v = (v_left + v_right) / 2.0
        w = (v_right - v_left) / rb.wheel_separation

        # Integración exacta del arco (en vez de Euler simple) para no sesgar
        # la pose cuando w != 0: equivale a mover v*dt sobre una curva de
        # curvatura w en vez de una recta.
        dyaw = w * dt
        if abs(w) < 1e-6:
            dx = v * dt * math.cos(self._yaw)
            dy = v * dt * math.sin(self._yaw)
        else:
            r = v / w
            dx = r * (math.sin(self._yaw + dyaw) - math.sin(self._yaw))
            dy = -r * (math.cos(self._yaw + dyaw) - math.cos(self._yaw))

        self._x += dx
        self._y += dy
        self._yaw = math.atan2(math.sin(self._yaw + dyaw), math.cos(self._yaw + dyaw))

        state.pose_x = self._x
        state.pose_y = self._y
        state.pose_yaw = self._yaw
        state.pose_v = v
        state.pose_w = w
        state.pose_stamp = now
        state.pose_valid = True

        bus.emit(Ev.POSE, {
            "x": self._x,
            "y": self._y,
            "yaw": self._yaw,
            "v": v,
            "w": w,
            "stamp": now,
            "valid": True,
        })


# Singleton: main.py llama attach() al arrancar; nav_server puede usar reset().
odometry = Odometry()
