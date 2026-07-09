"""Odometría diferencial integrada en la Jetson.

Reemplaza al par esp32_serial_bridge (/odom) + EKF del stack ROS2: el ESP32
sólo reporta velocidades crudas de rueda (vel_left_cps / vel_right_cps en
cuentas/seg, ver SensorHub::buildPayload) y aquí se integra la pose (x, y,
yaw) en el frame del mapa con el mismo esquema "midpoint" que usaba
diff_drive_controller.

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

_TWO_PI = 2.0 * math.pi

# Si pasa más de esto entre telemetrías, el dt no es confiable (link caído,
# arranque): se descarta el paso de integración.
_MAX_DT_S = 0.5


class Odometry:
    def __init__(self):
        self._x = CFG.nav.initial_x
        self._y = CFG.nav.initial_y
        self._theta = CFG.nav.initial_yaw
        self._last_t = None  # type: float

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
        self._theta = float(yaw)
        self._last_t = None
        log.info("Odometría reiniciada a x=%.3f y=%.3f yaw=%.3f", x, y, yaw)

    # ------------------------------------------------------------
    def _on_telemetry(self, data):
        # type: (dict) -> None
        if not isinstance(data, dict):
            return
        u = data.get("u")
        if not isinstance(u, dict):
            return
        try:
            vel_left_cps = float(u["vel_left_cps"])
            vel_right_cps = float(u["vel_right_cps"])
        except (KeyError, TypeError, ValueError):
            return

        rb = CFG.robot
        # cuentas/s -> rad/s de rueda -> m/s tangencial.
        omega_left = (vel_left_cps / rb.wheel_cpr) * _TWO_PI
        omega_right = (vel_right_cps / rb.wheel_cpr) * _TWO_PI
        v_left = omega_left * rb.wheel_radius
        v_right = omega_right * rb.wheel_radius

        # Cinemática diferencial estándar.
        v = (v_left + v_right) / 2.0
        w = (v_right - v_left) / rb.wheel_separation

        now = time.time()
        if self._last_t is None or (now - self._last_t) > _MAX_DT_S:
            dt = 0.0
        else:
            dt = now - self._last_t
        self._last_t = now

        # Integración midpoint (más precisa que Euler cuando w != 0).
        half_dtheta = 0.5 * w * dt
        self._x += v * math.cos(self._theta + half_dtheta) * dt
        self._y += v * math.sin(self._theta + half_dtheta) * dt
        self._theta += w * dt
        self._theta = math.atan2(math.sin(self._theta), math.cos(self._theta))

        state.pose_x = self._x
        state.pose_y = self._y
        state.pose_yaw = self._theta
        state.pose_v = v
        state.pose_w = w
        state.pose_stamp = now
        state.pose_valid = True

        bus.emit(Ev.POSE, {
            "x": self._x,
            "y": self._y,
            "yaw": self._theta,
            "v": v,
            "w": w,
            "stamp": now,
            "valid": True,
        })


# Singleton: main.py llama attach() al arrancar; nav_server puede usar reset().
odometry = Odometry()
