"""Estado compartido del servicio."""
# PY36: Este archivo no tenía `from __future__ import annotations` en el original,
#       así que no hay nada que quitar. Tampoco usa genéricos built-in ni el
#       operador `|`. Las dataclasses requieren `pip install dataclasses` (backport
#       oficial para 3.6) o Python 3.7+.
from dataclasses import dataclass
import time


@dataclass
class ServiceState:
    host_ip: str = ""
    host_last_seen: float = 0.0
    esp32_connected: bool = False
    esp32_last_seen: float = 0.0
    video_state: str = "stopped"
    emergency_active: bool = False

    # Pose del robot en el frame del mapa (odometría on-board del ESP32,
    # encoders+IMU, reexpresada al frame del mapa por core/odometry.py).
    pose_x: float = 0.0
    pose_y: float = 0.0
    pose_yaw: float = 0.0
    pose_v: float = 0.0        # m/s
    pose_w: float = 0.0        # rad/s
    pose_stamp: float = 0.0    # time.time() de la última muestra
    pose_valid: bool = False   # True tras la primera telemetría del ESP32

    # Contadores para diagnóstico
    cmds_received: int = 0
    cmds_dropped: int = 0   # frames corruptos o versión mala
    acks_sent: int = 0
    telemetry_published: int = 0

    def touch_host(self, ip: str) -> None:
        self.host_ip = ip
        self.host_last_seen = time.time()

    def touch_esp32(self) -> None:
        self.esp32_connected = True
        self.esp32_last_seen = time.time()


state = ServiceState()