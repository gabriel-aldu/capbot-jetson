"""Configuración centralizada del servicio Jetson.

Todos los valores pueden sobrescribirse desde CLI (ver main.py).
"""
import os

from dataclasses import dataclass, field

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


@dataclass
class NetworkConfig:
    # Escucha de comandos
    listen_host: str = "0.0.0.0"
    udp_cmd_port: int = 5005        # host -> jetson
    udp_ack_port: int = 5006        # jetson -> host
    ws_telemetry_port: int = 8765   # jetson -> host (WS)
    video_port: int = 5000          # jetson -> host (UDP RTP H264)
    # WS de navegación (mismo puerto/protocolo JSON que el antiguo
    # gui_bridge_node de ROS2: goal/cancel <-, pose/nav_status/map_name ->).
    nav_ws_port: int = 8766

    # IP del host. Se setea desde CLI o por detección automática
    # (el primer comando UDP recibido fija la IP).
    host_ip: str = ""

    # Heartbeat: si no llega nada del host en este tiempo, detener motores.
    # M ms en el requisito -> 500ms es un valor conservador por defecto.
    host_heartbeat_timeout_ms: int = 500


@dataclass
class ProtocolConfig:
    magic: int = 0xABCD
    version: int = 1
    frame_size: int = 16


@dataclass
class SerialConfig:
    port: str = "/dev/ttyTHS1"
    baudrate: int = 115200
    # Si la Jetson no recibe nada del ESP32 en este tiempo,
    # consideramos el link caído (no detiene motores directamente;
    # eso lo hace el ESP32 por su propio watchdog).
    rx_timeout_ms: int = 300


@dataclass
class TelemetryConfig:
    publish_hz: int = 50
    # Si la cola WS supera este tamaño asumimos que un cliente está atascado
    # y soltamos paquetes antiguos para no crecer sin límite.
    ws_queue_max: int = 100


@dataclass
class VideoConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    bitrate_kbps: int = 4000  # H264 hardware. Bajar para menos latencia.
    # Iframe frecuente ayuda a recuperar tras pérdidas de paquetes
    iframe_interval: int = 15


@dataclass
class RobotConfig:
    """Geometría del robot. Debe calzar con el firmware (Cfg::WHEEL_CPR) y con
    los valores que usaba el stack ROS (esp32_serial_bridge.py)."""
    wheel_radius: float = 0.035      # m
    wheel_separation: float = 0.17   # m (track width)
    wheel_cpr: float = 910.0         # cuentas/vuelta del encoder (cuadratura 4x)
    max_linear_speed: float = 0.15   # m/s, clamp del (v,w) de navegación
    max_angular_speed: float = 1.0   # rad/s


@dataclass
class NavConfig:
    """Navegación autónoma (reemplaza a nav2 + gui_bridge_node de ROS2)."""
    # Nombre del mapa activo. Debe existir en AVAILABLE_MAPS y coincidir con
    # los assets del host (el host lo auto-selecciona al recibir map_name).
    # OJO: "small" es un rectángulo vacío sin paredes interiores; con él el
    # planner traza líneas rectas. Para la arena con paredes usar "maze".
    map_name: str = "maze"

    # Pose inicial del robot en el frame del mapa (m, m, rad). La odometría
    # integra desde aquí; sin corrección externa (ArUco/EKF) la pose deriva.
    initial_x: float = 0.0
    initial_y: float = 0.0
    initial_yaw: float = 0.0

    # Planificación
    inflation_radius_m: float = 0.01   # radio de inflado de obstáculos
    occupied_below: int = 220          # pixel PGM < esto => celda bloqueada (205=unknown)

    # Seguimiento de trayectoria (pure pursuit)
    lookahead_m: float = 0.15
    goal_tolerance_m: float = 0.08
    yaw_tolerance_rad: float = 0.15
    control_rate_hz: float = 20.0
    cruise_speed: float = 0.1          # m/s en tramo recto
    k_heading: float = 2.0             # w = k * error de rumbo

    # Publicación de pose al host (WS nav)
    pose_publish_hz: float = 10.0


# Mapas disponibles (nombre -> (pgm, yaml)). Copias de capbot-host/assets.
AVAILABLE_MAPS = {
    "small": (
        os.path.join(_ASSETS_DIR, "test_map_small.pgm"),
        os.path.join(_ASSETS_DIR, "test_map_small.yaml"),
    ),
    "maze": (
        os.path.join(_ASSETS_DIR, "test_map_maze.pgm"),
        os.path.join(_ASSETS_DIR, "test_map_maze.yaml"),
    ),
}


@dataclass
class Config:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    serial: SerialConfig = field(default_factory=SerialConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    nav: NavConfig = field(default_factory=NavConfig)


# Singleton mutable — se sobrescribe desde main.py tras parsear argv
CFG = Config()