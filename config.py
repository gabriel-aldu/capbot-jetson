"""Configuración centralizada del servicio Jetson.

Todos los valores pueden sobrescribirse desde CLI (ver main.py).
"""
import os

from dataclasses import dataclass, field

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(_BASE_DIR, "assets")


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
    baudrate: int = 460800
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
    wheel_radius: float = 0.034      # m
    wheel_separation: float = 0.17   # m (track width)
    wheel_cpr: float = 898.0         # cuentas/vuelta del encoder (cuadratura 4x)
    max_linear_speed: float = 0.1   # m/s, clamp del (v,w) de navegación
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
    # integra desde aquí; sin corrección externa la pose deriva.
    # Esquina suroeste (abajo-izquierda) del área libre de test_map_small,
    # con el centro de rotación del robot (caja de 20x17cm, centro de
    # rotación/IMU a 5cm de la parte trasera) desplazado 11cm de cada
    # pared para no arrancar incrustado en ellas. Yaw 0 = mirando al este (+X).
    initial_x: float = 0.0
    initial_y: float = -0.75
    initial_yaw: float = 0.0

    # Planificación
    # El planner trata al robot como un punto en su centro de rotación
    # (ver planner.py). Con la caja real de 20x17cm y el centro de
    # rotación a 5cm de la parte trasera (12cm del frente), el radio
    # circunscrito para poder girar en el sitio sin chocar es
    # sqrt(10^2 + 12^2) ≈ 0.156m; se redondea a 0.16 con margen.
    inflation_radius_m: float = 0.12   # radio de inflado de obstáculos
    occupied_below: int = 220          # pixel PGM < esto => celda bloqueada (205=unknown)
    # Resolución de la rejilla de PLANIFICACIÓN (m/celda). El PGM se
    # submuestrea a esta resolución antes de A* (min-pooling conservador:
    # una pared de 1 px nunca desaparece). Con el maze @ 0.003 m/px, 0.015
    # deja la rejilla en ~101x121 celdas (25x menos que el PGM). Si es <=
    # la resolución nativa del mapa no se submuestrea (p.ej. "small" @ 0.025).
    planning_resolution_m: float = 0.03
    # Sesgo hacia el centro: más allá del inflado, las celdas a menos de
    # center_bias_radius_m de una pared pagan un costo extra en A* (decae
    # linealmente a 0 en ese radio), empujando la ruta a pasar por el medio
    # del pasillo en vez de rozar el borde del inflado. 0 = sin sesgo.
    center_bias_radius_m: float = 0.15
    center_bias_weight: float = 6.0

    # Esquinas con pivote: el planner coloca en cada giro brusco un waypoint
    # en el "bolsillo" de la intersección (el punto con más holgura, donde el
    # barrido diagonal de 15.6 cm del chasis de 22x22 sí cabe al rotar). Si
    # el camino gira más que pivot_turn_min_rad en un waypoint, pure pursuit
    # se acerca a corner_capture_m de él (en vez del lookahead) y el umbral
    # de giro en el lugar hace el resto: parar, pivotear, seguir recto.
    pivot_turn_min_rad: float = 0.9    # ~52°: >45° del escalonado diagonal
    corner_capture_m: float = 0.03

    # Seguridad del lazo de control, medida contra la pared REAL (no el
    # inflado; ver AStarPlanner.clearance): abortar sólo si el centro de
    # rotación queda a menos de abort_clearance_m de una pared (con la caja
    # de 20 cm de ancho, 0.10 = el costado ya está tocando), y replanificar
    # sólo si el tramo recto al target pasa a menos de segment_clearance_m.
    abort_clearance_m: float = 0.10
    segment_clearance_m: float = 0.11

    # Seguimiento de trayectoria (pure pursuit)
    lookahead_m: float = 0.1
    goal_tolerance_m: float = 0.08 * 0.7
    yaw_tolerance_rad: float = 0.15 * 0.7
    control_rate_hz: float = 20.0
    cruise_speed: float = 0.1          # m/s en tramo recto
    k_heading: float = 2.0             # w = k * error de rumbo

    # Publicación de pose al host (WS nav)
    pose_publish_hz: float = 5.0

    # Persistencia de la edición de paredes del maze (core/maze_walls.py):
    # el conjunto de paredes editado se guarda aquí y se reaplica al arrancar.
    # El PGM original nunca se modifica en disco; borrar este archivo (o usar
    # "Restaurar paredes originales" en el host) vuelve al laberinto del PGM.
    walls_state_path: str = os.path.join(_ASSETS_DIR, "maze_walls.json")


@dataclass
class PerceptionConfig:
    """Detección de obstáculos con la DNN (capbot-identification-test).

    La cámara CSI se comparte con el streaming al host mediante un `tee` en
    la pipeline de GStreamer (net/video_pipeline.py): una rama sigue mandando
    H.264 al host y otra entrega frames BGR reducidos a un appsink que
    consume el hilo de inferencia (perception/detector.py). Cada detección
    se proyecta al piso con el modelo pinhole+plano (perception/ground_plane
    .py), se reexpresa en el frame del mapa con la pose odométrica y se
    acumula por celda de 30 cm (controller/obstacle_tracker.py): una celda
    con objeto se estampa como ocupada en el planner y se difunde al host.
    """
    enabled: bool = True
    # Engine TensorRT (serializado con trtexec/ultralytics para ESTA Jetson;
    # copiar aquí el bottles_fp16.engine de capbot-identification-test).
    engine_path: str = os.path.join(_ASSETS_DIR, "bottles_fp16.engine")
    imgsz: int = 416
    conf_threshold: float = 0.25
    iou_threshold: float = 0.50

    # Rama de análisis del tee (resolución reducida para bajar el costo de
    # nvvidconv/videoconvert; la letterbox del modelo reescala igual).
    infer_width: int = 640
    infer_height: int = 360
    # Techo de inferencias por segundo (el appsink descarta frames viejos;
    # 5 Hz sobra para marcar celdas y deja GPU para el encoder H.264).
    infer_max_hz: float = 5.0

    # Geometría de la cámara (misma convención que capbot-identification-test
    # /test_stream.py, en metros). min_ground: distancia medida de la cámara
    # al punto de piso visible en el borde INFERIOR de la imagen; con ella se
    # calibra el pitch y pitch_deg se ignora.
    cam_height_m: float = 0.09
    cam_pitch_deg: float = 0.0
    cam_hfov_deg: float = 62.2
    cam_min_ground_m: float = 0.29
    # Posición de la cámara respecto del centro de rotación del robot
    # (caja 20x17 cm, centro de rotación a 5 cm de la trasera -> borde
    # frontal a ~0.12 m). lateral: + izquierda en el frame del robot.
    cam_forward_offset_m: float = 0.12
    cam_lateral_offset_m: float = 0.0

    # Sólo se confía en detecciones hasta esta distancia al robot (la
    # estimación por plano de piso se degrada lejos y con cajas recortadas).
    max_range_m: float = 1.2
    # Una celda se BLOQUEA tras verse ocupada de forma continua este tiempo
    # (filtra falsos positivos de un frame).
    mark_debounce_s: float = 0.5
    # Una celda bloqueada se LIBERA tras este tiempo continuo sin detecciones
    # que caigan en ella, sólo mientras la celda está a la vista de la cámara
    # (dentro del FOV, en rango y con línea de vista libre de paredes).
    clear_debounce_s: float = 5.0
    # Margen (grados) restado a cada lado del HFOV para el test "a la vista"
    # (los bordes del frame recortan cajas y no son confiables).
    fov_margin_deg: float = 8.0


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
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)


# Singleton mutable — se sobrescribe desde main.py tras parsear argv
CFG = Config()