"""Bus de eventos asyncio.

Equivalente al SignalBus de Qt del dashboard, pero basado en asyncio.
Cada evento es un nombre + payload. Los productores hacen `bus.emit(event, data)`
y los consumidores se suscriben con `bus.on(event, callback)`.

Esto desacopla:
  - El UDP server del ESP32 link (el primero empuja comandos al bus, el segundo
    los consume sin conocer UDP).
  - El ESP32 link del WS server (el primero publica telemetría al bus, el segundo
    la re-emite a los clientes WS sin tocar el serial).
  - El heartbeat de los motores (el watchdog emite un evento, el link ESP32 lo
    consume y envía STOP).
"""
# PY36: Eliminado `from __future__ import annotations`.
import asyncio
import logging
from collections import defaultdict

# PY36: `Dict`, `List` desde typing (en 3.6 no se puede escribir `dict[...]`
#       ni `list[...]` como anotaciones). `Union` reemplaza al operador `|`.
from typing import Any, Awaitable, Callable, Dict, List, Union

log = logging.getLogger(__name__)

# PY36: El alias original era
#           Callable[[Any], Awaitable[None] | None]
#       El `|` entre tipos requiere 3.10. Lo reescribimos con `Union`.
Callback = Callable[[Any], Union[Awaitable[None], None]]


class EventBus:
    """Bus pub/sub ligero sobre asyncio."""

    def __init__(self) -> None:
        # PY36: `dict[str, list[Callback]]` como anotación no funciona en 3.6.
        #       Usamos `Dict[str, List[Callback]]` de `typing`.
        self._subs = defaultdict(list)  # type: Dict[str, List[Callback]]

    def on(self, event: str, cb: Callback) -> None:
        self._subs[event].append(cb)

    def off(self, event: str, cb: Callback) -> None:
        if cb in self._subs[event]:
            self._subs[event].remove(cb)

    def emit(self, event: str, data: Any = None) -> None:
        """Emite un evento sin esperar a los consumidores.

        Los callbacks asincrónicos se lanzan como tareas; los sincrónicos
        se ejecutan en línea.
        """
        for cb in list(self._subs.get(event, ())):
            try:
                result = cb(data)
                if asyncio.iscoroutine(result):
                    # PY36: `asyncio.create_task` no existe (se añadió en 3.7).
                    #       Usamos `ensure_future`, que toma el loop actual si
                    #       hay uno en ejecución. Si `emit` se llama desde un
                    #       contexto sin loop activo, esto lanzará RuntimeError,
                    #       igual que el `create_task` original.
                    asyncio.ensure_future(result)
            except Exception:
                log.exception("Error en callback de evento %s", event)


# Singleton
bus = EventBus()


# ------------------------------------------------------------
# Nombres de eventos (para evitar strings mágicos dispersos)
# ------------------------------------------------------------
class Ev:
    # Desde UDP server (host -> jetson)
    CMD_MOTOR = "cmd.motor"              # dict {left, right, aux, seq}
    CMD_VEL = "cmd.vel"                  # dict {linear, angular, seq}
    CMD_HEARTBEAT = "cmd.heartbeat"      # dict {seq}
    CMD_EMERGENCY = "cmd.emergency"      # dict {seq}
    CMD_PID_PARAM = "cmd.pid_param"      # dict {ctrl_id, param_id, value, seq}
    CMD_SETPOINT = "cmd.setpoint"        # dict {comp_id, value, seq}
    CMD_MODE = "cmd.mode"                # dict {mode, seq}

    # Estados de conexión / watchdog
    HOST_ONLINE = "host.online"          # str (ip)
    HOST_OFFLINE = "host.offline"        # None
    ESP32_ONLINE = "esp32.online"        # None
    ESP32_OFFLINE = "esp32.offline"      # str (reason)

    # Desde ESP32 link (esp32 -> jetson)
    TELEMETRY = "telemetry"              # dict (ya decodificado)

    # Pose fusionada por el ESP32 (encoders+IMU), reexpresada en el frame del
    # mapa por core/odometry.py
    POSE = "pose"                        # dict {x, y, yaw, v, w, stamp, valid}

    # Navegación (host <-> jetson por WS nav, reemplazo de gui_bridge_node)
    NAV_GOAL = "nav.goal"                # dict {x, y, yaw}
    NAV_CANCEL = "nav.cancel"            # None
    NAV_STATUS = "nav.status"            # dict {state, [distance_remaining]}

    # Edición de paredes del maze (host -> jetson por WS nav; ver
    # controller/wall_editor.py y core/maze_walls.py)
    WALL_EDIT = "wall.edit"              # dict {action:"add"|"remove"|"reset", [o,i,j]}
    WALL_RESULT = "wall.result"          # dict {ok, action, [reason]}
    WALLS_CHANGED = "walls.changed"      # dict {map, walls, connected, unreachable}

    # Percepción de obstáculos (DNN sobre la cámara CSI; ver perception/ y
    # controller/obstacle_tracker.py)
    # points: proyectados al mapa (para el tracker); boxes: TODAS las cajas
    # crudas normalizadas 0..1 (para dibujarlas sobre el video en el host).
    DETECTIONS = "perception.detections"   # dict {stamp, fps, pose|None,
                                           #   points:[{x,y,cls,conf,clipped,dist_m}],
                                           #   boxes:[{box:[x1,y1,x2,y2],cls,conf,clipped,dist_m|None}]}
    OBSTACLES_CHANGED = "obstacles.changed"  # dict {map, cells:[[i,j],..]}

    # Orden interna: detener motores ya (por watchdog o emergencia)
    STOP_MOTORS = "motors.stop"          # None

    # Video
    VIDEO_STATE = "video.state"          # str