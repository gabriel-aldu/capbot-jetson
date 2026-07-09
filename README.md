# Jetson Nano Service

Servicio Python que corre en la Jetson Nano. Reemplaza por completo al stack
ROS2 (nav2 + EKF + gui_bridge_node): hace de puente entre el PC host y el
ESP32, gestiona el video de la cámara IMX219, coordina heartbeats/paros de
emergencia, integra la odometría y ejecuta la navegación autónoma
(clic en el mapa del host → el robot viaja al punto).

```
                UDP 5005 (comandos)              Serial COBS+CRC16
┌──────────┐ ─────────────────────>  ┌────────┐ ────────────> ┌───────┐
│ PC Host  │ <────────────────────── │ Jetson │ <──────────── │ ESP32 │
│          │    UDP 5006 (ACKs)      │        │   telemetría  │       │
│          │ <────────────────────── │        │               └───────┘
│          │   WS 8765 (telemetría)  │        │
│          │ <─────────────────────> │        │
│          │   WS 8766 (navegación)  │        │
│          │ <────────────────────── │        │
│          │   UDP 5000 (video H264) │        │
└──────────┘                         └────────┘
```

## Navegación (reemplazo de ROS2)

Flujo del "2D Goal Pose" del host:

1. El host manda `{"type":"goal","x":..,"y":..,"yaw":..}` por WS 8766
   (mismo protocolo JSON que el antiguo `gui_bridge_node`; el host no cambia).
2. `controller/planner.py` planifica con A* sobre el mapa de ocupación
   (`assets/*.pgm|.yaml`, copias de los del host) inflado por el radio del
   robot.
3. `controller/controller.py` sigue el camino con pure pursuit usando la
   odometría diferencial integrada en `core/odometry.py` (a partir de
   `vel_left_cps`/`vel_right_cps` del ESP32) y emite `(v, w)` del chasis.
4. `hw/esp32_link.py` convierte `(v, w)` a rad/s por rueda (cinemática
   diferencial) y manda `VEL_CMD` al ESP32, cuyo PID por rueda hace el resto.
5. El progreso (`accepted/active/succeeded/...` + `distance_remaining`) y la
   pose se difunden por WS 8766 con el mismo JSON que usaba ROS2.

Al aceptar un goal, la Jetson manda `MODE_CMD(1)` (AUTONOMOUS_NAV) al ESP32
automáticamente; el switch de modo del host sigue funcionando para volver a
manual.

**Localización**: sólo odometría de ruedas (sin ArUco/EKF por ahora): la pose
deriva con el tiempo y la pose inicial debe fijarse con `--start-x/y/yaw`
según dónde se coloque el robot en el mapa.

## Arquitectura

```
jetson_service/
├── main.py                 # Entry point (asyncio loop)
├── config.py               # Puertos, timeouts, serial, robot, navegación
├── assets/                 # Mapas .pgm/.yaml (idénticos a capbot-host/assets)
├── core/
│   ├── bus.py              # Bus de eventos asyncio
│   ├── state.py            # Estado compartido (incluye pose)
│   ├── heartbeat.py        # Watchdogs de red + serial
│   ├── odometry.py         # Odometría diferencial desde telemetría del ESP32
│   └── occupancy_map.py    # Loader de mapas ROS (.pgm + .yaml)
├── protocol/
│   ├── udp_frame.py        # Frame binario 16B (idéntico al host)
│   └── cobs_frame.py       # COBS + CRC16 para el serial del ESP32
├── net/
│   ├── udp_server.py       # Recibe comandos, envía ACKs
│   ├── ws_server.py        # Publica telemetría a 50Hz (WS 8765)
│   ├── nav_server.py       # Goals/pose/estado de navegación (WS 8766)
│   └── video_pipeline.py   # GStreamer IMX219 → H264 HW → UDP
├── controller/
│   ├── planner.py          # A* sobre rejilla de ocupación inflada
│   └── controller.py       # Pure pursuit + ciclo de vida del goal
├── hw/
│   └── esp32_link.py       # Serial bidireccional con ESP32 (mixing v,w→ruedas)
└── scripts/
    └── jetson-service.service
```

## Requisitos runtime en la Jetson

- JetPack 4.6+ (con nvarguscamerasrc y nvv4l2h264enc)
- Python 3.6+
- `pip install -r requirements.txt` (pyserial, websockets 9.1, dataclasses)
- Chrony o ntpd configurado apuntando al PC host

## Ejecución manual

```bash
python3 main.py --map small --start-x 0.0 --start-y 0.0 --start-yaw 0.0
```
