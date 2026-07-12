"""Servidor WebSocket de navegación (reemplaza a gui_bridge_node de ROS2).

Mismo puerto (8766) y mismo protocolo JSON que el nodo ROS, así el host
(network/nav_client.py) no necesita ningún cambio:

  Jetson -> GUI: {"type":"pose","x":..,"y":..,"yaw":..,"valid":bool,"stamp":..}
                 {"type":"nav_status","state":"accepted|rejected|active|
                    succeeded|aborted|canceled","distance_remaining":..}
                 {"type":"map_name","name":"small"}      (al conectar)
  GUI -> Jetson: {"type":"goal","x":..,"y":..,"yaw":..}
                 {"type":"cancel"}

La pose sale de core/odometry.py (state.pose_*, reexpresando en el frame del
mapa la odometría on-board del ESP32) a nav.pose_publish_hz; los
goals/cancel se convierten en Ev.NAV_GOAL / Ev.NAV_CANCEL que consume
controller/controller.py, y los Ev.NAV_STATUS de éste se difunden aquí.
"""
# PY36: sin `from __future__ import annotations`; genéricos desde typing.
import asyncio
import json
import logging

from typing import Set  # PY36

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:  # pragma: no cover
    websockets = None
    ConnectionClosed = Exception  # type: ignore

from config import CFG
from core.bus import Ev, bus
from core.state import state

log = logging.getLogger(__name__)


class NavServer:
    def __init__(self, loop):
        # type: (asyncio.AbstractEventLoop) -> None
        self._loop = loop
        self._clients = set()  # type: Set

    def attach(self):
        # type: () -> None
        bus.on(Ev.NAV_STATUS, self._on_nav_status)

    # ------------------------------------------------------------
    # Difusión
    # ------------------------------------------------------------
    def _on_nav_status(self, data):
        # type: (dict) -> None
        if not isinstance(data, dict):
            return
        msg = {"type": "nav_status"}
        msg.update(data)
        self._broadcast_threadunsafe(json.dumps(msg))

    def _broadcast_threadunsafe(self, payload):
        # type: (str) -> None
        # Se llama siempre desde el loop asyncio (bus.emit corre en el loop).
        for ws in list(self._clients):
            asyncio.ensure_future(self._safe_send(ws, payload), loop=self._loop)

    @staticmethod
    async def _safe_send(ws, payload):
        try:
            await ws.send(payload)
        except (ConnectionClosed, OSError):
            pass
        except Exception:
            log.exception("Error enviando por WS nav")

    # ------------------------------------------------------------
    # Publicación periódica de pose
    # ------------------------------------------------------------
    async def pose_loop(self, stop_event):
        # type: (asyncio.Event) -> None
        period = 1.0 / CFG.nav.pose_publish_hz
        while not stop_event.is_set():
            try:
                await asyncio.sleep(period)
            except asyncio.CancelledError:
                return
            if not self._clients:
                continue
            if state.pose_valid:
                msg = {
                    "type": "pose",
                    "x": round(state.pose_x, 4),
                    "y": round(state.pose_y, 4),
                    "yaw": round(state.pose_yaw, 4),
                    "valid": True,
                    "stamp": state.pose_stamp,
                }
            else:
                # Igual que gui_bridge_node cuando aún no hay TF map->base_link.
                msg = {"type": "pose", "valid": False}
            self._broadcast_threadunsafe(json.dumps(msg))

    # ------------------------------------------------------------
    # Handler de clientes
    # ------------------------------------------------------------
    # PY36: websockets 9.x pasa (ws, path); en 10+ sólo (ws). path=None cubre ambos.
    async def handler(self, ws, path=None):
        self._clients.add(ws)
        log.info("Cliente nav conectado: %s (total=%d)", ws.remote_address, len(self._clients))
        # Anunciar el mapa activo para que el host lo auto-seleccione.
        await self._safe_send(ws, json.dumps({"type": "map_name", "name": CFG.nav.map_name}))
        try:
            async for msg in ws:
                self._on_message(msg)
        except ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)
            log.info("Cliente nav desconectado (total=%d)", len(self._clients))

    def _on_message(self, msg):
        # type: (object) -> None
        try:
            if isinstance(msg, bytes):
                msg = msg.decode("utf-8")
            data = json.loads(msg)
        except (UnicodeDecodeError, ValueError):
            return
        if not isinstance(data, dict):
            return
        mtype = data.get("type")
        if mtype == "goal":
            try:
                goal = {
                    "x": float(data["x"]),
                    "y": float(data["y"]),
                    "yaw": float(data.get("yaw", 0.0)),
                }
            except (KeyError, TypeError, ValueError):
                return
            log.info("Goal del host: x=%.2f y=%.2f yaw=%.2f",
                     goal["x"], goal["y"], goal["yaw"])
            bus.emit(Ev.NAV_GOAL, goal)
        elif mtype == "cancel":
            log.info("Cancel del host")
            bus.emit(Ev.NAV_CANCEL, None)


async def run_nav_server(stop_event, loop):
    # type: (asyncio.Event, asyncio.AbstractEventLoop) -> None
    if websockets is None:
        log.error("websockets no está instalado; servidor nav no arranca")
        await stop_event.wait()
        return

    server = NavServer(loop)
    server.attach()
    pose_task = loop.create_task(server.pose_loop(stop_event))

    try:
        async with websockets.serve(
            server.handler,
            CFG.network.listen_host,
            CFG.network.nav_ws_port,
            ping_interval=5,
            ping_timeout=5,
        ):
            log.info("WS navegación en ws://%s:%d",
                     CFG.network.listen_host, CFG.network.nav_ws_port)
            await stop_event.wait()
    finally:
        pose_task.cancel()
        await asyncio.gather(pose_task, return_exceptions=True)
