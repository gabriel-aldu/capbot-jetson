"""Pipeline GStreamer que captura de la cámara IMX219 y la envía por UDP/RTP.

En Jetson Nano (JetPack 4.x) la captura es por `nvarguscamerasrc` y el
encoder H.264 por hardware es `nvv4l2h264enc`.

Pipeline:
    nvarguscamerasrc sensor-id=0
      ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1
      ! nvvidconv
      ! video/x-raw(memory:NVMM),format=NV12
      ! nvv4l2h264enc insert-sps-pps=true iframeinterval=15 idrinterval=15
                       maxperf-enable=1 preset-level=1 bitrate=4000000 control-rate=1
      ! h264parse config-interval=1
      ! rtph264pay pt=96 config-interval=1 mtu=1400
      ! udpsink host=<HOST> port=5000 sync=false async=false
"""
# PY36: Eliminado `from __future__ import annotations`.
import asyncio
import logging
import time
from typing import Optional

try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib
    _GST_AVAILABLE = True
except (ImportError, ValueError):  # pragma: no cover
    Gst = GLib = None
    _GST_AVAILABLE = False

import numpy as np

from aruco.localizer_standalone import (
    ArucoLocalizer,
    load_camera_info,
    load_extrinsics,
    load_markers_db,
)
from config import CFG
from core.bus import Ev, bus
from core.state import state

log = logging.getLogger(__name__)


def _build_pipeline_str(host_ip: str) -> str:
    v = CFG.video
    src = (
        "nvarguscamerasrc sensor-id=0 "
        "! video/x-raw(memory:NVMM),width={w},height={h},"
        "framerate={f}/1,format=NV12"
    ).format(w=v.width, h=v.height, f=v.fps)

    video_branch = (
        "nvvidconv "
        "! video/x-raw(memory:NVMM),format=NV12 "
        "! nvv4l2h264enc "
        "insert-sps-pps=true iframeinterval={iv} "
        "idrinterval={iv} maxperf-enable=1 preset-level=1 "
        "bitrate={br} control-rate=1 "
        "! h264parse config-interval=1 "
        "! rtph264pay pt=96 config-interval=1 mtu=1400 "
        "! udpsink host={host} port={port} sync=false async=false"
    ).format(
        iv=v.iframe_interval,
        br=v.bitrate_kbps * 1000,
        host=host_ip,
        port=CFG.network.video_port,
    )
    # PY36: f-strings sí funcionan en 3.6 (PEP 498), pero preferí `.format()`
    #       aquí porque el original encadenaba varias f-strings anidadas con
    #       parámetros calculados (`v.bitrate_kbps * 1000`) y así queda más
    #       claro lo que se inyecta. Es estilo, no obligatorio.

    if not CFG.aruco.enabled:
        return src + " ! " + video_branch

    # Rama ArUco: mismo truco que csi_camera_node.py (capbot-ros-foxy) para
    # compartir la unica sesion de captura nvarguscamerasrc entre el stream
    # H264 al host y los frames crudos que necesita cv2.aruco en Python.
    aruco_branch = (
        "nvvidconv flip-method={flip} "
        "! video/x-raw,width={w},height={h},format=BGRx "
        "! videoconvert ! video/x-raw,format=BGR "
        "! appsink name=aruco_sink emit-signals=true max-buffers=1 drop=true sync=false"
    ).format(flip=CFG.aruco.flip_method, w=v.width, h=v.height)

    # leaky=2 (downstream): descarta frames viejos en vez de bufferear, asi
    # ninguna rama atrasa a la otra.
    q = "queue leaky=2 max-size-buffers=2 max-size-bytes=0 max-size-time=0"

    return (
        src + " ! tee name=t "
        "t. ! " + q + " ! " + video_branch + " "
        "t. ! " + q + " ! " + aruco_branch
    )


class VideoPipeline:
    """Wraps la pipeline GStreamer y su MainLoop GLib."""

    # PY36: `loop` explícito por el mismo motivo que en Esp32Link: necesitamos
    #       `run_in_executor` y no hay `get_running_loop()` en 3.6.
    def __init__(self, loop=None):  # PY36: añadido loop
        self._loop = loop or asyncio.get_event_loop()  # PY36: añadido
        self._pipeline = None
        # PY36: `Optional[GLib.MainLoop]` preservado.
        self._glib_loop = None  # type: Optional["GLib.MainLoop"]
        # PY36: Se cambia anotación `Optional[asyncio.Task]` a simple None,
        #       porque `run_in_executor` devuelve un `Future`, no un `Task`
        #       (esto ya era un bug menor en el original).
        self._glib_thread = None
        self._current_host = ""  # type: str

        # ArUco: se construye una sola vez (independiente del ciclo start/
        # stop/retarget de la pipeline, que solo depende del host_ip).
        self._localizer = self._build_localizer() if CFG.aruco.enabled else None
        self._aruco_last_t = 0.0

    @staticmethod
    def _build_localizer():
        # type: () -> Optional[ArucoLocalizer]
        try:
            K, D, w_cal, h_cal = load_camera_info(CFG.aruco.camera_info_path)
            markers_db, marker_size, dict_name = load_markers_db(CFG.aruco.markers_db_path)
            T_cam_base = load_extrinsics(CFG.aruco.extrinsics_path)
        except (OSError, KeyError, ValueError) as exc:
            log.warning("ArUco deshabilitado: no se pudo cargar calibracion (%s)", exc)
            return None

        if (CFG.video.width, CFG.video.height) != (w_cal, h_cal):
            log.warning(
                "ArUco: capturando a %dx%d pero calibrado a %dx%d; K/D quedan mal escaladas",
                CFG.video.width, CFG.video.height, w_cal, h_cal,
            )

        return ArucoLocalizer(
            K=K, D=D, markers_db=markers_db, marker_size=marker_size,
            aruco_dict_name=dict_name, T_cam_base=T_cam_base,
            max_distance=CFG.aruco.max_distance,
            max_reproj_error_px=CFG.aruco.max_reproj_error_px,
            min_marker_area_px=CFG.aruco.min_marker_area_px,
            ambiguity_ratio_threshold=CFG.aruco.ambiguity_ratio_threshold,
            filter_window=CFG.aruco.filter_window,
        )

    async def start(self, host_ip: str) -> None:
        if not _GST_AVAILABLE:
            state.video_state = "error"
            bus.emit(Ev.VIDEO_STATE, "error: GStreamer no disponible")
            log.error("GStreamer no disponible; pipeline no arranca")
            return

        if not host_ip:
            log.warning("No hay host_ip todavía; pipeline de video esperará")
            return

        if self._pipeline is not None:
            if host_ip == self._current_host:
                return
            await self.stop()

        Gst.init(None)
        pipeline_str = _build_pipeline_str(host_ip)
        log.info("Pipeline: %s", pipeline_str)

        try:
            self._pipeline = Gst.parse_launch(pipeline_str)
        except GLib.Error as exc:
            state.video_state = "error"
            # PY36: `.format()` para evitar f-string con expresión tras colon.
            bus.emit(Ev.VIDEO_STATE, "parse_launch: {}".format(exc))
            log.exception("parse_launch falló")
            return

        gbus = self._pipeline.get_bus()
        gbus.add_signal_watch()
        gbus.connect("message::error", self._on_error)
        gbus.connect("message::eos", self._on_eos)

        if self._localizer is not None:
            aruco_sink = self._pipeline.get_by_name("aruco_sink")
            if aruco_sink is not None:
                aruco_sink.connect("new-sample", self._on_aruco_sample)

        self._pipeline.set_state(Gst.State.PLAYING)
        self._current_host = host_ip
        state.video_state = "running"
        bus.emit(Ev.VIDEO_STATE, "running -> {}:{}".format(host_ip, CFG.network.video_port))

        # PY36: `loop = asyncio.get_running_loop()` no existe en 3.6.
        #       Usamos el loop guardado en __init__.
        loop = self._loop
        self._glib_loop = GLib.MainLoop()
        self._glib_thread = loop.run_in_executor(None, self._glib_loop.run)

    async def stop(self) -> None:
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
        if self._glib_loop and self._glib_loop.is_running():
            self._glib_loop.quit()
        if self._glib_thread:
            try:
                await self._glib_thread
            except Exception:
                pass
            self._glib_thread = None
        self._current_host = ""
        state.video_state = "stopped"
        bus.emit(Ev.VIDEO_STATE, "stopped")

    async def retarget(self, host_ip: str) -> None:
        """Cambia el destino de la pipeline (reconstruye)."""
        if host_ip == self._current_host:
            return
        log.info("Retargeteando video a %s", host_ip)
        await self.stop()
        await self.start(host_ip)

    def _on_error(self, _bus, msg) -> None:
        err, dbg = msg.parse_error()
        state.video_state = "error"
        bus.emit(Ev.VIDEO_STATE, "error: {}".format(err.message))
        log.error("GStreamer error: %s (%s)", err.message, dbg)

    def _on_eos(self, _bus, _msg) -> None:
        state.video_state = "eos"
        bus.emit(Ev.VIDEO_STATE, "eos")
        log.info("GStreamer EOS")

    # ------------------------------------------------------------
    # Rama ArUco (corre en el hilo de GStreamer, NO en el loop asyncio)
    # ------------------------------------------------------------
    def _on_aruco_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        # Throttle: cv2.aruco es costoso en la Nano; se pulea el sample igual
        # (libera el buffer) pero se descarta si llego antes de tiempo. La
        # rama de video hacia el host no se ve afectada (es otra rama del tee).
        now = time.monotonic()
        period = (1.0 / CFG.aruco.process_rate_hz) if CFG.aruco.process_rate_hz > 0 else 0.0
        if period and (now - self._aruco_last_t) < period:
            return Gst.FlowReturn.OK
        self._aruco_last_t = now

        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        width = caps.get_value("width")
        height = caps.get_value("height")

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            # reshape defensivo: algunas resoluciones traen padding de fila
            # (stride > width*3).
            row_bytes = mapinfo.size // height
            frame = (np.frombuffer(mapinfo.data, dtype=np.uint8)
                      .reshape((height, row_bytes))[:, : width * 3]
                      .reshape((height, width, 3))
                      .copy())
        finally:
            buf.unmap(mapinfo)

        res = self._localizer.process_frame(frame)
        if res["x"] is not None:
            self._loop.call_soon_threadsafe(
                self._apply_aruco_pose, res["x"], res["y"], res["yaw"]
            )
        return Gst.FlowReturn.OK

    @staticmethod
    def _apply_aruco_pose(x, y, yaw):
        # type: (float, float, float) -> None
        # Sobreescritura directa, sin fusion con la odometria de ruedas: sirve
        # solo para validar la estimacion de ArUco de punta a punta.
        state.pose_x = x
        state.pose_y = y
        state.pose_yaw = yaw
        state.pose_stamp = time.time()
        state.pose_valid = True


# PY36: El original no recibía `loop`. Lo añadimos para poder pasarlo al
#       `VideoPipeline` (que lo necesita para `run_in_executor`).
async def run_video_pipeline(stop_event: asyncio.Event,
                             loop: asyncio.AbstractEventLoop) -> None:
    """Gestiona la pipeline en función del host detectado."""
    pipe = VideoPipeline(loop=loop)  # PY36: loop propagado

    # Si al arrancar ya tenemos host (por CLI), lanzamos inmediatamente.
    if CFG.network.host_ip:
        await pipe.start(CFG.network.host_ip)

    async def on_host_online(ip: str) -> None:
        await pipe.retarget(ip)

    async def on_host_offline(_data) -> None:
        pass

    bus.on(Ev.HOST_ONLINE, on_host_online)
    bus.on(Ev.HOST_OFFLINE, on_host_offline)

    try:
        await stop_event.wait()
    finally:
        await pipe.stop()