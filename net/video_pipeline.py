"""Pipeline GStreamer que captura de la cámara IMX219 y la envía por UDP/RTP.

En Jetson Nano (JetPack 4.x) la captura es por `nvarguscamerasrc` y el
encoder H.264 por hardware es `nvv4l2h264enc`.

Pipeline (rama de video):
    nvarguscamerasrc sensor-id=0
      ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1
      ! nvvidconv
      ! video/x-raw(memory:NVMM),format=NV12
      ! nvv4l2h264enc insert-sps-pps=true iframeinterval=15 idrinterval=15
                       maxperf-enable=1 preset-level=1 bitrate=4000000 control-rate=1
      ! h264parse config-interval=1
      ! rtph264pay pt=96 config-interval=1 mtu=1400
      ! udpsink host=<HOST> port=5000 sync=false async=false

Con CFG.perception.enabled la cámara se COMPARTE con la DNN de obstáculos
mediante un `tee` (nvarguscamerasrc sólo admite UN proceso dueño de la CSI):
la rama de video anterior queda igual y una rama de análisis entrega frames
BGR reducidos a un appsink que consume perception/detector.py:

    tee name=t
    t. ! queue ! <rama de video de arriba>
    t. ! queue max-size-buffers=1 leaky=downstream
       ! nvvidconv ! video/x-raw,format=BGRx,width=640,height=360
       ! videoconvert ! video/x-raw,format=BGR
       ! appsink name=detsink drop=true max-buffers=1 sync=false

En ese modo la pipeline arranca AUNQUE no haya host todavía (sólo con la
rama de análisis) para que la detección corra siempre; al detectarse el host
se reconstruye con ambas ramas.
"""
# PY36: Eliminado `from __future__ import annotations`.
import asyncio
import logging
from typing import Optional

try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib
    _GST_AVAILABLE = True
except (ImportError, ValueError):  # pragma: no cover
    Gst = GLib = None
    _GST_AVAILABLE = False

from config import CFG
from core.bus import Ev, bus
from core.state import state
# Registro del appsink de la rama de análisis (la DNN lo consume desde su
# hilo). El import es liviano: detector.py protege sus imports pesados.
from perception.detector import frames as det_frames

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

    # Rama de análisis para la DNN (perception/detector.py): frames BGR
    # reducidos en memoria de sistema. leaky/drop: si la inferencia va lenta
    # se descartan frames viejos y JAMÁS se atasca la rama del encoder.
    det_branch = (
        "queue max-size-buffers=1 leaky=downstream "
        "! nvvidconv "
        "! video/x-raw,format=BGRx,width={w},height={h} "
        "! videoconvert ! video/x-raw,format=BGR "
        "! appsink name=detsink drop=true max-buffers=1 sync=false"
    ).format(w=CFG.perception.infer_width, h=CFG.perception.infer_height)

    branches = []
    if host_ip:
        branches.append("queue ! " + video_branch)
    if CFG.perception.enabled:
        branches.append(det_branch)

    if not CFG.perception.enabled:
        # Sin percepción: pipeline idéntica a la de siempre (sin tee).
        return src + " ! " + video_branch

    return src + " ! tee name=t " + " ".join("t. ! " + b for b in branches)


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

    async def start(self, host_ip: str) -> None:
        if not _GST_AVAILABLE:
            state.video_state = "error"
            bus.emit(Ev.VIDEO_STATE, "error: GStreamer no disponible")
            log.error("GStreamer no disponible; pipeline no arranca")
            return

        if not host_ip and not CFG.perception.enabled:
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

        self._pipeline.set_state(Gst.State.PLAYING)
        self._current_host = host_ip
        state.video_state = "running"
        if host_ip:
            bus.emit(Ev.VIDEO_STATE, "running -> {}:{}".format(host_ip, CFG.network.video_port))
        else:
            bus.emit(Ev.VIDEO_STATE, "running (sólo análisis, sin host)")

        # Publicar el appsink de la rama de análisis para el hilo de la DNN.
        appsink = self._pipeline.get_by_name("detsink")
        if appsink is not None:
            det_frames.set_appsink(appsink)

        # PY36: `loop = asyncio.get_running_loop()` no existe en 3.6.
        #       Usamos el loop guardado en __init__.
        loop = self._loop
        self._glib_loop = GLib.MainLoop()
        self._glib_thread = loop.run_in_executor(None, self._glib_loop.run)

    async def stop(self) -> None:
        # Retirar el appsink ANTES de bajar la pipeline: el hilo de la DNN
        # deja de hacer pull y espera al que publique el siguiente.
        det_frames.set_appsink(None)
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


# PY36: El original no recibía `loop`. Lo añadimos para poder pasarlo al
#       `VideoPipeline` (que lo necesita para `run_in_executor`).
async def run_video_pipeline(stop_event: asyncio.Event,
                             loop: asyncio.AbstractEventLoop) -> None:
    """Gestiona la pipeline en función del host detectado."""
    pipe = VideoPipeline(loop=loop)  # PY36: loop propagado

    # Si al arrancar ya tenemos host (por CLI), lanzamos inmediatamente.
    # Con percepción habilitada se arranca SIEMPRE (aunque no haya host):
    # la rama de análisis alimenta a la DNN desde el primer momento.
    if CFG.network.host_ip or CFG.perception.enabled:
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