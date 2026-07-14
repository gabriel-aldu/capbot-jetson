"""Hilo de inferencia de la DNN sobre los frames de la cámara CSI.

La cámara la posee la pipeline de GStreamer de net/video_pipeline.py; cuando
CFG.perception.enabled, esa pipeline agrega un `tee` con una rama de análisis
que termina en un appsink (BGR reducido). La pipeline registra/desregistra el
appsink en el singleton `frames` de este módulo; el hilo de inferencia
(`run_perception` lo lanza vía run_in_executor) hace en cada iteración:

  1. try-pull-sample del appsink (frame BGR más reciente; drop=true descarta
     los viejos, así el ritmo lo pone infer_max_hz y no la cámara).
  2. YoloV8TRT.infer (TensorRT, contexto CUDA propio de ESTE hilo).
  3. Por cada detección, el borde inferior-centro de la caja se proyecta al
     piso (GroundPlaneMapper) -> (adelante, lateral) desde la cámara, se
     traslada al centro de rotación (cam_forward/lateral_offset) y se rota a
     coordenadas del mapa con el snapshot de pose odométrica.
  4. Emite Ev.DETECTIONS en el loop asyncio (call_soon_threadsafe), SIEMPRE
     — también con 0 detecciones: la ausencia sostenida es lo que permite a
     controller/obstacle_tracker.py liberar celdas.

Si TensorRT/pycuda/numpy/cv2 no están disponibles (PC de desarrollo) o el
engine no existe, la tarea queda dormida esperando el stop_event y el resto
del servicio funciona igual.
"""
# PY36: sin `from __future__ import annotations`; genéricos desde typing.
import asyncio
import logging
import math
import threading
import time

from typing import Optional  # PY36

try:
    import numpy as np
    import cv2
    import tensorrt  # noqa: F401  (sólo para detectar disponibilidad)
    import pycuda.driver as cuda
    _TRT_AVAILABLE = True
    _TRT_IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover
    np = cv2 = cuda = None
    _TRT_AVAILABLE = False
    _TRT_IMPORT_ERROR = str(exc)

try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    _GST_AVAILABLE = True
except (ImportError, ValueError):  # pragma: no cover
    Gst = None
    _GST_AVAILABLE = False

import os

from config import CFG
from core.bus import Ev, bus
from core.state import state
from perception.ground_plane import GroundPlaneMapper

log = logging.getLogger(__name__)


class _FrameSource(object):
    """Registro thread-safe del appsink activo (lo publica video_pipeline)."""

    def __init__(self):
        self._appsink = None
        self._lock = threading.Lock()

    def set_appsink(self, appsink):
        # type: (object) -> None
        with self._lock:
            self._appsink = appsink

    def get_appsink(self):
        # type: () -> object
        with self._lock:
            return self._appsink


# Singleton: net/video_pipeline.py lo alimenta; el hilo de inferencia lo lee.
frames = _FrameSource()


def _pull_frame(appsink):
    # type: (object) -> Optional[object]
    """Último frame BGR del appsink como ndarray (H, W, 3), o None."""
    # PY36/gi: emit("try-pull-sample") evita depender del binding GstApp.
    sample = appsink.emit("try-pull-sample", Gst.SECOND // 2)
    if sample is None:
        return None
    caps = sample.get_caps().get_structure(0)
    w = caps.get_value("width")
    h = caps.get_value("height")
    buf = sample.get_buffer()
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None
    try:
        data = mapinfo.data
        if len(data) < w * h * 3:
            return None
        # Copia: el buffer se desmapea al salir y la inferencia lo usa después.
        frame = np.frombuffer(data, dtype=np.uint8, count=w * h * 3)
        return frame.reshape(h, w, 3).copy()
    finally:
        buf.unmap(mapinfo)


def _project_detections(dets, mapper, frame_w, frame_h, pose):
    # type: (list, GroundPlaneMapper, int, int, Optional[tuple]) -> tuple
    """Detecciones -> (points, boxes).

    points: puntos en el frame del mapa (si hay pose válida y la detección
      proyecta al piso dentro de max_range_m); los consume
      controller/obstacle_tracker.py.
    boxes: TODAS las detecciones crudas para la GUI del host (caja
      normalizada 0..1 sobre el frame de análisis, clase, confianza y
      distancia si se pudo estimar), incluso las que no proyectan a un punto
      del mapa: el host las dibuja sobre el video."""
    p = CFG.perception
    points = []
    boxes = []
    for d in dets:
        # A float nativo YA: las cajas de yolov8_trt son escalares de numpy y
        # json.dumps no serializa numpy.float32/numpy.bool_ (el payload viaja
        # al host por el WS de navegación).
        x1, y1, x2, y2 = (float(v) for v in d["box"])
        # Fondo-centro de la caja = punto de contacto con el piso asumido. Si
        # la caja toca el borde inferior, el contacto quedó fuera de cuadro y
        # la distancia es sólo una cota superior (el objeto está MÁS cerca).
        clipped = bool(y2 >= frame_h - 3)
        pos = mapper.locate((x1 + x2) / 2.0, y2)
        dist = None if pos is None else math.hypot(pos[0], pos[1])

        # Caja normalizada (0..1): independiente de la resolución de la rama
        # de análisis; el host la escala a su frame de video (mismo aspecto).
        boxes.append({
            "box": [round(x1 / frame_w, 4), round(y1 / frame_h, 4),
                    round(x2 / frame_w, 4), round(y2 / frame_h, 4)],
            "cls": int(d["cls"]),
            "conf": round(float(d["conf"]), 3),
            "clipped": clipped,
            "dist_m": None if dist is None else round(dist, 3),
        })

        if pos is None or dist > p.max_range_m or pose is None:
            continue
        fwd, lat = pos
        px, py, pyaw = pose
        # Frame del robot: x adelante, y izquierda. `lat` es + derecha.
        xr = p.cam_forward_offset_m + fwd
        yr = p.cam_lateral_offset_m - lat
        cos_y, sin_y = math.cos(pyaw), math.sin(pyaw)
        points.append({
            "x": px + xr * cos_y - yr * sin_y,
            "y": py + xr * sin_y + yr * cos_y,
            "conf": round(float(d["conf"]), 3),
            "cls": int(d["cls"]),
            "clipped": clipped,
            "dist_m": round(dist, 3),
        })
    return points, boxes


def _perception_worker(thread_stop, loop):
    # type: (threading.Event, asyncio.AbstractEventLoop) -> None
    """Cuerpo del hilo de inferencia (contexto CUDA propio de este hilo)."""
    from perception.yolov8_trt import YoloV8TRT

    p = CFG.perception
    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()
    try:
        det = YoloV8TRT(p.engine_path, imgsz=p.imgsz,
                        conf_th=p.conf_threshold, iou_th=p.iou_threshold)
        log.info("Engine TensorRT cargado: %s (imgsz=%d conf=%.2f)",
                 p.engine_path, p.imgsz, p.conf_threshold)
        state.perception_state = "running"

        mapper = None      # se construye con las dimensiones del primer frame
        period = 1.0 / max(0.5, p.infer_max_hz)
        t_prev = time.time()

        while not thread_stop.is_set():
            appsink = frames.get_appsink()
            if appsink is None:
                # Pipeline de video caída o aún sin arrancar.
                time.sleep(0.2)
                continue
            try:
                frame = _pull_frame(appsink)
            except Exception:
                # El appsink puede morir en medio de un rebuild de la pipeline.
                log.debug("try-pull-sample falló (¿pipeline reconstruyéndose?)",
                          exc_info=True)
                time.sleep(0.2)
                continue
            if frame is None:
                continue

            if mapper is None or mapper.cx * 2 != frame.shape[1]:
                mapper = GroundPlaneMapper(
                    frame.shape[1], frame.shape[0],
                    height_m=p.cam_height_m,
                    pitch_deg=p.cam_pitch_deg,
                    hfov_deg=p.cam_hfov_deg,
                    min_ground_m=p.cam_min_ground_m,
                )
                near = mapper.locate(frame.shape[1] / 2.0, frame.shape[0])
                log.info("Plano de piso: pitch %.1f° abajo, piso visible más "
                         "cercano %s m",
                         math.degrees(mapper.pitch),
                         "inf" if near is None else "%.2f" % near[0])

            t0 = time.time()
            dets = det.infer(frame)
            fps = 1.0 / max(t0 - t_prev, 1e-6)
            t_prev = t0
            state.perception_fps = round(fps, 1)

            # Snapshot de pose lo más cerca posible de la captura (el robot
            # va a <=0.1 m/s: el desfase de un frame es milimétrico).
            pose = None
            if state.pose_valid:
                pose = (state.pose_x, state.pose_y, state.pose_yaw)

            points, boxes = _project_detections(
                dets, mapper, frame.shape[1], frame.shape[0], pose)
            payload = {
                "stamp": t0,
                "fps": round(fps, 1),
                "pose": None if pose is None else
                        {"x": pose[0], "y": pose[1], "yaw": pose[2]},
                "points": points,
                "boxes": boxes,
            }
            loop.call_soon_threadsafe(bus.emit, Ev.DETECTIONS, payload)

            # Ritmo: dormir lo que falte del período (la inferencia ya tomó
            # parte; el appsink con drop=true entrega siempre el más nuevo).
            elapsed = time.time() - t0
            if elapsed < period:
                # Dormir en tramos cortos para reaccionar rápido al stop.
                end = t0 + period
                while not thread_stop.is_set() and time.time() < end:
                    time.sleep(min(0.05, max(0.0, end - time.time())))
    except Exception as exc:
        state.perception_state = "error"
        log.exception("Hilo de percepción terminó con error: %s", exc)
    finally:
        state.perception_state = "stopped"
        try:
            cuda_ctx.pop()
        except Exception:
            pass


async def run_perception(stop_event, loop):
    # type: (asyncio.Event, asyncio.AbstractEventLoop) -> None
    """Tarea de main.py: lanza el hilo de inferencia y lo detiene al parar.

    Si la percepción no puede correr (deshabilitada, sin TensorRT, sin engine)
    espera el stop_event para no tumbar el servicio (asyncio.wait usa
    FIRST_COMPLETED en main.py)."""
    p = CFG.perception
    if not p.enabled:
        log.info("Percepción deshabilitada por configuración")
        await stop_event.wait()
        return
    if not _TRT_AVAILABLE:
        log.warning("Percepción deshabilitada: TensorRT/pycuda/numpy/cv2 no "
                    "disponibles (%s)", _TRT_IMPORT_ERROR)
        await stop_event.wait()
        return
    if not _GST_AVAILABLE:
        log.warning("Percepción deshabilitada: GStreamer no disponible")
        await stop_event.wait()
        return
    if not os.path.isfile(p.engine_path):
        state.perception_state = "error"
        log.error("Percepción deshabilitada: engine no encontrado en %s "
                  "(copiar bottles_fp16.engine de capbot-identification-test)",
                  p.engine_path)
        await stop_event.wait()
        return

    thread_stop = threading.Event()
    worker = loop.run_in_executor(None, _perception_worker, thread_stop, loop)
    log.info("Hilo de percepción lanzado (max %.1f Hz)", p.infer_max_hz)
    try:
        await stop_event.wait()
    finally:
        thread_stop.set()
        try:
            await worker
        except Exception:
            log.exception("Error esperando el hilo de percepción")
