#!/usr/bin/env python3
"""
localizer_standalone.py
-----------------------
Version standalone del aruco_localizer (sin ROS 2, sin tf2, sin rclpy).
Solo entrega estimacion de (x, y, yaw) del robot en el marco 'map' a partir
de marcadores ArUco con posiciones conocidas.

Mantiene toda la robustez del original:
  - solvePnPGeneric con IPPE_SQUARE (dos soluciones, se elige la de menor
    error de reproyeccion => resuelve el flip de pose plana).
  - Rechazo por area, distancia y error de reproyeccion.
  - Promedio ponderado (area / (1+err)) cuando hay varios marcadores.
  - Filtro temporal (media movil).
  - Aviso si la 2a solucion tiene error similar a la 1a (pose ambigua).

Uso:
  # En vivo con la IMX219 en la Jetson
  python3 localizer_standalone.py \\
      --camera-info ./imx219_camera_info.yaml \\
      --markers-db ./markers_db.yaml \\
      --extrinsics ./cam_to_base.yaml \\
      --show

  # Con un video de prueba (util para debug offline en PC)
  python3 localizer_standalone.py \\
      --camera-info ./imx219_camera_info.yaml \\
      --markers-db ./markers_db.yaml \\
      --video-source /path/to/test.mp4 --show

  # Salida a CSV
  python3 localizer_standalone.py ... --csv ./poses.csv

Formato markers_db.yaml:
  aruco_dict: DICT_4X4_250
  marker_size: 0.15            # lado del marcador ArUco en metros
  markers:
    - {id: 0, x: 0.0, y: 0.0, z: 0.5, roll: 0.0, pitch: 0.0, yaw: 0.0}
    - {id: 1, x: 2.0, y: 0.0, z: 0.5, roll: 0.0, pitch: 0.0, yaw: 1.5708}
    # ...

Formato cam_to_base.yaml (extrinsics camera_link_optical -> base_link):
  # Pose de base_link expresada en el marco de la camara (optical frame:
  # x-right, y-down, z-forward). Traslacion en metros, rotacion en radianes.
  translation: [0.0, 0.0, -0.10]     # base_link 10 cm debajo de la camara
  rotation_rpy: [0.0, 0.0, 0.0]

Si no pasas --extrinsics, se usa identidad => se reporta la pose de la camara
en map (no del base_link). Es un buen primer paso para validar detecciones.

Captura en vivo en la Jetson: el cv2 del sistema (con soporte GStreamer,
necesario para nvarguscamerasrc) no trae el modulo contrib (cv2.aruco); el
cv2 de pip que si trae cv2.aruco no trae GStreamer. Ningun cv2 instalado
tiene ambos a la vez. Como aqui la deteccion SI tiene que correr en la
Jetson (a diferencia de capture_jetson.py, que solo guarda fotos para
calibrar despues en el PC), la captura en vivo usa GStreamer/PyGObject
puro (clase GstCameraCapture) para obtener frames como numpy array, y el
cv2 de pip (con aruco) se usa solo para procesarlos en memoria. El modo
--video-source (debug offline en PC) no se ve afectado: sigue usando
cv2.VideoCapture normal, sin GStreamer.
"""

import argparse
import collections
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


# =============================================================================
# Helpers de deteccion y pose
# =============================================================================

def detect_markers(gray, aruco_dict, params):
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)


def estimate_marker_pose_dual(corners, marker_size, K, D):
    """Retorna hasta 2 (rvec, tvec, err_px) ordenadas por error ascendente."""
    half = marker_size / 2.0
    obj_pts = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)
    img_pts = corners.reshape(-1, 2).astype(np.float32)

    flag = (cv2.SOLVEPNP_IPPE_SQUARE
            if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
            else cv2.SOLVEPNP_ITERATIVE)

    if hasattr(cv2, "solvePnPGeneric"):
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj_pts, img_pts, K, D, flags=flag)
        if not ok or len(rvecs) == 0:
            return []
        results = [(r, t, _reproj_err(obj_pts, img_pts, r, t, K, D))
                   for r, t in zip(rvecs, tvecs)]
        results.sort(key=lambda x: x[2])
        return results

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D, flags=flag)
    if not ok:
        return []
    return [(rvec, tvec, _reproj_err(obj_pts, img_pts, rvec, tvec, K, D))]


def _reproj_err(obj_pts, img_pts, rvec, tvec, K, D):
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
    return float(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).mean())


def marker_area_px(corners):
    pts = corners.reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def to_matrix(translation, quat_xyzw):
    T = np.eye(4)
    T[:3, 3] = translation
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    return T


def average_poses(matrices, weights=None):
    if len(matrices) == 1:
        return matrices[0]
    if weights is None:
        weights = np.ones(len(matrices))
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    T_avg = np.eye(4)
    T_avg[:3, 3] = np.sum([w * T[:3, 3] for w, T in zip(weights, matrices)], axis=0)

    quats = np.array([R.from_matrix(T[:3, :3]).as_quat() for T in matrices])
    for i in range(1, len(quats)):
        if np.dot(quats[0], quats[i]) < 0:
            quats[i] = -quats[i]
    q = (weights[:, None] * quats).sum(axis=0)
    q /= np.linalg.norm(q)
    T_avg[:3, :3] = R.from_quat(q).as_matrix()
    return T_avg


class PoseFilter:
    def __init__(self, window_size):
        self.window = max(1, int(window_size))
        self.buffer = collections.deque(maxlen=self.window)

    def update(self, T):
        self.buffer.append(T)
        if len(self.buffer) == 1:
            return T
        return average_poses(list(self.buffer))


# =============================================================================
# Localizador
# =============================================================================

class ArucoLocalizer:
    def __init__(self, K, D, markers_db, marker_size, aruco_dict_name,
                 T_cam_base=np.eye(4),
                 max_distance=1.5, max_reproj_error_px=3.0,
                 min_marker_area_px=150.0, ambiguity_ratio_threshold=1.5,
                 filter_window=1, apply_marker_fix=True):
        self.K = K
        self.D = D
        self.marker_size = float(marker_size)
        self.T_cam_base = T_cam_base
        self.T_map_marker = markers_db

        self.max_distance = float(max_distance)
        self.max_reproj_error_px = float(max_reproj_error_px)
        self.min_marker_area_px = float(min_marker_area_px)
        self.ambiguity_ratio_threshold = float(ambiguity_ratio_threshold)
        self.pose_filter = PoseFilter(filter_window)

        # Fix de convencion de marcador ArUco (rot 90 deg Z). Del codigo original.
        if apply_marker_fix:
            self.T_marker_fix = np.eye(4)
            self.T_marker_fix[:3, :3] = R.from_euler(
                'xyz', [0, 0, np.pi / 2]
            ).as_matrix()
        else:
            self.T_marker_fix = np.eye(4)

        if not hasattr(cv2.aruco, aruco_dict_name):
            raise ValueError(f"Diccionario desconocido: {aruco_dict_name}")
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, aruco_dict_name)
        )
        self.aruco_params = (
            cv2.aruco.DetectorParameters()
            if hasattr(cv2.aruco, "DetectorParameters")
            else cv2.aruco.DetectorParameters_create()
        )
        if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
            self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    def process_frame(self, bgr):
        """
        Procesa un frame BGR y retorna dict con:
          x, y, yaw (float; None si no hubo estimacion valida)
          n_used, mean_reproj_error_px, diagnostics (list[str])
          overlay_corners, overlay_ids (para dibujar afuera, opcional)
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect_markers(gray, self.aruco_dict, self.aruco_params)

        result = {
            "x": None, "y": None, "yaw": None,
            "n_used": 0, "mean_reproj_error_px": None,
            "diagnostics": [],
            "per_marker": [],
            "overlay_corners": corners, "overlay_ids": ids,
        }
        if ids is None or len(ids) == 0:
            return result

        estimates = []
        for i, marker_id in enumerate(ids.flatten().tolist()):
            if marker_id not in self.T_map_marker:
                continue

            area = marker_area_px(corners[i])
            if area < self.min_marker_area_px:
                result["diagnostics"].append(
                    f"id={marker_id} REJ(area<{self.min_marker_area_px:.0f})"
                )
                continue

            sols = estimate_marker_pose_dual(corners[i], self.marker_size, self.K, self.D)
            if not sols:
                continue

            rvec, tvec, err = sols[0]
            dist = float(np.linalg.norm(tvec))
            if self.max_distance > 0 and dist > self.max_distance:
                result["diagnostics"].append(
                    f"id={marker_id} REJ(d={dist:.2f}>{self.max_distance})"
                )
                continue
            if err > self.max_reproj_error_px:
                result["diagnostics"].append(
                    f"id={marker_id} REJ(err={err:.2f}>{self.max_reproj_error_px}px)"
                )
                continue

            ambiguous = False
            if len(sols) > 1:
                err2 = sols[1][2]
                if err2 < err * self.ambiguity_ratio_threshold:
                    ambiguous = True

            T_cam_marker = np.eye(4)
            T_cam_marker[:3, :3] = R.from_rotvec(rvec.flatten()).as_matrix()
            T_cam_marker[:3, 3] = tvec.flatten()
            T_cam_marker = T_cam_marker @ self.T_marker_fix

            T_map_base = (
                self.T_map_marker[marker_id]
                @ np.linalg.inv(T_cam_marker)
                @ self.T_cam_base
            )

            weight = area / (1.0 + err)
            estimates.append((marker_id, T_map_base, weight, err))
            result["per_marker"].append({
                "id": marker_id,
                "x": float(T_map_base[0, 3]),
                "y": float(T_map_base[1, 3]),
                "yaw": float(R.from_matrix(T_map_base[:3, :3]).as_euler("xyz")[2]),
                "reproj_err_px": err,
                "ambiguous": ambiguous,
            })
            tag = "AMB" if ambiguous else "OK"
            result["diagnostics"].append(
                f"id={marker_id} {tag}(d={dist:.2f}m,err={err:.2f}px,"
                f"area={area:.0f}px2,w={weight:.0f})"
            )

        if not estimates:
            return result

        Ts = [e[1] for e in estimates]
        ws = [e[2] for e in estimates]
        T_combined = average_poses(Ts, ws)
        T_filtered = self.pose_filter.update(T_combined)

        x, y = float(T_filtered[0, 3]), float(T_filtered[1, 3])
        yaw = float(R.from_matrix(T_filtered[:3, :3]).as_euler("xyz")[2])
        result["x"], result["y"], result["yaw"] = x, y, yaw
        result["n_used"] = len(estimates)
        result["mean_reproj_error_px"] = float(np.mean([e[3] for e in estimates]))
        return result


# =============================================================================
# Carga de configuracion
# =============================================================================

def load_camera_info(path):
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    K = np.array(y["camera_matrix"]["data"]).reshape(3, 3)
    D = np.array(y["distortion_coefficients"]["data"])
    return K, D, int(y["image_width"]), int(y["image_height"])


def load_markers_db(path):
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    dict_name = y.get("aruco_dict", "DICT_4X4_250")
    marker_size = float(y.get("marker_size", 0.15))
    T_map_marker = {}
    for m in y["markers"]:
        T = np.eye(4)
        T[:3, 3] = [m["x"], m["y"], m["z"]]
        T[:3, :3] = R.from_euler("xyz", [m["roll"], m["pitch"], m["yaw"]]).as_matrix()
        T_map_marker[int(m["id"])] = T
    return T_map_marker, marker_size, dict_name


def load_extrinsics(path):
    """Retorna T_cam_base (4x4). Si path es None, identidad."""
    if path is None:
        return np.eye(4)
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    T = np.eye(4)
    T[:3, 3] = y.get("translation", [0.0, 0.0, 0.0])
    rpy = y.get("rotation_rpy", [0.0, 0.0, 0.0])
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    return T


# =============================================================================
# Fuentes de video
# =============================================================================

class GstCameraCapture:
    """Captura BGR desde nvarguscamerasrc via GStreamer/PyGObject puro.

    Expone isOpened()/read()/release() como cv2.VideoCapture para no
    tener que tocar el resto del script. Ver nota en el docstring del
    modulo sobre por que no se usa cv2.VideoCapture(..., CAP_GSTREAMER)
    aqui.
    """

    def __init__(self, sensor_id, width, height, fps, flip):
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        self._Gst = Gst

        Gst.init(None)
        self._width = width
        self._height = height

        pipeline_str = (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
            f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
            f"nvvidconv flip-method={flip} ! "
            f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
            f"videoconvert ! video/x-raw, format=(string)BGR ! "
            f"appsink name=sink emit-signals=false sync=false drop=true max-buffers=1"
        )
        print(f"[localizer] pipeline (GStreamer/PyGObject):\n  {pipeline_str}")

        self._pipeline = Gst.parse_launch(pipeline_str)
        self._sink = self._pipeline.get_by_name("sink")
        self._pipeline.set_state(Gst.State.PLAYING)
        state = self._pipeline.get_state(5 * Gst.SECOND)
        self._opened = state[0] == Gst.StateChangeReturn.SUCCESS
        if not self._opened:
            bus = self._pipeline.get_bus()
            msg = bus.timed_pop_filtered(0, Gst.MessageType.ERROR)
            if msg is not None:
                err, dbg = msg.parse_error()
                print(f"[localizer] GStreamer error: {err.message} ({dbg})", file=sys.stderr)

    def isOpened(self):
        return self._opened

    def read(self):
        Gst = self._Gst
        sample = self._sink.emit("pull-sample")
        if sample is None:
            return False, None
        buf = sample.get_buffer()
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return False, None
        try:
            # reshape defensivo: algunas resoluciones traen padding de fila
            # (stride > width*3), asi que se recorta antes de dar forma HxWx3.
            row_bytes = mapinfo.size // self._height
            frame = (np.frombuffer(mapinfo.data, dtype=np.uint8)
                      .reshape((self._height, row_bytes))[:, :self._width * 3]
                      .reshape((self._height, self._width, 3))
                      .copy())
        finally:
            buf.unmap(mapinfo)
        return True, frame

    def release(self):
        self._pipeline.set_state(self._Gst.State.NULL)


def open_video_source(args):
    if args.video_source is not None:
        try:
            src = int(args.video_source)
        except ValueError:
            src = args.video_source
        return cv2.VideoCapture(src)

    try:
        return GstCameraCapture(args.sensor_id, args.width, args.height, args.fps, args.flip)
    except ImportError:
        print("ERROR: falta PyGObject (python3-gi, gir1.2-gstreamer-1.0) "
              "para la captura en vivo por GStreamer.", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-info", required=True, help="YAML de calibracion")
    ap.add_argument("--markers-db", required=True, help="YAML con posiciones de marcadores")
    ap.add_argument("--extrinsics", default=None,
                    help="YAML con T_cam_base. Si se omite => identidad (pose de la camara)")

    # Fuente de video
    ap.add_argument("--sensor-id", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--flip", type=int, default=0)
    ap.add_argument("--video-source", default=None,
                    help="fuente alternativa (int para webcam, path para video)")

    # Robustez (mismos defaults que el original ROS)
    ap.add_argument("--max-distance", type=float, default=1.5)
    ap.add_argument("--max-reproj-error-px", type=float, default=3.0)
    ap.add_argument("--min-marker-area-px", type=float, default=150.0)
    ap.add_argument("--ambiguity-ratio-threshold", type=float, default=1.5)
    ap.add_argument("--filter-window", type=int, default=5)
    ap.add_argument("--no-marker-fix", action="store_true",
                    help="desactiva el fix de rotacion 90deg en Z sobre el marcador")

    # Salida
    ap.add_argument("--show", action="store_true", help="ventana con overlay")
    ap.add_argument("--csv", default=None, help="log a CSV")
    ap.add_argument("--print-hz", type=float, default=5.0,
                    help="rate maximo de impresion por consola (Hz)")
    args = ap.parse_args()

    K, D, w_cal, h_cal = load_camera_info(args.camera_info)
    print(f"[localizer] K/D cargadas ({w_cal}x{h_cal})")

    T_map_marker, marker_size, dict_name = load_markers_db(args.markers_db)
    print(f"[localizer] markers: {len(T_map_marker)} | dict={dict_name} "
          f"| size={marker_size}m")

    T_cam_base = load_extrinsics(args.extrinsics)
    if args.extrinsics is None:
        print("[localizer] sin extrinsics => reportando pose de la CAMARA en map")
    else:
        print(f"[localizer] extrinsics cargadas desde {args.extrinsics}")

    if (args.width, args.height) != (w_cal, h_cal) and args.video_source is None:
        print(f"AVISO: capturando a {args.width}x{args.height} pero calibrado a "
              f"{w_cal}x{h_cal}. K y D quedan mal escaladas.")

    localizer = ArucoLocalizer(
        K=K, D=D,
        markers_db=T_map_marker,
        marker_size=marker_size,
        aruco_dict_name=dict_name,
        T_cam_base=T_cam_base,
        max_distance=args.max_distance,
        max_reproj_error_px=args.max_reproj_error_px,
        min_marker_area_px=args.min_marker_area_px,
        ambiguity_ratio_threshold=args.ambiguity_ratio_threshold,
        filter_window=args.filter_window,
        apply_marker_fix=not args.no_marker_fix,
    )

    cap = open_video_source(args)
    if not cap.isOpened():
        print("ERROR: no pude abrir la fuente de video.", file=sys.stderr)
        sys.exit(1)

    csv_file = None
    csv_writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["t", "x", "y", "yaw_rad", "yaw_deg",
                             "n_used", "mean_reproj_error_px"])

    print_period = 1.0 / max(args.print_hz, 0.01)
    last_print = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.video_source is not None:
                    print("[localizer] fin del video")
                    break
                continue

            t_now = time.time()
            res = localizer.process_frame(frame)

            if csv_writer and res["x"] is not None:
                csv_writer.writerow([
                    f"{t_now:.6f}",
                    f"{res['x']:.4f}", f"{res['y']:.4f}",
                    f"{res['yaw']:.4f}", f"{np.degrees(res['yaw']):.2f}",
                    res["n_used"],
                    f"{res['mean_reproj_error_px']:.3f}",
                ])
                csv_file.flush()

            if (t_now - last_print) >= print_period:
                last_print = t_now
                if res["x"] is None:
                    if res["diagnostics"]:
                        print("[no pose] " + " | ".join(res["diagnostics"]))
                else:
                    print(f"x={res['x']:+.3f} m  y={res['y']:+.3f} m  "
                          f"yaw={np.degrees(res['yaw']):+7.2f} deg  "
                          f"n={res['n_used']}  "
                          f"reproj={res['mean_reproj_error_px']:.2f} px")

            if args.show:
                vis = frame.copy()
                if res["overlay_ids"] is not None:
                    cv2.aruco.drawDetectedMarkers(
                        vis, res["overlay_corners"], res["overlay_ids"]
                    )
                if res["x"] is not None:
                    txt = (f"x={res['x']:+.2f} y={res['y']:+.2f} "
                           f"yaw={np.degrees(res['yaw']):+.1f}")
                    cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("aruco localizer (ESC salir)", vis)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

    except KeyboardInterrupt:
        print("\n[localizer] interrumpido")
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()
            print(f"[localizer] CSV guardado: {args.csv}")


if __name__ == "__main__":
    main()
