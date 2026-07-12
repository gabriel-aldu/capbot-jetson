#!/usr/bin/env python3
"""
test_localizer.py
------------------
Script de prueba para localizer_standalone.py: imprime en terminal, por
cada marcador ArUco detectado, su ID y la pose (x, y, yaw) del robot que
ESE marcador por si solo implica (sin fusionar con los demas). Sirve para
validar visualmente, marcador por marcador, que las posiciones en el
markers-db y la calibracion de camara son correctas antes de confiar en
la fusion multi-marcador de localizer_standalone.py.

Usa las posiciones de markers_db_maze.yaml (copia local de
capbot-ros-foxy/src/test_bot/config/markers_db_maze.yaml, con
aruco_dict cambiado de DICT_5X5_250 a DICT_4X4_250).

Uso:
  python3 test_localizer.py \\
      --camera-info ./imx219_camera_info.yaml \\
      --markers-db ./markers_db_maze.yaml \\
      --show

  # Con extrinsics camara->base_link (si no se pasa, se reporta pose de
  # la CAMARA, no de base_link):
  python3 test_localizer.py \\
      --camera-info ./imx219_camera_info.yaml \\
      --markers-db ./markers_db_maze.yaml \\
      --extrinsics ./cam_to_base.yaml --show

  # Debug offline con un video:
  python3 test_localizer.py \\
      --camera-info ./imx219_camera_info.yaml \\
      --markers-db ./markers_db_maze.yaml \\
      --video-source /path/to/test.mp4 --show
"""

import argparse
import sys
import time

import cv2
import numpy as np

from localizer_standalone import (
    ArucoLocalizer,
    load_camera_info,
    load_extrinsics,
    load_markers_db,
    open_video_source,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-info", required=True, help="YAML de calibracion")
    ap.add_argument("--markers-db", required=True, help="YAML con posiciones de marcadores")
    ap.add_argument("--extrinsics", default=None,
                    help="YAML con T_cam_base. Si se omite => identidad (pose de la camara)")

    # Fuente de video (mismos defaults que localizer_standalone.py)
    ap.add_argument("--sensor-id", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--flip", type=int, default=0)
    ap.add_argument("--video-source", default=None,
                    help="fuente alternativa (int para webcam, path para video)")

    # Robustez (mismos defaults que el original)
    ap.add_argument("--max-distance", type=float, default=1.5)
    ap.add_argument("--max-reproj-error-px", type=float, default=3.0)
    ap.add_argument("--min-marker-area-px", type=float, default=150.0)
    ap.add_argument("--ambiguity-ratio-threshold", type=float, default=1.5)
    ap.add_argument("--no-marker-fix", action="store_true")

    # Salida
    ap.add_argument("--show", action="store_true", help="ventana con overlay")
    ap.add_argument("--print-hz", type=float, default=5.0,
                    help="rate maximo de impresion por consola (Hz)")
    args = ap.parse_args()

    K, D, w_cal, h_cal = load_camera_info(args.camera_info)
    print(f"[test] K/D cargadas ({w_cal}x{h_cal})")

    T_map_marker, marker_size, dict_name = load_markers_db(args.markers_db)
    print(f"[test] markers: {len(T_map_marker)} | dict={dict_name} | size={marker_size}m")

    T_cam_base = load_extrinsics(args.extrinsics)
    if args.extrinsics is None:
        print("[test] sin extrinsics => reportando pose de la CAMARA en map")
    else:
        print(f"[test] extrinsics cargadas desde {args.extrinsics}")

    if (args.width, args.height) != (w_cal, h_cal) and args.video_source is None:
        print(f"AVISO: capturando a {args.width}x{args.height} pero calibrado a "
              f"{w_cal}x{h_cal}. K y D quedan mal escaladas.")

    # filter_window=1: sin filtro temporal, se quiere ver la estimacion
    # cruda por marcador en cada frame, no una media movil.
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
        filter_window=1,
        apply_marker_fix=not args.no_marker_fix,
    )

    cap = open_video_source(args)
    if not cap.isOpened():
        print("ERROR: no pude abrir la fuente de video.", file=sys.stderr)
        sys.exit(1)

    print_period = 1.0 / max(args.print_hz, 0.01)
    last_print = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.video_source is not None:
                    print("[test] fin del video")
                    break
                continue

            t_now = time.time()
            res = localizer.process_frame(frame)

            if (t_now - last_print) >= print_period:
                last_print = t_now
                if res["per_marker"]:
                    for m in res["per_marker"]:
                        amb = " AMB" if m["ambiguous"] else ""
                        print(f"id={m['id']:>2}  x={m['x']:+.3f} m  y={m['y']:+.3f} m  "
                              f"yaw={np.degrees(m['yaw']):+7.2f} deg  "
                              f"err={m['reproj_err_px']:.2f}px{amb}")
                elif res["diagnostics"]:
                    print("[no pose] " + " | ".join(res["diagnostics"]))

            if args.show:
                vis = frame.copy()
                if res["overlay_ids"] is not None:
                    cv2.aruco.drawDetectedMarkers(
                        vis, res["overlay_corners"], res["overlay_ids"]
                    )
                cv2.imshow("test_localizer (ESC salir)", vis)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

    except KeyboardInterrupt:
        print("\n[test] interrumpido")
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
