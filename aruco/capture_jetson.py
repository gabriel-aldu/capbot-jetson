#!/usr/bin/env python3
"""
capture_stream.py
-----------------
Captura frames de un tablero ChArUco desde la IMX219, los transmite
via UDP (GStreamer) a un PC, y permite capturar frames presionando
teclas en la terminal SSH.
"""

import argparse
import sys
import select
import termios
import tty
from pathlib import Path
import cv2
import numpy as np

SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH_M = 0.020
MARKER_LENGTH_M = 0.015
ARUCO_DICT_NAME = "DICT_4X4_250"

def build_board_and_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))
    if hasattr(cv2.aruco, "CharucoBoard") and callable(cv2.aruco.CharucoBoard):
        try:
            board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH_M, MARKER_LENGTH_M, aruco_dict)
        except TypeError:
            board = cv2.aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_LENGTH_M, MARKER_LENGTH_M, aruco_dict)
    else:
        board = cv2.aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_LENGTH_M, MARKER_LENGTH_M, aruco_dict)

    charuco_detector = None
    if hasattr(cv2.aruco, "CharucoDetector"):
        try:
            charuco_detector = cv2.aruco.CharucoDetector(board)
        except Exception:
            charuco_detector = None

    aruco_params = (cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters") else cv2.aruco.DetectorParameters_create())
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    return board, aruco_dict, charuco_detector, aruco_params

def detect_charuco(gray, board, aruco_dict, charuco_detector, aruco_params):
    if charuco_detector is not None:
        ch_corners, ch_ids, _, _ = charuco_detector.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < 4:
            return None, None
        return ch_corners, ch_ids

    m_corners, m_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    if m_ids is None or len(m_ids) == 0:
        return None, None
    retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(m_corners, m_ids, gray, board)
    if retval is None or ch_ids is None or len(ch_ids) < 4:
        return None, None
    return ch_corners, ch_ids

def gst_pipeline_in(sensor_id, w, h, fps, flip):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){w}, height=(int){h}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width=(int){w}, height=(int){h}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1"
    )

def gst_pipeline_out(ip, port, w, h, fps):
    # Sends BGR frames -> BGRx -> NV12 -> H264 -> UDP payload
    return (
        f"appsrc ! videoconvert ! video/x-raw, format=(string)BGRx ! "
        f"nvvidconv ! nvv4l2h264enc insert-sps-pps=true ! "
        f"h264parse ! rtph264pay pt=96 ! "
        f"udpsink host={ip} port={port} sync=false"
    )

def is_data():
    """Non-blocking check for standard input."""
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="./calib_imgs")
    ap.add_argument("--sensor-id", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--flip", type=int, default=0)
    ap.add_argument("--min-corners", type=int, default=15)
    ap.add_argument("--auto-every", type=int, default=20)
    # Stream settings
    ap.add_argument("--stream-ip", required=True, help="IP del PC que va a recibir el video")
    ap.add_argument("--stream-port", type=int, default=5000, help="Puerto UDP para el stream")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    board, aruco_dict, charuco_detector, aruco_params = build_board_and_detector()

    pipeline_in = gst_pipeline_in(args.sensor_id, args.width, args.height, args.fps, args.flip)
    cap = cv2.VideoCapture(pipeline_in, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("ERROR: no pude abrir la camara.", file=sys.stderr)
        sys.exit(1)

    # Initialize VideoWriter for UDP output
    pipeline_out = gst_pipeline_out(args.stream_ip, args.stream_port, args.width, args.height, args.fps)
    writer = cv2.VideoWriter(pipeline_out, cv2.CAP_GSTREAMER, 0, float(args.fps), (args.width, args.height), True)
    
    if not writer.isOpened():
        print("ERROR: Fallo al abrir el streamer GStreamer.", file=sys.stderr)
        sys.exit(1)

    print(f"[capture] resolucion: {args.width}x{args.height}")
    print(f"[capture] Transmitiendo a: {args.stream_ip}:{args.stream_port}")
    print("[capture] SPACE=guardar | a=auto-toggle | ESC=salir (Presiona teclas en la terminal SSH)")

    idx = 0
    auto = False
    auto_counter = 0

    # Configure terminal to capture single keystrokes headlessly
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ch_corners, ch_ids = detect_charuco(gray, board, aruco_dict, charuco_detector, aruco_params)
            n = 0 if ch_ids is None else len(ch_ids)

            vis = frame.copy()
            if ch_ids is not None:
                cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids)

            color = (0, 255, 0) if n >= args.min_corners else (0, 165, 255)
            cv2.putText(vis,
                        f"guardadas: {idx} | esquinas: {n} | auto: {'ON' if auto else 'OFF'}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            # Send frame to PC
            writer.write(vis)

            # Check for terminal keypress
            save = False
            if is_data():
                key = sys.stdin.read(1)
                if key == '\x1b': # ESC
                    break
                elif key == 'a':
                    auto = not auto
                    auto_counter = 0
                elif key == ' ':
                    save = True

            if auto and n >= args.min_corners:
                auto_counter += 1
                if auto_counter >= args.auto_every:
                    save = True
                    auto_counter = 0

            if save:
                if n < args.min_corners:
                    sys.stdout.write(f"\r[capture] descartado ({n} < {args.min_corners})\n")
                else:
                    fname = out_dir / f"{idx:03d}.png"
                    cv2.imwrite(str(fname), frame)
                    sys.stdout.write(f"\r[capture] guardado {fname} ({n} esquinas)\n")
                    idx += 1

    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        cap.release()
        writer.release()
        print(f"\n[capture] total: {idx} imagenes en {out_dir}")

if __name__ == "__main__":
    main()