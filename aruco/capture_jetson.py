#!/usr/bin/env python3
"""
capture_stream.py
-----------------
Captura frames desde la IMX219, los transmite via UDP (GStreamer) a un
PC para verlos en vivo, y permite guardarlos presionando teclas en la
terminal SSH. La deteccion/calibracion ChArUco NO ocurre aqui: estas
imagenes se procesan despues con calibrate_pc.py en el PC.

Por que no usa cv2.aruco: en esta Jetson, el cv2 con soporte GStreamer
(necesario para nvarguscamerasrc) es el del sistema (apt, sin modulo
contrib/aruco); el cv2 con aruco es el de pip (~/.local, sin GStreamer).
No hay un solo cv2 con ambos, asi que este script fuerza el cv2 del
sistema (unico requisito real aqui) y deja la deteccion para el PC.
"""

import argparse
import sys
import select
import termios
import tty
from pathlib import Path

# Fuerza el cv2 del sistema (dist-packages, con GStreamer) por delante
# del cv2 de pip en ~/.local (sin GStreamer) que Python prioriza por
# defecto. Debe ir antes de "import cv2".
sys.path.insert(0, f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages")

import cv2
import numpy as np

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
    ap.add_argument("--auto-every", type=int, default=20)
    # Stream settings
    ap.add_argument("--stream-ip", required=True, help="IP del PC que va a recibir el video")
    ap.add_argument("--stream-port", type=int, default=5000, help="Puerto UDP para el stream")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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

            vis = frame.copy()
            color = (0, 255, 0) if auto else (0, 165, 255)
            cv2.putText(vis,
                        f"guardadas: {idx} | auto: {'ON' if auto else 'OFF'}",
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

            if auto:
                auto_counter += 1
                if auto_counter >= args.auto_every:
                    save = True
                    auto_counter = 0

            if save:
                fname = out_dir / f"{idx:03d}.png"
                cv2.imwrite(str(fname), frame)
                sys.stdout.write(f"\r[capture] guardado {fname}\n")
                idx += 1

    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        cap.release()
        writer.release()
        print(f"\n[capture] total: {idx} imagenes en {out_dir}")

if __name__ == "__main__":
    main()