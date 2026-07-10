"""Enlace serial standalone con el ESP32 para las herramientas de tuning de PID.

A diferencia de hw/esp32_link.py (asyncio, atado al bus/state del servicio
principal), este módulo abre el puerto directo con threads simples: los
scripts de tuning corren con jetson-service DETENIDO, uno a la vez, sin el
resto del stack (nav, video, WS) activo.

Reusa el framing y los builders de protocol/cobs_frame.py y los defaults de
config.CFG — NO reimplementa COBS/CRC ni la cinemática de ruedas, para no
divergir del resto del proyecto.
"""
import csv
import json
import math
import os
import sys
import threading
import time

# Repo root (capbot-jetson/) en sys.path: este archivo vive en
# scripts/pid_tuning/, dos niveles bajo el root donde están config.py y
# protocol/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import serial

from config import CFG
from protocol.cobs_frame import (
    SerMsgType,
    SerialFrameBuffer,
    build_brake,
    build_heartbeat,
    build_motor,
    build_mode_cmd,
    build_pid_param,
    build_vel_cmd,
)

# ---- IDs para PID_PARAM (firmware: capbot-ESP32/src/main.cpp onPidParam) ----
CTRL_LEFT = 0
CTRL_RIGHT = 1
PARAM_KP = 0
PARAM_KI = 1
PARAM_KD = 2
PARAM_KSTATIC = 3  # feedforward de fricción (offset)
PARAM_KV = 4       # feedforward de fricción (pendiente PWM/(rad/s))

CPS_TO_RADPS = (2.0 * math.pi) / CFG.robot.wheel_cpr

# El watchdog del ESP32 frena a los 200 ms sin RX: heartbeat cada 50 ms.
HEARTBEAT_INTERVAL_S = 0.05

CSV_COLUMNS = [
    "t",            # s desde el inicio del log (reloj local al recibir)
    "mode",         # "manual" | "nav2" (reportado por el firmware)
    "sp_left", "sp_right",       # rad/s (setpoint visto por el firmware)
    "vel_left", "vel_right",     # rad/s (encoder)
    "pwm_left", "pwm_right",     # counts [-32767, 32767] aplicados al motor
    "braking",                   # 0/1
    "enc_left", "enc_right",     # cuentas acumuladas
    "cmd_a", "cmd_b",            # anotación del script (según test; ver header del CSV)
]


class CapbotLink:
    """Abre el serial, manda heartbeat solo, y loguea cada TELEMETRY a memoria.

    cmd_note: par (a, b) que el script de test setea con lo que está
    comandando en ese instante (p.ej. PWM de la rampa o rad/s del step);
    se copia en cada fila para poder correlacionar comando y respuesta.
    """

    def __init__(self, port=None, baud=None):
        self._ser = serial.Serial(
            port=port or CFG.serial.port,
            baudrate=baud or CFG.serial.baudrate,
            timeout=0.05,
            write_timeout=0.2,
        )
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()
        self._wlock = threading.Lock()
        self._buffer = SerialFrameBuffer()
        self._running = True
        self._t0 = time.monotonic()
        self.rows = []
        self.cmd_note = (0.0, 0.0)
        self.hello_seen = False
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._hb_thread = threading.Thread(target=self._hb_loop, daemon=True)
        self._rx_thread.start()
        self._hb_thread.start()

    # ---- TX ----
    def _write(self, pkt):
        with self._wlock:
            self._ser.write(pkt)

    def send_mode(self, mode):
        self._write(build_mode_cmd(mode))

    def send_vel(self, left, right):
        self._write(build_vel_cmd(left, right))

    def send_motor(self, left, right):
        self._write(build_motor(left, right))

    def send_brake(self):
        self._write(build_brake())

    def send_pid_param(self, ctrl_id, param_id, value):
        self._write(build_pid_param(ctrl_id, param_id, value))

    # ---- RX ----
    def _rx_loop(self):
        while self._running:
            try:
                waiting = self._ser.in_waiting
                data = self._ser.read(waiting if waiting > 0 else 1)
            except (OSError, serial.SerialException):
                return
            if data:
                for fr in self._buffer.feed(data):
                    self._on_frame(fr)

    def _on_frame(self, fr):
        if fr.msg_type == SerMsgType.ESP_HELLO:
            self.hello_seen = True
            return
        if fr.msg_type != SerMsgType.TELEMETRY:
            return
        try:
            data = json.loads(fr.payload.decode("utf-8"))
            u = data["u"]
            c = data["ctrl"]
        except (UnicodeDecodeError, ValueError, KeyError):
            return
        note = self.cmd_note
        self.rows.append([
            round(time.monotonic() - self._t0, 4),
            data.get("mode", "?"),
            round(float(c["sp_left"]), 4), round(float(c["sp_right"]), 4),
            round(float(u["vel_left_cps"]) * CPS_TO_RADPS, 4),
            round(float(u["vel_right_cps"]) * CPS_TO_RADPS, 4),
            int(u["pwm_left"]), int(u["pwm_right"]),
            1 if u.get("braking") else 0,
            int(u["enc_left"]), int(u["enc_right"]),
            note[0], note[1],
        ])

    # ---- Heartbeat ----
    def _hb_loop(self):
        while self._running:
            try:
                self._write(build_heartbeat())
            except (OSError, serial.SerialException):
                return
            time.sleep(HEARTBEAT_INTERVAL_S)

    # ---- Utilidades ----
    def latest(self):
        return self.rows[-1] if self.rows else None

    def save_csv(self, path, comment=None):
        with open(path, "w", newline="") as f:
            if comment:
                f.write("# " + comment + "\n")
            w = csv.writer(f)
            w.writerow(CSV_COLUMNS)
            w.writerows(self.rows)
        return len(self.rows)

    def close(self, brake=True):
        if brake:
            try:
                self.send_brake()
                self.send_mode(0)
            except (OSError, serial.SerialException):
                pass
        self._running = False
        time.sleep(0.15)
        try:
            self._ser.close()
        except (OSError, serial.SerialException):
            pass
