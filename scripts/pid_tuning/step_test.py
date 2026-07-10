"""Step de velocidad al PID de rueda del ESP32 (Fase 3 del tuning).

Corre EN la Jetson con jetson-service DETENIDO:

    sudo systemctl stop jetson-service
    cd ~/capbot-jetson
    python3 scripts/pid_tuning/step_test.py

Pone el firmware en AUTONOMOUS_NAV y manda un escalón de setpoint por rueda
(default 5 rad/s en ambas), logueando la telemetría a 50 Hz en un CSV para
graficar con capbot-ESP32/tools/plot_run.py (en el PC).

Secuencia:
  1. (opcional) manda ganancias PID/feedforward en caliente (PID_PARAM)
  2. MODE_CMD -> AUTONOMOUS_NAV, --settle s de baseline en reposo
  3. VEL_CMD(left, right) reenviado cada 100 ms durante --duration s
  4. VEL_CMD(0, 0) durante --tail s (captura la desaceleración)
  5. freno + vuelta a MANUAL

Uso típico (5 rad/s, robot en banco o con ~1 m libre en el piso;
5 rad/s * r=0.035 m ~ 0.175 m/s):

    python3 scripts/pid_tuning/step_test.py                       # 5 rad/s ambas ruedas, 5 s
    python3 scripts/pid_tuning/step_test.py --left 5 --right 5 --duration 5 --out step_5rads.csv
    python3 scripts/pid_tuning/step_test.py --kp 1500 --ki 500     # prueba ganancias sin reflashear
    python3 scripts/pid_tuning/step_test.py --ctrl 0 --kstatic 11800 --kv 1450  # FF sólo rueda izq.

--kp/--ki/--kd/--kstatic/--kv aplican a ambas ruedas salvo que se acote con
--ctrl 0 (izquierda) o --ctrl 1 (derecha). Los valores enviados en caliente
NO persisten tras un reset del ESP32: los definitivos van a
capbot-ESP32/src/main.cpp DEFAULT_CTRL_CFG.
"""
import argparse
import time

from serial_link import CapbotLink, PARAM_KP, PARAM_KI, PARAM_KD, PARAM_KSTATIC, PARAM_KV


def send_gains(link, args):
    ctrls = [0, 1] if args.ctrl is None else [args.ctrl]
    sent = []
    for name, param_id, value in (
        ("kp", PARAM_KP, args.kp),
        ("ki", PARAM_KI, args.ki),
        ("kd", PARAM_KD, args.kd),
        ("kstatic", PARAM_KSTATIC, args.kstatic),
        ("kv", PARAM_KV, args.kv),
    ):
        if value is None:
            continue
        for ctrl in ctrls:
            link.send_pid_param(ctrl, param_id, value)
            time.sleep(0.02)  # un frame por vez, sin saturar el RX del ESP32
        sent.append("{}={} (ctrl {})".format(name, value, ctrls))
    if sent:
        print("Ganancias enviadas: " + "; ".join(sent))
    return sent


def main():
    ap = argparse.ArgumentParser(description="Step de velocidad al PID de rueda")
    ap.add_argument("--port", default=None, help="default: CFG.serial.port (config.py)")
    ap.add_argument("--baud", type=int, default=None, help="default: CFG.serial.baudrate")
    ap.add_argument("--left", type=float, default=5.0, help="setpoint rueda izquierda rad/s (default 5)")
    ap.add_argument("--right", type=float, default=5.0, help="setpoint rueda derecha rad/s (default 5)")
    ap.add_argument("--duration", type=float, default=5.0, help="segundos de step (default 5)")
    ap.add_argument("--settle", type=float, default=1.0, help="segundos de baseline previo (default 1)")
    ap.add_argument("--tail", type=float, default=1.0, help="segundos en setpoint 0 al final (default 1)")
    # Ganancias en caliente (opcionales)
    ap.add_argument("--ctrl", type=int, choices=[0, 1], default=None,
                    help="acotar ganancias a una rueda: 0=izq, 1=der (default ambas)")
    ap.add_argument("--kp", type=float, default=None)
    ap.add_argument("--ki", type=float, default=None)
    ap.add_argument("--kd", type=float, default=None)
    ap.add_argument("--kstatic", type=float, default=None, help="FF de fricción: offset (counts)")
    ap.add_argument("--kv", type=float, default=None, help="FF de fricción: counts/(rad/s)")
    ap.add_argument("--out", default="step_{}.csv".format(time.strftime("%Y%m%d_%H%M%S")))
    args = ap.parse_args()

    link = CapbotLink(args.port, args.baud)
    print("Puerto abierto.")
    try:
        send_gains(link, args)

        # Baseline en NAV sin setpoint (el firmware queda frenado y quieto).
        link.send_mode(1)
        print("Modo AUTONOMOUS_NAV. Baseline {:.1f} s...".format(args.settle))
        time.sleep(args.settle)

        # Step: reenviar cada 100 ms (NAV_VEL_TIMEOUT_MS=1000 en el firmware,
        # el reenvío cubre frames perdidos y mantiene fresco el setpoint).
        print("STEP: left={} rad/s, right={} rad/s por {:.1f} s".format(
            args.left, args.right, args.duration))
        link.cmd_note = (args.left, args.right)
        t_end = time.monotonic() + args.duration
        while time.monotonic() < t_end:
            link.send_vel(args.left, args.right)
            time.sleep(0.1)

        # Cola en cero para registrar la desaceleración controlada.
        print("Setpoint 0 por {:.1f} s...".format(args.tail))
        link.cmd_note = (0.0, 0.0)
        t_end = time.monotonic() + args.tail
        while time.monotonic() < t_end:
            link.send_vel(0.0, 0.0)
            time.sleep(0.1)
    finally:
        comment = "step_test left={} right={} duration={} cmd_a/cmd_b=setpoint comandado (rad/s)".format(
            args.left, args.right, args.duration)
        if args.kp is not None or args.ki is not None or args.kd is not None \
                or args.kstatic is not None or args.kv is not None:
            comment += " | gains: kp={} ki={} kd={} kstatic={} kv={} ctrl={}".format(
                args.kp, args.ki, args.kd, args.kstatic, args.kv, args.ctrl)
        n = link.save_csv(args.out, comment=comment)
        link.close()  # manda freno + MODE 0
        print("CSV guardado: {} ({} muestras, ~{:.1f} s)".format(
            args.out, n, n * 0.02))
    print("Copiar al PC y graficar:  python plot_run.py {}".format(args.out))


if __name__ == "__main__":
    main()
