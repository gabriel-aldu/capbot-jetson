"""Rampa de caracterización PWM -> velocidad de rueda (Fase 2 del tuning).

Corre EN la Jetson con jetson-service DETENIDO:

    sudo systemctl stop jetson-service
    cd ~/capbot-jetson
    python3 scripts/pid_tuning/ramp_test.py

Modo MANUAL: sube el PWM en escalones, espera régimen en cada nivel y mide
la velocidad de cada rueda. Al final ajusta por mínimos cuadrados

    pwm = kStatic + kV * vel        (por rueda, por dirección)

que son exactamente los parámetros del feedforward de fricción del firmware
(Controlador::Config leftKStatic/leftKV/... en capbot-ESP32). También
reporta el PWM de breakaway (primer nivel donde la rueda rompe a girar).

Uso típico (robot EN BANCO con las ruedas libres, o en el piso con espacio):

    python3 scripts/pid_tuning/ramp_test.py
    python3 scripts/pid_tuning/ramp_test.py --max 24000 --step 1500
    python3 scripts/pid_tuning/ramp_test.py --out ramp_banco.csv

El CSV completo queda para graficar con capbot-ESP32/tools/plot_run.py
(en el PC) --ramp.
"""
import argparse
import time

from serial_link import CapbotLink


def steady_speed(link, t_from):
    """Mediana de la velocidad (rad/s) por rueda de las filas con t >= t_from."""
    left = sorted(r[4] for r in link.rows if r[0] >= t_from)
    right = sorted(r[5] for r in link.rows if r[0] >= t_from)
    if not left:
        return 0.0, 0.0
    return left[len(left) // 2], right[len(right) // 2]


def fit_line(points):
    """Mínimos cuadrados pwm = a + b*vel sobre [(vel, pwm), ...]. -> (a, b)"""
    n = len(points)
    if n < 2:
        return None
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    den = n * sxx - sx * sx
    if abs(den) < 1e-9:
        return None
    b = (n * sxy - sx * sy) / den
    a = (sy - b * sx) / n
    return a, b


def run_direction(link, sign, args):
    """Rampa ascendente en una dirección. Retorna [(vel_l, vel_r, pwm), ...]."""
    levels = []
    pwm = args.step
    while pwm <= args.max:
        cmd = sign * pwm
        link.cmd_note = (cmd, cmd)
        link.send_motor(cmd, cmd)
        t_level = time.monotonic()
        # Reenviar el comando durante el hold (el firmware no lo cachea a
        # nivel de watchdog, pero así el nivel sobrevive a un frame perdido).
        while time.monotonic() - t_level < args.hold:
            link.send_motor(cmd, cmd)
            time.sleep(0.1)
        # Régimen: mediana del último 40% del hold.
        t_now = link.rows[-1][0] if link.rows else 0.0
        vl, vr = steady_speed(link, t_now - args.hold * 0.4)
        levels.append((vl, vr, cmd))
        print("  pwm={:>7d}  vel_left={:7.2f} rad/s  vel_right={:7.2f} rad/s".format(cmd, vl, vr))
        pwm += args.step
    # Volver a cero y dejar que se detenga antes de la siguiente dirección.
    link.cmd_note = (0.0, 0.0)
    link.send_motor(0, 0)
    time.sleep(1.5)
    return levels


def analyze(levels, min_move):
    """Por rueda: breakaway y ajuste (kStatic, kV) con los puntos en movimiento."""
    out = {}
    for wheel, idx in (("left", 0), ("right", 1)):
        breakaway = None
        moving = []
        for lv in levels:
            vel, pwm = lv[idx], lv[2]
            if abs(vel) > min_move:
                if breakaway is None:
                    breakaway = abs(pwm)
                moving.append((abs(vel), abs(pwm)))
        fit = fit_line(moving) if len(moving) >= 2 else None
        out[wheel] = {"breakaway": breakaway, "fit": fit, "n": len(moving)}
    return out


def main():
    ap = argparse.ArgumentParser(description="Rampa PWM->velocidad para medir kStatic/kV")
    ap.add_argument("--port", default=None, help="default: CFG.serial.port (config.py)")
    ap.add_argument("--baud", type=int, default=None, help="default: CFG.serial.baudrate")
    ap.add_argument("--step", type=int, default=1000, help="incremento de PWM por nivel (default 1000)")
    ap.add_argument("--max", type=int, default=30000, help="PWM máximo de la rampa (default 30000)")
    ap.add_argument("--hold", type=float, default=1.0, help="segundos por nivel (default 1.0)")
    ap.add_argument("--min-move", type=float, default=0.3,
                    help="rad/s para considerar la rueda en movimiento (default 0.3)")
    ap.add_argument("--skip-reverse", action="store_true", help="sólo dirección adelante")
    ap.add_argument("--out", default="ramp_{}.csv".format(time.strftime("%Y%m%d_%H%M%S")))
    args = ap.parse_args()

    link = CapbotLink(args.port, args.baud)
    print("Puerto abierto. Modo MANUAL, rampa hasta {} en pasos de {} ({} s/nivel)".format(
        args.max, args.step, args.hold))
    results = {}
    try:
        link.send_mode(0)  # MANUAL
        time.sleep(0.5)

        print("== Dirección ADELANTE ==")
        fwd = run_direction(link, +1, args)
        results["fwd"] = analyze(fwd, args.min_move)

        if not args.skip_reverse:
            print("== Dirección REVERSA ==")
            rev = run_direction(link, -1, args)
            results["rev"] = analyze(rev, args.min_move)
    finally:
        link.send_motor(0, 0)
        n = link.save_csv(args.out, comment="ramp_test step={} max={} hold={} cmd_a/cmd_b=pwm comandado".format(
            args.step, args.max, args.hold))
        link.close()
        print("\nCSV guardado: {} ({} muestras)".format(args.out, n))

    # ---- Reporte ----
    print("\n================ RESULTADOS ================")
    suggestion = {}
    for wheel in ("left", "right"):
        ks_list, kv_list = [], []
        for direction in results:
            r = results[direction][wheel]
            print("{:5s} {}: breakaway={} counts, puntos en mov.={}".format(
                wheel, direction, r["breakaway"], r["n"]))
            if r["fit"]:
                a, b = r["fit"]
                print("            ajuste: pwm = {:.0f} + {:.0f} * vel[rad/s]".format(a, b))
                ks_list.append(a)
                kv_list.append(b)
        if ks_list:
            suggestion[wheel] = (sum(ks_list) / len(ks_list), sum(kv_list) / len(kv_list))

    if suggestion:
        print("\nValores sugeridos para DEFAULT_CTRL_CFG (capbot-ESP32/src/main.cpp):")
        for wheel in ("left", "right"):
            if wheel in suggestion:
                ks, kv = suggestion[wheel]
                print("  {}KStatic = {:.0f}   {}KV = {:.0f}".format(wheel, ks, wheel, kv))
        print("\nO en caliente antes de un step_test:")
        for wheel, ctrl in (("left", 0), ("right", 1)):
            if wheel in suggestion:
                ks, kv = suggestion[wheel]
                print("  step_test.py ... --ctrl {} --kstatic {:.0f} --kv {:.0f}".format(ctrl, ks, kv))


if __name__ == "__main__":
    main()
