# Herramientas de tuning del PID de velocidad por rueda

Scripts para caracterizar los motores y tunear el PID + feedforward de
fricción del firmware (`DEFAULT_CTRL_CFG` en `capbot-ESP32/src/main.cpp`).

Corren **en la Jetson** (hablan directo por el serial con el ESP32), a
diferencia de `capbot-ESP32/tools/plot_run.py` que corre en el PC.

| Script | Qué hace |
|---|---|
| `serial_link.py` | Módulo compartido: enlace serial standalone (threads), reusa `protocol/cobs_frame.py` y `config.CFG` del proyecto — no reimplementa framing ni cinemática |
| `ramp_test.py` | Rampa PWM→velocidad en MANUAL; mide `kStatic`/`kV` por rueda |
| `step_test.py` | Escalón de setpoint (default 5 rad/s) al PID en AUTONOMOUS_NAV |

Como viven dentro de `capbot-jetson`, llegan a la Jetson con el `git pull`
normal del repo — no hace falta copiarlos aparte.

## Por qué en la Jetson y no en capbot-ESP32

El ESP32 sólo tiene un puerto serial y un solo dueño a la vez: el enlace
COBS+CRC vive físicamente entre la Jetson y el ESP32
(`/dev/ttyTHS1`, ver `hw/esp32_link.py`). Estos scripts necesitan ese mismo
puerto, así que deben ejecutarse donde el puerto existe: la Jetson.
`capbot-ESP32` sigue siendo el dueño de los *parámetros físicos* que se
tunean (WHEEL_CPR, PID_PARAM ids, DEFAULT_CTRL_CFG); estos scripts sólo los
ejercitan desde el otro extremo del cable.

## Flujo de trabajo

```bash
# 1. En la Jetson: traer los scripts (ya están si el repo está actualizado)
ssh <jetson_user>@<jetson_ip>
cd ~/capbot-jetson && git pull

# 2. Liberar el puerto serial (un solo dueño)
sudo systemctl stop jetson-service

# 3. Caracterización (Fase 2) — robot en banco con ruedas libres
python3 scripts/pid_tuning/ramp_test.py --out ramp_banco.csv
#   -> imprime kStatic y kV sugeridos por rueda

# 4. Step de 5 rad/s (Fase 3) — banco primero, después piso
#    (5 rad/s * r=0.035 m ~ 0.175 m/s; con ~2 m libres alcanza)
python3 scripts/pid_tuning/step_test.py --out step_banco.csv
#    iterar ganancias en caliente sin reflashear:
python3 scripts/pid_tuning/step_test.py --kp 1500 --ki 500 --kstatic 12000 --kv 1600 --out step_v2.csv

# 5. Copiar los CSV al PC y graficar (capbot-ESP32/tools/plot_run.py)
#    desde el PC:
#    scp <jetson_user>@<jetson_ip>:~/capbot-jetson/*.csv .
#    python "capbot-ESP32/tools/plot_run.py" step_banco.csv
#    python "capbot-ESP32/tools/plot_run.py" ramp_banco.csv --ramp

# 6. Al terminar: fijar los valores ganadores en DEFAULT_CTRL_CFG
#    (capbot-ESP32/src/main.cpp), recompilar/flashear, y reanudar el servicio
sudo systemctl start jetson-service
```

Los valores enviados con `--kp/--ki/--kd/--kstatic/--kv` viven en RAM del
ESP32: se pierden al resetear. Los definitivos van a `DEFAULT_CTRL_CFG` en
`capbot-ESP32`.

## Matriz de pruebas sugerida (Fases 3-4)

1. Step +5 rad/s ambas ruedas (banco, luego piso)
2. Step -5 rad/s (reversa: verifica simetría del feedforward)
3. Step bajo (~1 rad/s): régimen dominado por fricción, el caso difícil
4. Reversión -5 → +5 rad/s: cruce por cero con cambio de signo del FF
   (`step_test.py` corre un step por invocación; para la reversión lanzar
   dos corridas seguidas)

## Seguridad

- El heartbeat lo manda el script cada 50 ms; si el script muere, el
  watchdog del ESP32 (200 ms) frena solo.
- `step_test.py` siempre termina con freno + vuelta a MANUAL, incluso con
  Ctrl+C (bloque `finally`).
- `ramp_test.py` mueve ambas ruedas en la misma dirección: en banco no hay
  riesgo, en piso el robot avanza en línea recta — dejar espacio libre.
- En el piso, tener a mano el botón de emergencia.
