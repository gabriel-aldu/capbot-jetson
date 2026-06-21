"""Puente ROS2 <-> EventBus del servicio Jetson.

Expone dos tópicos (std_msgs/Float32MultiArray, sin paquete de interfaces
propio):

  from_bridge  (publica)   [x, y, theta, v, w, setpoint_x, setpoint_y, setpoint_theta]
      - x, y, theta, v, w: odometría reportada por el ESP32 vía
        telemetría serial (Ev.TELEMETRY). v = linear.x, w = angular.z.
        El firmware manda estos valores anidados bajo la clave "odo" (ver
        SensorHub::buildPayload): odo.x, odo.y, odo.a (¡no "theta"!), odo.v,
        odo.w. Ajustar `_on_telemetry` si el firmware cambia.
      - setpoint_x/y/theta: último setpoint recibido del host por UDP
        (Ev.CMD_SETPOINT, comp_id 0/1/2). Se mantiene en memoria porque el
        host manda un componente por mensaje, no los tres juntos.

  to_bridge    (suscribe)  [left, right, stop]
      - left, right: consigna de velocidad por rueda -> Ev.CMD_MOTOR.
      - stop: != 0 frena ya (Ev.STOP_MOTORS) e ignora left/right.

rclpy no comparte el loop de asyncio de main.py, así que este nodo spinea
en un hilo dedicado (vía run_in_executor). Dirección bus->ROS corre en el
hilo del loop asyncio (donde se emiten TELEMETRY/CMD_SETPOINT) y publicar
ahí es seguro. Dirección ROS->bus corre en el hilo del executor de rclpy;
como `core.bus.bus` no es thread-safe (usa asyncio.Queue por debajo en los
consumidores), esas llamadas se reenvían al loop con `call_soon_threadsafe`.
"""
import asyncio
import logging
import threading

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import Float32MultiArray
except ImportError:  # pragma: no cover
    rclpy = None
    Node = object

from core.bus import Ev, bus

log = logging.getLogger(__name__)

_SPIN_TIMEOUT_S = 0.05


class RosBridge(Node):
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__('capbot_jetson_bridge')
        self._loop = loop
        self._setpoint = [0.0, 0.0, 0.0]  # x, y, theta

        self._pub = self.create_publisher(Float32MultiArray, 'from_bridge', 10)
        self.create_subscription(Float32MultiArray, 'to_bridge', self._on_to_bridge, 10)

        bus.on(Ev.TELEMETRY, self._on_telemetry)
        bus.on(Ev.CMD_SETPOINT, self._on_setpoint)

    # -------------------- bus -> ROS --------------------
    def _on_telemetry(self, data) -> None:
        if not isinstance(data, dict):
            return
        odo = data.get("odo")
        if not isinstance(odo, dict):
            return
        try:
            x = float(odo.get("x", 0.0))
            y = float(odo.get("y", 0.0))
            theta = float(odo.get("a", 0.0))
            v = float(odo.get("v", 0.0))
            w = float(odo.get("w", 0.0))
        except (TypeError, ValueError):
            return

        msg = Float32MultiArray()
        msg.data = [x, y, theta, v, w] + self._setpoint
        self._pub.publish(msg)

    def _on_setpoint(self, data) -> None:
        try:
            comp_id = data["comp_id"]
            value = float(data["value"])
        except (KeyError, TypeError, ValueError):
            return
        if 0 <= comp_id <= 2:
            self._setpoint[comp_id] = value

    # -------------------- ROS -> bus --------------------
    def _on_to_bridge(self, msg) -> None:
        if len(msg.data) < 3:
            self.get_logger().warn('to_bridge: se esperaban 3 valores [left, right, stop]')
            return
        left, right, stop = msg.data[0], msg.data[1], msg.data[2]

        if stop != 0.0:
            self._loop.call_soon_threadsafe(bus.emit, Ev.STOP_MOTORS, None)
        else:
            payload = {"left": int(left), "right": int(right), "aux": 0, "seq": 0}
            self._loop.call_soon_threadsafe(bus.emit, Ev.CMD_MOTOR, payload)


def _spin_until_stopped(node: "RosBridge", stop_event: asyncio.Event) -> None:
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while not stop_event.is_set():
            executor.spin_once(timeout_sec=_SPIN_TIMEOUT_S)
    finally:
        executor.remove_node(node)


async def run_ros_bridge(stop_event: asyncio.Event, loop: asyncio.AbstractEventLoop) -> None:
    """Tarea asyncio para main.py: arranca rclpy y spinea hasta `stop_event`."""
    if rclpy is None:
        log.error("rclpy no está instalado; puente ROS2 deshabilitado")
        await stop_event.wait()
        return

    rclpy.init(args=None)
    # Construido en el hilo del loop asyncio para que bus.on() no compita
    # con bus.emit() llamado desde otro hilo durante el arranque.
    node = RosBridge(loop)
    try:
        await loop.run_in_executor(None, _spin_until_stopped, node, stop_event)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(args=None) -> None:
    """Entry point standalone (p.ej. `ros2 run`). Sin main.py corriendo en
    el mismo proceso no hay telemetría/setpoint reales fluyendo por el bus;
    esto sólo sirve para probar los tópicos ROS2 de forma aislada."""
    if rclpy is None:
        raise RuntimeError("rclpy no está instalado")

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    rclpy.init(args=args)
    node = RosBridge(loop)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        loop.call_soon_threadsafe(loop.stop)


if __name__ == '__main__':
    main()
