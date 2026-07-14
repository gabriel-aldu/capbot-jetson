"""Percepción de obstáculos con la DNN (YOLOv8 + TensorRT).

Portado de capbot-identification-test:
  * yolov8_trt.py    — wrapper del engine TensorRT (copia VERBATIM del repo
                       capbot-identification-test; cambios deben copiarse).
  * ground_plane.py  — proyección pixel -> piso (distancia por plano de piso).
  * detector.py      — hilo de inferencia: consume frames del appsink de la
                       pipeline de video (rama de análisis del tee), corre la
                       DNN y emite Ev.DETECTIONS con los puntos ya
                       reexpresados en el frame del mapa.

controller/obstacle_tracker.py consume Ev.DETECTIONS y mantiene las celdas
bloqueadas del maze.
"""
