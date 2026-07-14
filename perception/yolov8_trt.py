# yolov8_trt.py  (Python 3.6, JetPack 4.6)
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda

class YoloV8TRT:
    def __init__(self, engine_path, imgsz=416, conf_th=0.40, iou_th=0.50):
        self.imgsz, self.conf_th, self.iou_th = imgsz, conf_th, iou_th

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()

        # buffers (shapes read from the engine, not assumed)
        self.in_shape  = tuple(self.engine.get_binding_shape(0))   # (1,3,S,S)
        self.out_shape = tuple(self.engine.get_binding_shape(1))   # (1,84,N)
        self.h_in  = cuda.pagelocked_empty(int(np.prod(self.in_shape)),  np.float32)
        self.h_out = cuda.pagelocked_empty(int(np.prod(self.out_shape)), np.float32)
        self.d_in  = cuda.mem_alloc(self.h_in.nbytes)
        self.d_out = cuda.mem_alloc(self.h_out.nbytes)
        self.stream = cuda.Stream()

    # ---------- preprocessing: letterbox (same convention as Ultralytics) ----
    def _letterbox(self, img):
        h0, w0 = img.shape[:2]
        s = self.imgsz
        r = min(s / h0, s / w0)                    # scale ratio
        nw, nh = int(round(w0 * r)), int(round(h0 * r))
        pw, ph = (s - nw) // 2, (s - nh) // 2      # padding offsets
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        canvas[ph:ph + nh, pw:pw + nw] = resized
        return canvas, r, pw, ph

    # ---------- inference ----------------------------------------------------
    def infer(self, frame_bgr):
        img, r, pw, ph = self._letterbox(frame_bgr)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.ascontiguousarray(img.transpose(2, 0, 1)).ravel()
        np.copyto(self.h_in, chw)

        cuda.memcpy_htod_async(self.d_in, self.h_in, self.stream)
        self.ctx.execute_async_v2([int(self.d_in), int(self.d_out)],
                                  self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
        self.stream.synchronize()

        out = self.h_out.reshape(self.out_shape)[0]     # (84, N)
        return self._postprocess(out, r, pw, ph, frame_bgr.shape)

    # ---------- postprocessing -----------------------------------------------
    def _postprocess(self, out, r, pw, ph, orig_shape):
        out = out.T                                     # (N, 84)
        scores = out[:, 4:].max(axis=1)
        keep = scores > self.conf_th
        if not keep.any():
            return []
        boxes  = out[keep, :4]                          # cx,cy,w,h  (letterbox px)
        scores = scores[keep]
        clses  = out[keep, 4:].argmax(axis=1)

        # undo letterbox -> original-image pixels
        boxes[:, 0] = (boxes[:, 0] - pw) / r            # cx
        boxes[:, 1] = (boxes[:, 1] - ph) / r            # cy
        boxes[:, 2] /= r                                # w
        boxes[:, 3] /= r                                # h

        # NMS: old OpenCV wants [x,y,w,h]
        xywh = np.copy(boxes)
        xywh[:, 0] -= xywh[:, 2] / 2
        xywh[:, 1] -= xywh[:, 3] / 2
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(),
                               self.conf_th, self.iou_th)
        if len(idx) == 0:
            return []
        idx = np.array(idx).flatten()

        H, W = orig_shape[:2]
        dets = []
        for i in idx:
            x, y, w, h = xywh[i]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W - 1, x + w), min(H - 1, y + h)
            dets.append(dict(box=(x1, y1, x2, y2),
                             conf=float(scores[i]),
                             cls=int(clses[i])))
        return dets