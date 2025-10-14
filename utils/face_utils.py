from __future__ import annotations
import os
from typing import Tuple, List
import numpy as np
import cv2

class BaseDetector:
    """
    All detectors must implement:
        process(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
    returning:
        boxes:  (N,4) float32 in pixel coords [x0,y0,x1,y1]
        scores: (N,)  float32
    """
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        """Optional cleanup."""
        pass


# ================= MediaPipe wrapper =================

class MediaPipeDetector(BaseDetector):
    """
    MediaPipe FaceDetection (no external model file).
    Args:
        model_selection: 0 (short range) or 1 (full range)
        min_conf: minimum detection confidence
    """
    def __init__(self, model_selection: int = 1, min_conf: float = 0.3) -> None:
        import mediapipe as mp  # lazy import
        self._mp = mp
        self.det = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_conf
        )
    
    @staticmethod
    def adjust_box(box):
        cx = (box[0]+box[2])/2
        w = (box[2]-box[0])/2
        cy = (box[1]+box[3])/2
        h = (box[3]-box[1])/2
        w*= 1.3
        h*= 1.3
        cy = cy-h*0.2
        return cx-w, cy-h, cx+w, cy+h

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H, W = frame.shape[:2]
        res = self.det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.detections is None:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

        boxes, scores = [], []
        for d in res.detections:
            s = float(d.score[0])
            rb = d.location_data.relative_bounding_box
            x0 = rb.xmin * W
            y0 = rb.ymin * H
            x1 = (rb.xmin + rb.width) * W
            y1 = (rb.ymin + rb.height) * H
            x0, y0, x1, y1 = MediaPipeDetector.adjust_box([x0,y0,x1,y1])
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(float(W), x1), min(float(H), y1)
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append([x0, y0, x1, y1])
            scores.append(s)

        if not boxes:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        return np.asarray(boxes, np.float32), np.asarray(scores, np.float32)

    def close(self) -> None:
        try:
            self.det.close()
        except Exception:
            pass


# ================= YuNet (OpenCV FaceDetectorYN) wrapper =================
class YuNetDetector(BaseDetector):
    """
    OpenCV YuNet via FaceDetectorYN with:
      - exact input-size handling to avoid spatial blind-spots,
      - optional internal downscaling (downscale in (0,1], default 1.0),
      - score normalization to [0,1],
      - configurable top_k to reduce NMS overhead.
    NOTE: No setBackend/setTarget calls (not present in some OpenCV builds).
    """
    def __init__(self,
                 model_path: str,
                 score_thresh: float = 0.3,
                 nms_thresh: float = 0.3,
                 top_k: int = 100,
                 downscale: float = 1.0,
                 normalize_scores: bool = False) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        if not (0.0 < downscale <= 1.0):
            raise ValueError("downscale must be in (0,1].")

        self.downscale = float(downscale)
        self.normalize_scores = bool(normalize_scores)

        # Create with a dummy size; always set the real size right before detect()
        self.det = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(640, 480),
            score_threshold=float(score_thresh),
            nms_threshold=float(nms_thresh),
            top_k=int(top_k),
        )

        # Cache of last (W,H) set on the detector
        self._last_size: Tuple[int, int] = (640, 480)

        try:
            cv2.setUseOptimized(True)
        except Exception:
            pass

    @staticmethod
    def _floor_to_mult32(x: int) -> int:
        x = max(32, x)
        return (x // 32) * 32

    def _set_input_size(self, width: int, height: int) -> None:
        """FaceDetectorYN expects (W, H). Keep it exactly the same as the image passed to detect()."""
        size = (int(width), int(height))
        if size != self._last_size:
            self.det.setInputSize(size)
            self._last_size = size

    @staticmethod
    def _normalize_scores_to_unit(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        smax = float(scores.max())
        if smax <= 1.0:
            return scores
        if smax <= 100.0:
            return scores * (1.0 / 100.0)
        if smax <= 255.0:
            return scores * (1.0 / 255.0)
        return np.clip(scores, 0.0, 1.0)

    def process(self, frame: np.ndarray):
        H, W = frame.shape[:2]

        if self.downscale < 1.0:
            # Downscale, but round each dimension to a multiple of 32 to keep feature-map alignment stable
            in_w = self._floor_to_mult32(int(W * self.downscale))
            in_h = self._floor_to_mult32(int(H * self.downscale))
            # Keep aspect ratio (for 640x480 both stay multiples of 32 naturally)
            small = cv2.resize(frame, (in_w, in_h), interpolation=cv2.INTER_AREA)

            # MUST match the exact size we pass to detect()
            self._set_input_size(in_w, in_h)
            out = self.det.detect(small)
            if out is None or out[1] is None or len(out[1]) == 0:
                return np.zeros((0,4), np.float32), np.zeros((0,), np.float32)

            faces = out[1]  # [x, y, w, h, score, (landmarks...)]
            sx = float(W) / float(in_w)
            sy = float(H) / float(in_h)

            boxes = np.empty((faces.shape[0], 4), np.float32)
            boxes[:, 0] = faces[:, 0] * sx
            boxes[:, 1] = faces[:, 1] * sy
            boxes[:, 2] = (faces[:, 0] + faces[:, 2]) * sx
            boxes[:, 3] = (faces[:, 1] + faces[:, 3]) * sy

            scores = faces[:, 14].astype(np.float32)
            if self.normalize_scores:
                scores = self._normalize_scores_to_unit(scores)
            return boxes, scores

        # Full-size path (640x480 already multiples of 32)
        self._set_input_size(W, H)
        out = self.det.detect(frame)
        if out is None or out[1] is None or len(out[1]) == 0:
            return np.zeros((0,4), np.float32), np.zeros((0,), np.float32)

        faces = out[1]
        boxes = np.empty((faces.shape[0], 4), np.float32)
        boxes[:, 0] = faces[:, 0]
        boxes[:, 1] = faces[:, 1]
        boxes[:, 2] = faces[:, 0] + faces[:, 2]
        boxes[:, 3] = faces[:, 1] + faces[:, 3]
        scores = faces[:, 4].astype(np.float32)
        if self.normalize_scores:
            scores = self._normalize_scores_to_unit(scores)
        return boxes, scores

    def close(self) -> None:
        pass
# ================= SCRFD wrapper (InsightFace) =================
class SCRFDDetector(BaseDetector):
    """
    SCRFD via InsightFace (robust to detect() signature differences).
    Sets detection threshold and internal size in .prepare(), not in .detect().
    """
    def __init__(self, model_path: str,
                 score_thresh: float = 0.3,
                 nms_thresh: float = 0.4,
                 input_size: tuple[int, int] | None = (640, 640)):
        import insightface
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.det = insightface.model_zoo.get_model(model_path)   # ONNX path
        # Many builds support det_thresh + det_size here (and ignore nms thresh)
        # Some ignore nms here; that's fine — internal postproc handles it.
        self.det.prepare(ctx_id=-1,
                         det_thresh=float(score_thresh),
                         det_size=tuple(input_size) if input_size else None)
        # Keep copies for factory consistency
        self.score_thresh = float(score_thresh)
        self.nms_thresh = float(nms_thresh)
        self.input_size = tuple(input_size) if input_size else None

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # NOTE: do NOT pass thresh/threshold here; older builds don’t accept them.
        # Also avoid passing input_size again to sidestep kw clashes.
        bboxes, _ = self.det.detect(frame, input_size=self.input_size, max_num=50)
        if bboxes is None or len(bboxes) == 0:
            return np.zeros((0,4), np.float32), np.zeros((0,), np.float32)
        boxes  = bboxes[:, :4].astype(np.float32)
        scores = bboxes[:, 4].astype(np.float32)
        return boxes, scores

    def close(self) -> None:
        pass

def _softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-9)
    
# ================= UltraFace (ONNXRuntime) wrapper =================
class UltraFaceDetector(BaseDetector):
    """
    Ultra-Light-Fast-Generic-Face-Detector-1MB (RFB).
    Handles both head styles:
      - loc as SSD deltas (decode with priors)
      - loc as corners normalized in [0,1] (use as-is)
    """
    def __init__(
        self,
        model_path: str,
        score_thresh: float = 0.3,
        nms_thresh: float = 0.45,
        input_size=(640, 480),    # use 320 for speed -> (320,240)
        min_size: int = 24,
        top_k: int = 200,
        ort_log_level: int = 3,
        intra_threads: int = 0,
    ) -> None:
        import onnxruntime as ort
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        # Accept int or tuple (map 640->(640,480), 320->(320,240))
        if isinstance(input_size, int):
            self.in_w, self.in_h = (640, 480) if input_size == 640 else ((320, 240) if input_size == 320 else (int(input_size), int(round(input_size*0.75))))
        else:
            self.in_w, self.in_h = int(input_size[0]), int(input_size[1])

        so = ort.SessionOptions()
        so.log_severity_level = int(ort_log_level)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_threads and intra_threads > 0:
            so.intra_op_num_threads = intra_threads
            so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])

        self.iname = self.sess.get_inputs()[0].name
        self.score_thresh = float(score_thresh)
        self.nms_thresh = float(nms_thresh)
        self.min_size = int(min_size)
        self.top_k = int(top_k)
        self._priors = self._generate_priors(self.in_w, self.in_h)

    @staticmethod
    def _generate_priors(w: int, h: int) -> np.ndarray:
        steps = [8, 16, 32, 64]
        min_sizes = [[10,16,24], [32,48], [64,96], [128,192,256]]
        priors = []
        for idx, s in enumerate(steps):
            fw = (w + s - 1) // s
            fh = (h + s - 1) // s
            for y in range(fh):
                for x in range(fw):
                    for ms in min_sizes[idx]:
                        cx = (x + 0.5) * s / w
                        cy = (y + 0.5) * s / h
                        sw = ms / w
                        sh = ms / h
                        priors.append([cx, cy, sw, sh])
        return np.asarray(priors, dtype=np.float32)

    @staticmethod
    def _ssd_decode(loc: np.ndarray, priors: np.ndarray, variances=[0.1, 0.2]) -> np.ndarray:
        # SSD-style decode to [x1,y1,x2,y2] normalized
        boxes = np.concatenate([
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
        ], axis=1)
        boxes[:, 0:2] -= boxes[:, 2:4] / 2
        boxes[:, 2:4] += boxes[:, 0:2]
        return boxes

    @staticmethod
    def _nms(bboxes: np.ndarray, scores: np.ndarray, iou_thresh=0.45, top_k=750) -> List[int]:
        if len(bboxes) == 0:
            return []
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0 and len(keep) < top_k:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]
        return keep

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        inp = cv2.resize(frame, (self.in_w, self.in_h)).astype(np.float32)
        inp = (inp - 127.5) / 128.0
        blob = inp.transpose(2, 0, 1)[None, ...]

        outs = self.sess.run(None, {self.iname: blob})

        # Collect heads: (N,4) loc, (N,2) conf (or (N,) score)
        loc = None; conf2 = None; score1 = None
        for o in outs:
            a = np.squeeze(o)
            if a.ndim == 2 and a.shape[1] == 4:
                loc = a.astype(np.float32)
            elif a.ndim == 2 and a.shape[1] == 2:
                conf2 = a.astype(np.float32)
            elif a.ndim == 1:
                score1 = a.astype(np.float32)

        if loc is None:
            return np.zeros((0,4), np.float32), np.zeros((0,), np.float32)

        # scores (probabilities or logits; treat as probabilities here)
        scores = conf2[:, 1] if conf2 is not None else (score1 if score1 is not None else None)
        if scores is None:
            return np.zeros((0,4), np.float32), np.zeros((0,), np.float32)

        # filter low scores
        mask = scores >= self.score_thresh
        # breakpoint()
        if not np.any(mask):
            return np.zeros((0,4), np.float32), np.zeros((0,), np.float32)
        loc = loc[mask]
        scores = scores[mask]
        scores = scores*scores

        if loc.ndim == 1:
            loc = loc.reshape(-1, 4)

        # Try to auto-detect if loc is already corners in [0,1]
        # Heuristic: values mostly in [0,1] and x2>x1, y2>y1 for majority
        loc_min, loc_max = float(loc.min()), float(loc.max())
        corners_like = (0.0 <= loc_min <= 1.0) and (0.0 <= loc_max <= 1.0)
        if corners_like:
            x1_ok = (loc[:, 2] > loc[:, 0]).mean() > 0.8
            y1_ok = (loc[:, 3] > loc[:, 1]).mean() > 0.8
            corners_like = corners_like and x1_ok and y1_ok

        if corners_like:
            boxes_rel = loc  # already normalized corners
        else:
            # SSD-style decode with priors
            num_priors = loc.shape[0]
            if self._priors.shape[0] != num_priors:
                self._priors = self._generate_priors(self.in_w, self.in_h)
                m = min(self._priors.shape[0], num_priors)
                self._priors = self._priors[:m]
                loc = loc[:m]; scores = scores[:m]
            boxes_rel = self._ssd_decode(loc, self._priors)

        # Map to original image coords
        boxes = np.empty_like(boxes_rel)
        boxes[:, [0, 2]] = (boxes_rel[:, [0, 2]] * w).clip(0, w)
        boxes[:, [1, 3]] = (boxes_rel[:, [1, 3]] * h).clip(0, h)

        # Remove tiny boxes
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        big = (ws >= self.min_size) & (hs >= self.min_size)
        if not np.any(big):
            return np.zeros((0,4), np.float32), np.zeros((0,), np.float32)
        boxes = boxes[big]; scores = scores[big]

        keep = self._nms(boxes, scores, self.nms_thresh, top_k=self.top_k)
        if not keep:
            return np.zeros((0,4), np.float32), np.zeros((0,), np.float32)
        boxes = boxes[keep]; scores = scores[keep]
        return boxes.astype(np.float32), scores.astype(np.float32)

# ================= Factory =================

def select_face_detector(
    detector_name: str,
    res_dir: str = "res",
    **kwargs
) -> BaseDetector:
    """
    Create a detector by name:
        'mediapipe' | 'yunet' | 'scrfd' | 'ultraface'

    Common kwargs:
        mediapipe: model_selection=1, min_conf=0.3
        yunet:     model="face_detection_yunet_2023mar.onnx", score_thresh=0.3, nms_thresh=0.3, top_k=5000, downscale=1.0
        scrfd:     model="scrfd_2.5g.onnx", score_thresh=0.3, nms_thresh=0.4
        ultraface: model="version-RFB-640.onnx", score_thresh=0.3, nms_thresh=0.45, input_size=(640,480) or 640
    """
    name = (detector_name or "").strip().lower()

    if name in ("mediapipe", "mp", "blazeface"):
        return MediaPipeDetector(
            model_selection=kwargs.get("model_selection", 0),
            min_conf=kwargs.get("min_conf", 0.3),
        )

    if name in ("yunet", "opencv-yunet"):
        model = kwargs.get("model", "face_detection_yunet_2023mar.onnx")
        return YuNetDetector(
            model_path=os.path.join(res_dir, model),
            score_thresh=kwargs.get("score_thresh", 0.3),
            nms_thresh=kwargs.get("nms_thresh", 0.3),
            top_k=kwargs.get("top_k", 100),
            downscale=kwargs.get("downscale", 0.5),
        )

    if name in ("scrfd",):
        model = kwargs.get("model", "scrfd_2.5g.onnx")
        return SCRFDDetector(
            model_path=os.path.join(res_dir, model),
            score_thresh=kwargs.get("score_thresh", 0.3),
            nms_thresh=kwargs.get("nms_thresh", 0.4),
            input_size=kwargs.get("input_size", (416, 416)),  # good CPU default
        )

    if name in ("ultraface", "ultra-face", "ultralight"):
        model = kwargs.get("model", "version-RFB-320.onnx")  # 320 model
        return UltraFaceDetector(
            model_path=os.path.join(res_dir, model),
            score_thresh=kwargs.get("score_thresh", 0.3),
            nms_thresh=kwargs.get("nms_thresh", 0.45),
            input_size=kwargs.get("input_size", 320),   # -> (320,240)
            min_size=kwargs.get("min_size", 24),
            top_k=kwargs.get("top_k", 200),
            intra_threads=kwargs.get("intra_threads", 0),
        )


    raise ValueError(f"Unknown detector: {detector_name!r}")
