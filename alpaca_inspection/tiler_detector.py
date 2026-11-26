"""Inferencia de impurezas sobre recortes nativos y reconstrucción del fotograma."""

from functools import lru_cache
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .paths import MODELS_DIR
from .tiling_geometry import clip_bbox, deduplicate_cross_tile_annotations, positions

TILE_SIZE = 640
OVERLAP = 128
CLASS_NAMES = {0: "particle", 1: "straw"}
BOX_COLORS = {0: (0, 0, 255), 1: (255, 0, 0)}
AUDIT_COLORS = {0: (255, 210, 0), 1: (0, 145, 255)}
DEFAULT_WEIGHTS = MODELS_DIR / "bestS.pt"


@lru_cache(maxsize=32)
def tile_grid(height, width, roi=None):
    x1, y1, x2, y2 = roi if roi is not None else (0, 0, width, height)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("La ROI debe estar dentro de la imagen.")
    xs = positions(x2 - x1, TILE_SIZE, OVERLAP)
    ys = positions(y2 - y1, TILE_SIZE, OVERLAP)
    return tuple((x1 + x, y1 + y) for y in ys for x in xs)


class TiledImpurityDetector:
    def __init__(self, weights=DEFAULT_WEIGHTS, *, tile_batch_size=20, half=True, device=None,
                 warmup_batches=1, engine_batch_size=None):
        from ultralytics import YOLO
        from ultralytics.cfg import DEFAULT_CFG_DICT
        import torch

        weights = Path(weights)
        if tile_batch_size < 1 or (engine_batch_size is not None and engine_batch_size < 1):
            raise ValueError("El tamaño del lote debe ser positivo.")
        if weights.suffix not in {".pt", ".engine"} or not weights.is_file():
            raise ValueError(f"Se esperaba un archivo local de pesos .pt o .engine: {weights}")
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        use_half = half and self.device != "cpu"
        self.precision = {"quantize": 16 if use_half else None} if "quantize" in DEFAULT_CFG_DICT else {"half": use_half}
        self.warmup_batches = warmup_batches
        self.model = YOLO(str(weights), task="detect")
        if weights.suffix == ".engine":
            if self.device == "cpu":
                raise ValueError("TensorRT requiere un dispositivo CUDA.")
            # Ultralytics expone la forma del lote en el backend antes de inferir.
            predictor = self.model._smart_load("predictor")(
                overrides={**self.precision, "device": self.device, "task": "detect",
                           "imgsz": TILE_SIZE, "verbose": False},
                _callbacks=self.model.callbacks,
            )
            predictor.setup_model(model=self.model.model, verbose=False)
            self.model.predictor = predictor
            backend = predictor.model
            shape = backend.bindings["images"].shape
            if tuple(shape[-2:]) != (TILE_SIZE, TILE_SIZE):
                raise ValueError("El motor TensorRT debe recibir imágenes de 640 × 640.")
            tile_batch_size = min(tile_batch_size, shape[0])
            if not backend.dynamic:
                if engine_batch_size is not None and engine_batch_size != shape[0]:
                    raise ValueError(f"El batch fijo del motor TensorRT es {shape[0]}.")
                engine_batch_size = shape[0]
        self.engine_batch_size = engine_batch_size
        self.tile_batch_size = min(tile_batch_size, engine_batch_size or tile_batch_size)
        if dict(self.model.names) != CLASS_NAMES:
            raise ValueError(f"El modelo debe usar las clases {CLASS_NAMES}: {self.model.names}")

    def _predict(self, tiles, conf, iou):
        count = len(tiles)
        if self.engine_batch_size and count < self.engine_batch_size:
            tiles = tiles + [tiles[-1]] * (self.engine_batch_size - count)
        return self.model.predict(
            tiles, imgsz=TILE_SIZE, conf=conf, iou=iou, **self.precision,
            device=self.device, verbose=False,
        )[:count]

    def predict_frame(self, frame, *, conf_particle=0.10, conf_straw=0.10, roi=None,
                      dedup="cross_tile_merge", merge="wbf", class_agnostic=False,
                      tile_iou=0.45, color_order="BGR"):
        """Devuelve detecciones globales y tiempos en milisegundos, con los orígenes de tiles."""
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Se esperaba una imagen de tres canales uint8.")
        height, width = frame.shape[:2]
        coords = tile_grid(height, width, tuple(roi) if roi is not None else None)
        thresholds = {0: conf_particle, 1: conf_straw}
        if any(not 0 < threshold <= 1 for threshold in thresholds.values()):
            raise ValueError("Los umbrales de confianza deben estar en (0, 1].")
        base_conf = min(thresholds.values())
        if self.warmup_batches:
            blank = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
            for _ in range(self.warmup_batches):
                self._predict([blank] * min(self.tile_batch_size, len(coords)), base_conf, tile_iou)
            self.warmup_batches = 0

        start = perf_counter()
        if color_order == "RGB":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif color_order != "BGR":
            raise ValueError("El orden de color debe ser BGR o RGB.")
        tiles = [frame[y:y + TILE_SIZE, x:x + TILE_SIZE] for x, y in coords]
        tiled = perf_counter()
        results = []
        for offset in range(0, len(tiles), self.tile_batch_size):
            results.extend(self._predict(tiles[offset:offset + self.tile_batch_size], base_conf, tile_iou))
        inferred = perf_counter()

        if dedup == "batched_nms":
            detections = self._nms(results, coords, width, height, class_agnostic)
        elif dedup == "cross_tile_merge":
            detections = self._merge(results, coords, width, height, merge, class_agnostic)
        else:
            raise ValueError(f"Método de deduplicación desconocido: {dedup}")

        detections = [d for d in detections if d["score"] >= thresholds[d["class_id"]]]
        end = perf_counter()
        return detections, {
            "tiles": coords,
            "tiling_ms": (tiled - start) * 1000,
            "infer_ms": (inferred - tiled) * 1000,
            "postprocess_ms": (end - inferred) * 1000,
            "total_ms": (end - start) * 1000,
        }

    @staticmethod
    def _merge(results, coords, width, height, merge, class_agnostic):
        annotations = []
        for tile_id, (result, (x, y)) in enumerate(zip(results, coords)):
            for x1, y1, x2, y2, score, cls in result.boxes.data.cpu().tolist():
                bbox = clip_bbox([x + x1, y + y1, x2 - x1, y2 - y1], width, height)
                if bbox[2] > 0 and bbox[3] > 0:
                    annotations.append({
                        "bbox": bbox, "score": score, "category_id": int(cls),
                        "_tile_id": tile_id, "_tile_coord": (x, y),
                    })
        merged, _ = deduplicate_cross_tile_annotations(
            annotations, merge, 0.40, 0.75, width, height, class_agnostic=class_agnostic,
        )
        detections = []
        for ann in merged:
            x, y, w, h = ann["bbox"]
            detections.append(_detection((x, y, x + w, y + h), ann["score"], ann["category_id"],
                                         ann["extra"]["source_tiles"]))
        return detections

    @staticmethod
    def _nms(results, coords, width, height, class_agnostic):
        import torch
        from torchvision.ops import batched_nms

        projected = []
        for result, (x, y) in zip(results, coords):
            data = result.boxes.data.float()
            shift = data.new_tensor([x, y, x, y])
            limits = data.new_tensor([width, height, width, height])
            boxes = torch.minimum((data[:, :4] + shift).clamp_min(0), limits)
            projected.append(torch.cat((boxes, data[:, 4:6]), dim=1))
        data = torch.cat(projected).float()
        batch_sizes = torch.tensor([len(boxes) for boxes in projected], device=data.device)
        tile_ids = torch.arange(len(coords), device=data.device).repeat_interleave(batch_sizes)
        valid = (data[:, 2] > data[:, 0]) & (data[:, 3] > data[:, 1])
        data, tile_ids = data[valid], tile_ids[valid]
        classes = torch.zeros_like(data[:, 5]) if class_agnostic else data[:, 5]
        keep = batched_nms(data[:, :4], data[:, 4], classes, 0.45)
        return [
            _detection(row[:4], row[4], int(row[5]), [coords[int(row[6])]])
            for row in torch.cat((data[keep], tile_ids[keep, None]), dim=1).cpu().tolist()
        ]


def _detection(box, score, cls, source_tiles):
    tiles = tuple(dict.fromkeys(source_tiles))
    return dict(bbox_xyxy=tuple(box), score=score, class_id=cls, class_name=CLASS_NAMES[cls],
                merged_count=len(tiles), source_tiles=tiles)


def render_detections(frame, detections, metrics, *, audit=False):
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    if audit:
        for x, y in metrics["tiles"]:
            cv2.rectangle(canvas, (x, y), (x + TILE_SIZE - 1, y + TILE_SIZE - 1), (110, 110, 110), 1)
            cv2.putText(canvas, f"({x}, {y})", (x + 4, y + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    for detection in detections:
        x1, y1, x2, y2 = (round(v) for v in detection["bbox_xyxy"])
        lo = (min(x1, width - 1), min(y1, height - 1))
        hi = (min(x2, width - 1), min(y2, height - 1))
        cls = detection["class_id"]
        color = (AUDIT_COLORS if audit else BOX_COLORS)[cls]
        cv2.rectangle(canvas, lo, hi, color, 3)
        if audit:
            label = f"{detection['class_name']} {detection['score']:.2f}"
            count = detection["merged_count"]
            if count > 1:
                label += f" [x{count}]"
                color = (120, 255, 50)
                cv2.rectangle(canvas, (max(0, lo[0] - 3), max(0, lo[1] - 3)),
                              (min(width - 1, hi[0] + 3), min(height - 1, hi[1] + 3)), color, 2)
            cv2.putText(canvas, label, (lo[0], max(48, lo[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    if audit:
        cv2.rectangle(canvas, (0, 0), (width - 1, 25), (30, 30, 30), -1)
        cv2.putText(canvas, f"{len(metrics['tiles'])} tiles | {len(detections)} cajas | {metrics['total_ms']:.1f} ms",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas
