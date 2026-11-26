"""Modelos por etapa y dibujo de predicciones Ultralytics."""
YOLO = None
YOLO_IMPORT_ERROR = None
torch = None

try:
    import torch
    from ultralytics import YOLO
except Exception as exc:
    YOLO_IMPORT_ERROR = exc

import cv2
import numpy as np

PROCESS_MODELS = {
    "escarmenado": {
        "label": "Escarmenado y Lavado",
        "filename": "bestS.pt",
        "task": "detect",
        "desc": "Tiling 640 × 640",
        "tiled": True,
    },
    "cardado": {
        "label": "Cardado",
        "filename": "bestC.pt",
        "task": "segment",
        "desc": "Segmentación (Máscara + BBox)",
        "class_names_override": {
            0: "cinta",
        },
        "colors": {
            0: (155, 115, 35),   # Azul petróleo (clase única: cinta)
        }
    },
    "cardado_impurezas": {
        "label": "Impurezas en Cardado",
        "filename": "bestS.pt",
        "task": "detect",
        "desc": "Tiling en filas centrales",
        "tiled": True,
        "skip_outer_rows": True,
    },
    "hilado": {
        "label": "Hilado",
        "filename": "bestH.pt",
        "task": "segment",
        "desc": "Segmentación (Máscara + BBox)",
        "colors": {
            0: (72, 150, 52),    # Verde esmeralda técnico
            1: (32, 165, 218),   # Ámbar dorado (hace match con verde)
            2: (140, 155, 60),   # Verde agua
            3: (55, 120, 165),   # Bronce suave
        }
    }
}

DEFAULT_FALLBACK_COLORS_BGR = [
    (56, 58, 176),
    (168, 108, 42),
    (72, 150, 52),
    (32, 165, 218),
]


def _get_class_color(cls_id, color_map=None):
    """Devuelve un color BGR determinístico según el mapa de la etapa activa."""
    if color_map:
        if cls_id in color_map:
            return color_map[cls_id]
        palette = list(color_map.values())
        return palette[cls_id % len(palette)]
    return DEFAULT_FALLBACK_COLORS_BGR[cls_id % len(DEFAULT_FALLBACK_COLORS_BGR)]


def draw_custom_boxes(image, results, class_names, color_map=None):
    """
    Dibuja cuadros delimitadores (bounding boxes) y etiquetas personalizadas en la imagen.
    Se conserva para compatibilidad con la lógica de detección.
    """
    return draw_custom_predictions(image, results, class_names, draw_masks=False, color_map=color_map)


def draw_custom_predictions(image, results, class_names, draw_masks=False, mask_alpha=0.36, color_map=None):
    """
    Dibuja predicciones personalizadas sobre la imagen:
    - Si draw_masks=True y existen máscaras (Cardado e Hilado), dibuja máscara semitransparente + contorno.
    - En todos los casos dibuja bounding boxes y etiquetas con fondo sólido legible.
    """
    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return image

    h, w = image.shape[:2]

    # 1. Capa de máscaras (para segmentación en Cardado e Hilado)
    if draw_masks and getattr(res, "masks", None) is not None:
        overlay = image.copy()
        polys = getattr(res.masks, "xy", None)

        if polys is not None and len(polys) == len(res.boxes):
            for idx, box in enumerate(res.boxes):
                cls_id = int(box.cls[0])
                color = _get_class_color(cls_id, color_map)
                poly = polys[idx]
                if poly is not None and len(poly) >= 3:
                    pts = np.round(poly).astype(np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, mask_alpha, image, 1.0 - mask_alpha, 0, image)

            # Contorno nítido sobre la máscara
            for idx, box in enumerate(res.boxes):
                cls_id = int(box.cls[0])
                color = _get_class_color(cls_id, color_map)
                poly = polys[idx]
                if poly is not None and len(poly) >= 3:
                    pts = np.round(poly).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], True, color, 2, cv2.LINE_AA)
        elif getattr(res.masks, "data", None) is not None:
            mask_data = res.masks.data.cpu().numpy()
            for idx, box in enumerate(res.boxes):
                if idx >= len(mask_data):
                    break
                cls_id = int(box.cls[0])
                color = _get_class_color(cls_id, color_map)
                m = cv2.resize(mask_data[idx], (w, h), interpolation=cv2.INTER_NEAREST)
                m_bool = m > 0.5
                overlay[m_bool] = color
            cv2.addWeighted(overlay, mask_alpha, image, 1.0 - mask_alpha, 0, image)

    # 2. Capa de bounding boxes y etiquetas (todos los modelos)
    for box in res.boxes:
        cls_id = int(box.cls[0])
        label = class_names.get(cls_id, str(cls_id)) if isinstance(class_names, dict) else class_names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        color = _get_class_color(cls_id, color_map)

        # Rectángulo principal
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Etiqueta con fondo sólido, evitando recorte en el borde superior
        font_scale = 0.5
        text = f'{label} {conf:.2f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        label_y1 = y1 - th - 6
        label_y2 = y1
        if label_y1 < 0:
            label_y1 = y1
            label_y2 = min(h - 1, y1 + th + 6)

        label_x2 = min(w - 1, x1 + tw + 6)
        cv2.rectangle(image, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            image,
            text,
            (x1 + 3, label_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return image


def get_detected_classes(results, class_names):
    """
    Extrae un conjunto de nombres únicos de las clases detectadas en el frame actual.
    Útil para generar reportes o estadísticas por frame.
    """
    if results[0].boxes is None:
        return []
    classes_in_frame = set()
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = class_names.get(cls_id, str(cls_id)) if isinstance(class_names, dict) else class_names[cls_id]
        classes_in_frame.add(label)
    return sorted(classes_in_frame)
