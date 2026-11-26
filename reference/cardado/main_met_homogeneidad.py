# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.11.2",
#     "numpy>=2.5.5",
#     "opencv-python>=5.1.0.95",
#     "pyqt5>=5.15.11",
#     "pyserial>=3.5",
#     "torch>=2.15.0",
#     "ultralytics>=8.4.218",
# ]
# ///
import sys
import os
import io
import gc
import time
import subprocess
import json
import glob
import platform
from dataclasses import dataclass

# PySerial es necesario solo para el control Arduino/L298N.
# Si no está instalado, la parte de visión puede seguir iniciando.
try:
    import serial
    SERIAL_IMPORT_ERROR = None
except Exception as e:
    serial = None
    SERIAL_IMPORT_ERROR = e

# Orden de imports dependiente de la plataforma:
# - Windows: PyTorch/Ultralytics primero para reducir conflictos de DLL (c10.dll).
# - Jetson/Ubuntu: PyQt5 antes de OpenCV/PyTorch para reducir conflictos TLS/OpenGL.
YOLO = None
YOLO_IMPORT_ERROR = None
torch = None
mvsdk = None

if platform.system() == "Windows":
    try:
        import torch
        from ultralytics import YOLO
    except Exception as e:
        YOLO_IMPORT_ERROR = e

    from PyQt5.QtWidgets import *
    from PyQt5.QtGui import QImage, QPixmap, QFont, QKeySequence
    from PyQt5.QtCore import QTimer, Qt, QSize

    import cv2
    import numpy as np
else:
    # Jetson / Ubuntu: mantener PyQt5 antes de OpenCV.
    from PyQt5.QtWidgets import *
    from PyQt5.QtGui import QImage, QPixmap, QFont, QKeySequence
    from PyQt5.QtCore import QTimer, Qt, QSize

    import cv2
    import numpy as np

    try:
        import torch
        from ultralytics import YOLO
    except Exception as e:
        YOLO_IMPORT_ERROR = e

CALIBRATION_PROFILE_PATH = "mindvision_color_profile.json"

DEFAULT_COLOR_PARAMS = {
    "blue_scale": 1.05,
    "green_scale": 0.80,
    "red_scale": 1.15,
    "direct_beta": 4,
    "gray_strength": 0.15,
    "gray_max_gain": 1.30,
    "gray_min_gain": 0.80,
    "vis_alpha": 1.05,
    "vis_beta": 6,
    "color_fix_enabled": True
}

# Configuración de modelos y paletas diferenciadas (BGR para OpenCV) por etapa
PROCESS_MODELS = {
    "escarmenado": {
        "label": "Escarmenado",
        "filename": "bestE.pt",
        "task": "detect",
        "desc": "Detección (BBox)",
        "colors": {
            0: (52, 60, 185),    # Rojo ladrillo
            1: (170, 110, 45),   # Azul acero
            2: (95, 70, 145),    # Ciruela sobrio
            3: (48, 128, 175),   # Terracota
        }
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

# Sub-modos seleccionables dentro de la etapa de Cardado
CARDADO_MODES = {
    "segmentacion": {
        "label": "Segmentación (Homogeneidad)",
        "filename": "bestC.pt",
        "task": "segment",
        "desc": "Segmentación (Velo + Homogeneidad)",
        "class_names_override": {
            0: "cinta",
        },
        "colors": {
            0: (155, 115, 35),   # Azul petróleo (clase única: cinta)
        },
    },
    "imperfecciones": {
        "label": "Imperfecciones (Bounding Boxes)",
        "filename": "bestS.pt",
        "task": "detect",
        "desc": "Detección de imperfecciones (BBox)",
        "colors": {
            0: (52, 60, 185),    # Rojo ladrillo
            1: (170, 110, 45),   # Azul acero
            2: (95, 70, 145),    # Ciruela sobrio
            3: (48, 128, 175),   # Terracota
        },
    },
}

DEFAULT_FALLBACK_COLORS_BGR = [
    (56, 58, 176),
    (168, 108, 42),
    (72, 150, 52),
    (32, 165, 218),
]


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as mcolors


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


@dataclass(frozen=True)
class ResultadoHomogeneidad:
    """Resultado completo del análisis de homogeneidad del velo cardado."""
    decision: str
    metodo1: str
    metodo2: str
    metodo3: str
    metodo4: str
    cv1: float
    entropia1: float
    cv2: float
    homogeneidad3: float
    cv4: float
    votos_alta: int
    votos_media: int
    votos_baja: int


class ServicioHomogeneidad:
    """
    Evalúa la homogeneidad del velo segmentado en 640x640 mediante 4 métodos
    (intensidad/entropía, perfil del velo, textura GLCM y bloques 4x4) y votación.
    """

    SIZE = 640

    def __init__(self):
        self.ultima_mascara = None
        self.ultima_gris_roi = None
        self.ultima_gris_original = None

    def extraer_mascara_yolo(self, res, frame_shape):
        """
        Extrae y combina las máscaras de segmentación de YOLO en una matriz binaria 640x640.
        Devuelve None si no hay máscaras válidas en la detección.
        """
        masks = getattr(res, "masks", None)
        if masks is None or res.boxes is None or len(res.boxes) == 0:
            return None

        h, w = frame_shape[:2]
        mask_640 = np.zeros((self.SIZE, self.SIZE), dtype=np.uint8)

        polys = getattr(masks, "xy", None)
        if polys is not None and len(polys) > 0:
            scale_x = self.SIZE / float(w)
            scale_y = self.SIZE / float(h)
            has_poly = False
            for poly in polys:
                if poly is not None and len(poly) >= 3:
                    scaled = np.empty_like(poly, dtype=np.float32)
                    scaled[:, 0] = poly[:, 0] * scale_x
                    scaled[:, 1] = poly[:, 1] * scale_y
                    pts = np.round(scaled).astype(np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask_640, [pts], 1)
                    has_poly = True
            if has_poly and np.any(mask_640 > 0):
                return mask_640.astype(np.float32)

        mask_data = getattr(masks, "data", None)
        if mask_data is not None and len(mask_data) > 0:
            data_np = mask_data.cpu().numpy()
            combined = np.any(data_np > 0.5, axis=0).astype(np.uint8)
            resized = cv2.resize(combined, (self.SIZE, self.SIZE), interpolation=cv2.INTER_NEAREST)
            if np.any(resized > 0):
                return resized.astype(np.float32)

        return None

    def analizar_con_mascara(self, frame_bgr, mascara_velo=None):
        """
        Ejecuta los 4 métodos de homogeneidad y la votación final sobre `frame_bgr`
        usando `mascara_velo` (640x640). Si `mascara_velo` es None, usa Otsu como respaldo.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        redimensionada = cv2.resize(frame_bgr, (self.SIZE, self.SIZE), interpolation=cv2.INTER_LINEAR)
        gris = self._a_gris(redimensionada)

        if mascara_velo is not None:
            if mascara_velo.shape != (self.SIZE, self.SIZE):
                mascara_velo = cv2.resize(
                    mascara_velo.astype(np.float32),
                    (self.SIZE, self.SIZE),
                    interpolation=cv2.INTER_NEAREST,
                )
            mascara_limpia = self._limpiar_mascara(mascara_velo)
        else:
            mascara_limpia = self._limpiar_mascara(self._segmentar_con_otsu(gris))

        roi_gris = self._aplicar_mascara(gris, mascara_limpia)

        self.ultima_mascara = mascara_limpia
        self.ultima_gris_roi = roi_gris
        self.ultima_gris_original = gris

        m1 = self._metodo1_intensidad_entropia(gris, mascara_limpia)
        m2 = self._metodo2_perfil_velo(mascara_limpia)
        m3 = self._metodo3_glcm(roi_gris, mascara_limpia)
        m4 = self._metodo4_bloques(roi_gris, mascara_limpia)

        votos = self._votar_final(m1["clase"], m2["clase"], m3["clase"], m4["clase"])

        return ResultadoHomogeneidad(
            decision=votos["decision"],
            metodo1=m1["clase"],
            metodo2=m2["clase"],
            metodo3=m3["clase"],
            metodo4=m4["clase"],
            cv1=m1["cv"],
            entropia1=m1["entropia"],
            cv2=m2["cv"],
            homogeneidad3=m3["homogeneidad"],
            cv4=m4["cv"],
            votos_alta=votos["altas"],
            votos_media=votos["medias"],
            votos_baja=votos["bajas"],
        )

    def _limpiar_mascara(self, mascara):
        suavizada = self._filtro_mayoria(mascara, minimo_vecinos=4)
        cerrada = self._erosionar(self._dilatar(suavizada))
        principal = self._componente_principal(cerrada)
        return principal if principal is not None else mascara

    def _filtro_mayoria(self, mascara, minimo_vecinos=4):
        binary = (mascara > 0.5).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        vecinos = cv2.filter2D(binary, cv2.CV_16U, kernel, borderType=cv2.BORDER_CONSTANT)
        salida = (vecinos >= minimo_vecinos).astype(np.float32)
        salida[0, :] = 0.0
        salida[-1, :] = 0.0
        salida[:, 0] = 0.0
        salida[:, -1] = 0.0
        return salida

    def _dilatar(self, mascara):
        binary = (mascara > 0.5).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        salida = cv2.dilate(binary, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
        salida[0, :] = 0.0
        salida[-1, :] = 0.0
        salida[:, 0] = 0.0
        salida[:, -1] = 0.0
        return salida

    def _erosionar(self, mascara):
        binary = (mascara > 0.5).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        salida = cv2.erode(binary, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
        salida[0, :] = 0.0
        salida[-1, :] = 0.0
        salida[:, 0] = 0.0
        salida[:, -1] = 0.0
        return salida

    def _componente_principal(self, mascara):
        binary = (mascara > 0.5).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)
        if num_labels <= 1:
            return None

        areas = stats[1:, cv2.CC_STAT_AREA]
        mejor_idx = int(np.argmax(areas)) + 1
        mejor_tamano = int(stats[mejor_idx, cv2.CC_STAT_AREA])
        if mejor_tamano < 500:
            return None

        return (labels == mejor_idx).astype(np.float32)

    def _segmentar_con_otsu(self, gris):
        idx = np.clip(np.rint(gris / 256.0 * 255.0), 0, 255).astype(np.int32).ravel()
        histograma = np.bincount(idx, minlength=256).astype(np.float64)
        total = float(gris.size)
        suma_total = float(np.dot(np.arange(256, dtype=np.float64), histograma))

        mejor_var = 0.0
        mejor_umbral = 127
        peso0 = 0.0
        suma0 = 0.0

        for t in range(256):
            peso0 += histograma[t]
            if peso0 == 0.0:
                continue
            peso1 = total - peso0
            if peso1 == 0.0:
                break

            suma0 += t * histograma[t]
            media0 = suma0 / peso0
            media1 = (suma_total - suma0) / peso1
            varianza = peso0 * peso1 * (media0 - media1) ** 2
            if varianza > mejor_var:
                mejor_var = varianza
                mejor_umbral = t

        limite = mejor_umbral * 256.0 / 255.0
        return (gris > limite).astype(np.float32)

    def _a_gris(self, imagen_bgr):
        b = imagen_bgr[:, :, 0].astype(np.float32)
        g = imagen_bgr[:, :, 1].astype(np.float32)
        r = imagen_bgr[:, :, 2].astype(np.float32)
        return 0.299 * r + 0.587 * g + 0.114 * b

    def _aplicar_mascara(self, gris, mascara):
        return np.where(mascara > 0.5, gris, 0.0).astype(np.float32)

    def _metodo1_intensidad_entropia(self, gris, mascara):
        valores = gris[mascara > 0.5]
        if valores.size == 0:
            return {"cv": 0.0, "entropia": 0.0, "clase": "BAJA HOMOGENEIDAD"}

        valores_f64 = valores.astype(np.float64)
        media = float(np.mean(valores_f64))
        varianza = float(np.mean((valores_f64 - media) ** 2))
        std = float(np.sqrt(varianza))
        cv = std / media if media > 0.0 else 0.0

        idx = np.clip(valores_f64, 0.0, 255.0).astype(np.int32)
        histograma = np.bincount(idx, minlength=256)
        probs = histograma[histograma > 0].astype(np.float64) / float(valores_f64.size)
        entropia = float(-np.sum(probs * np.log(probs))) + 0.0

        if cv < 0.10 and entropia < 4.0:
            clase = "ALTA HOMOGENEIDAD"
        elif cv < 0.15 and entropia < 5.0:
            clase = "MEDIA HOMOGENEIDAD"
        else:
            clase = "BAJA HOMOGENEIDAD"

        return {"cv": cv, "entropia": entropia, "clase": clase}

    def _metodo2_perfil_velo(self, mascara):
        perfil = np.sum(mascara > 0.5, axis=0).astype(np.float64)
        media = float(np.mean(perfil))
        std = float(np.sqrt(np.mean((perfil - media) ** 2)))
        cv = std / media if media > 0.0 else 0.0

        if cv < 0.10:
            clase = "ALTA HOMOGENEIDAD"
        elif cv < 0.20:
            clase = "MEDIA HOMOGENEIDAD"
        else:
            clase = "BAJA HOMOGENEIDAD"

        return {"cv": cv, "clase": clase}

    def _metodo3_glcm(self, roi_gris, mascara):
        cuantizada = np.clip(roi_gris / 4.0, 0.0, 63.0).astype(np.int32)
        distancia = 5
        valid = (mascara[:, :-distancia] > 0.5) & (mascara[:, distancia:] > 0.5)
        i_vals = cuantizada[:, :-distancia][valid]
        j_vals = cuantizada[:, distancia:][valid]

        cuenta = int(i_vals.size * 2)
        if cuenta == 0:
            return {"homogeneidad": 0.0, "clase": "BAJA HOMOGENEIDAD"}

        n_niveles = 64
        flat_ij = i_vals * n_niveles + j_vals
        flat_ji = j_vals * n_niveles + i_vals
        glcm = (
            np.bincount(flat_ij, minlength=n_niveles * n_niveles)
            + np.bincount(flat_ji, minlength=n_niveles * n_niveles)
        ).reshape(n_niveles, n_niveles).astype(np.float64)
        glcm /= float(cuenta)

        ii, jj = np.indices((n_niveles, n_niveles))
        homogeneidad = float(np.sum(glcm / (1.0 + np.abs(ii - jj))))

        if homogeneidad > 0.85:
            clase = "ALTA HOMOGENEIDAD"
        elif homogeneidad > 0.70:
            clase = "MEDIA HOMOGENEIDAD"
        else:
            clase = "BAJA HOMOGENEIDAD"

        return {"homogeneidad": homogeneidad, "clase": clase}

    def _metodo4_bloques(self, roi_gris, mascara):
        n = 4
        tamano = self.SIZE // n
        medias = []

        for bi in range(n):
            y1, y2 = bi * tamano, (bi + 1) * tamano
            for bj in range(n):
                x1, x2 = bj * tamano, (bj + 1) * tamano
                block_mask = mascara[y1:y2, x1:x2] > 0.5
                cuenta = int(np.count_nonzero(block_mask))
                if cuenta > 10:
                    medias.append(float(np.mean(roi_gris[y1:y2, x1:x2][block_mask])))

        if not medias:
            return {"cv": 0.0, "clase": "BAJA HOMOGENEIDAD"}

        medias_arr = np.asarray(medias, dtype=np.float64)
        media = float(np.mean(medias_arr))
        std = float(np.sqrt(np.mean((medias_arr - media) ** 2)))
        cv = std / media if media > 0.0 else 0.0

        if cv < 0.08:
            clase = "ALTA HOMOGENEIDAD"
        elif cv < 0.15:
            clase = "MEDIA HOMOGENEIDAD"
        else:
            clase = "BAJA HOMOGENEIDAD"

        return {"cv": cv, "clase": clase}

    def _votar_final(self, r1, r2, r3, r4):
        resultados = (r1, r2, r3, r4)
        altas = sum(1 for r in resultados if "ALTA" in r)
        medias = sum(1 for r in resultados if "MEDIA" in r)
        bajas = sum(1 for r in resultados if "BAJA" in r)

        if bajas >= 2:
            decision = "BAJA HOMOGENEIDAD"
        elif bajas == 0 and altas >= 2:
            decision = "HOMOGÉNEO"
        else:
            decision = "MEDIA HOMOGENEIDAD"

        return {"decision": decision, "altas": altas, "medias": medias, "bajas": bajas}


def gray_world_color_correction_bgr(frame, strength=0.65, max_gain=2.2, min_gain=0.45):
    """
    Corrección automática tipo Gray-World para reducir dominantes de color.
    Trabaja en BGR. Es más segura que ganancias fijas porque calcula la corrección
    en cada frame y limita los factores para evitar imagen negra o saturada.
    """
    if frame is None or frame.size == 0:
        return frame

    img = frame.astype(np.float32)

    means = img.mean(axis=(0, 1))
    global_mean = float(means.mean())

    if global_mean < 1.0 or np.any(means < 1.0):
        return frame

    gains = global_mean / means
    gains = np.clip(gains, min_gain, max_gain)

    corrected = img * gains.reshape(1, 1, 3)
    corrected = (1.0 - strength) * img + strength * corrected

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected


def enhance_visibility_bgr(frame, alpha=1.15, beta=8):
    """
    Pequeño ajuste de contraste y brillo para evitar que la imagen quede oscura.
    alpha controla contraste, beta brillo.
    """
    if frame is None:
        return frame
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def suppress_green_cast_bgr(frame, green_scale=0.52, red_scale=1.55, blue_scale=1.25, beta=10):
    """
    Corrección específica para dominante verde persistente.
    Orden de canales: B, G, R.
    """
    if frame is None or frame.size == 0:
        return frame

    img = frame.astype(np.float32)

    img[:, :, 0] *= blue_scale
    img[:, :, 1] *= green_scale
    img[:, :, 2] *= red_scale

    img += beta
    img = np.clip(img, 0, 255)

    return img.astype(np.uint8)


class USBCamera:
    """
    Adaptador simple para una cámara UVC/V4L2 usando OpenCV.
    Mantiene la misma interfaz open()/isOpened()/read()/release() que usaba
    la cámara MindVision para no alterar el resto de la aplicación.
    """

    def __init__(self, device="/dev/video0", width=800, height=600, fps=30, fourcc="YUYV"):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.cap = None
        self.opened = False

    def _configure_v4l2(self):
        """Fija exposición a ~8.3 ms y anti-flicker a 60 Hz en Linux/Jetson."""
        if platform.system() != "Linux":
            return
        try:
            subprocess.run([
                "v4l2-ctl", "-d", str(self.device),
                "--set-ctrl=exposure_auto=1",
                "--set-ctrl=exposure_absolute=83",
                "--set-ctrl=power_line_frequency=2"
            ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Aviso: no se pudo aplicar configuración V4L2: {e}")

    def open(self):
        self._configure_v4l2()
        if platform.system() == "Windows":
            cam_idx = 0 if str(self.device).startswith("/dev/") else int(self.device)
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara USB en {self.device}.")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.cap.release()
            self.cap = None
            raise RuntimeError("La cámara abrió, pero no entregó frames.")

        self.opened = True
        print(
            f"Cámara USB abierta: {self.device} | "
            f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
            f"{self.cap.get(cv2.CAP_PROP_FPS):.1f} FPS | {self.fourcc}"
        )
        return True

    def isOpened(self):
        return self.opened and self.cap is not None and self.cap.isOpened()

    def read(self):
        if not self.isOpened():
            return False, None
        return self.cap.read()

    def capture_raw_frame_for_calibration(self, timeout_ms=1000):
        """Compatibilidad con el método de calibración de referencia blanca."""
        return self.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.opened = False
        print("Cámara USB liberada.")


class StreamToLogger(io.TextIOBase):
    """
    Clase utilitaria para redirigir flujos de salida (stdout/stderr) a un widget de la GUI.
    Permite ver los print() y errores directamente en la interfaz de la aplicación.
    """

    def __init__(self, log_function):
        self.log_function = log_function
        self.line_buffer = ''

    def write(self, text):
        for char in text:
            if char == '\n':
                if self.line_buffer.strip():
                    self.log_function(self.line_buffer.strip())
                self.line_buffer = ''
            else:
                self.line_buffer += char
        return len(text)

    def flush(self):
        if self.line_buffer.strip():
            self.log_function(self.line_buffer.strip())
            self.line_buffer = ''


class MainWindow(QMainWindow):
    """
    Ventana principal responsiva y táctil de la aplicación de inspección de fibra de alpaca.
    Se adapta automáticamente a pantallas pequeñas (empotradas/Jetson), medianas y grandes,
    soporta desplazamiento táctil con el dedo y permite plegar el panel desde la parte superior.
    """

    def __init__(self):
        super().__init__()

        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # --- Parámetros de calibración por defecto ---
        self.params = DEFAULT_COLOR_PARAMS.copy()
        self.color_fix_enabled = bool(self.params.get("color_fix_enabled", True))

        # --- Estado de Modelo (carga bajo demanda) ---
        self.model = None
        self.current_stage_key = None
        self.current_cardado_mode = "segmentacion"
        self.loaded_model_key = None
        self.loaded_cardado_mode = None
        self.loaded_model_id = None
        self.infer_conf = 0.50
        self.infer_imgsz = 416
        self.homogeneity_service = ServicioHomogeneidad()

        # --- Estado temporal para pruebas con imágenes ---
        self.test_image_bgr = None
        self.test_image_path = None
        self.last_rendered_pixmap = None

        # --- Estado responsivo y panel plegable ---
        self._layout_mode = None
        self._sidebar_visible = True
        self._saved_sidebar_width = 330
        self.logo_left_pixmap = None
        self.logo_right_pixmap = None

        # --- Configuración Inicial Adaptativa ---
        self.setWindowTitle("Sistema de Inspección de Fibra de Alpaca")
        self.setMinimumSize(640, 420)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)

        self._init_ui()
        self._enable_touch_scrolling()
        self._apply_styles()

        # Atajos útiles en pantallas pequeñas empotradas
        self.fullscreen_shortcut = QShortcut(QKeySequence("F11"), self)
        self.fullscreen_shortcut.activated.connect(self._toggle_fullscreen)
        self.sidebar_shortcut = QShortcut(QKeySequence("Ctrl+B"), self)
        self.sidebar_shortcut.activated.connect(self._toggle_sidebar)

        self.cap = None

        # --- Control de motor Arduino/L298N ---
        self.serial_port = None
        self.motor_speed = 90
        self.motor_running = False
        self.motor_direction = "FORWARD"

        self.motor_serial_timer = QTimer(self)
        self.motor_serial_timer.timeout.connect(self._read_motor_serial)
        self.motor_serial_timer.start(50)

        # Timer para el loop de procesamiento de video (~30 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Redirección de consola al widget de log en la GUI
        sys.stdout = StreamToLogger(self.log)
        sys.stderr = StreamToLogger(self.log)

        # Seleccionar la primera etapa por defecto (sin cargar hasta usarla)
        self.select_process_stage("escarmenado", load_now=False)

        # Ajustar tamaño inicial según la pantalla detectada
        self._configure_initial_window_geometry()

    def _enable_touch_scrolling(self):
        """Habilita desplazamiento táctil suave con el dedo en el panel lateral y el log."""
        try:
            for viewport in (self.sidebar_scroll.viewport(), self.log_area.viewport()):
                viewport.setAttribute(Qt.WA_AcceptTouchEvents, True)
                QScroller.grabGesture(viewport, QScroller.LeftMouseButtonGesture)
                scroller = QScroller.scroller(viewport)
                props = scroller.scrollerProperties()
                props.setScrollMetric(QScrollerProperties.OvershootDragResistanceFactor, 0.35)
                props.setScrollMetric(QScrollerProperties.OvershootScrollDistanceFactor, 0.20)
                props.setScrollMetric(QScrollerProperties.DecelerationFactor, 0.65)
                scroller.setScrollerProperties(props)
        except Exception as e:
            print(f"Aviso: no se pudo activar gesto táctil en scroll: {e}")

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _toggle_sidebar(self):
        """Oculta o despliega el panel lateral de configuración para ampliar el visor."""
        if self._sidebar_visible:
            sizes = self.main_splitter.sizes()
            if len(sizes) == 2 and sizes[1] >= 200:
                self._saved_sidebar_width = sizes[1]
            self.sidebar_scroll.setVisible(False)
            self._sidebar_visible = False
            self.toggle_sidebar_btn.setText("❮")
            self.toggle_sidebar_btn.setToolTip("Mostrar panel de configuración (Ctrl+B)")
        else:
            self.sidebar_scroll.setVisible(True)
            self._sidebar_visible = True
            self.toggle_sidebar_btn.setText("❯")
            self.toggle_sidebar_btn.setToolTip("Ocultar panel de configuración (Ctrl+B)")
            total_w = max(self.main_splitter.width(), self.width() - 24)
            sidebar_w = max(250, min(self._saved_sidebar_width, 460))
            video_w = max(320, total_w - sidebar_w)
            self.main_splitter.setSizes([video_w, sidebar_w])

        # Reescalar imagen en el visor en el siguiente ciclo de layout
        QTimer.singleShot(0, self._render_last_pixmap)

    def _configure_initial_window_geometry(self):
        """Adapta el tamaño inicial de la ventana al monitor actual."""
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            # En pantallas pequeñas/empotradas (ej. 800x480, 1024x600, 1280x720) abrir maximizada
            if avail.width() <= 1366 or avail.height() <= 768:
                self.setGeometry(avail)
                self._apply_responsive_layout(avail.width(), avail.height())
                self.showMaximized()
                return

            # En pantallas medianas o grandes, usar 90% del área útil centrada
            w = min(int(avail.width() * 0.90), 1760)
            h = min(int(avail.height() * 0.90), 980)
            x = avail.x() + (avail.width() - w) // 2
            y = avail.y() + (avail.height() - h) // 2
            self.setGeometry(x, y, w, h)
            self._apply_responsive_layout(w, h)
            self.show()
        else:
            self.resize(1280, 720)
            self._apply_responsive_layout(1280, 720)
            self.show()

    def _make_divider(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setObjectName("divider")
        return line

    def _load_logo_pixmap(self, filename):
        for path in (os.path.join(self.base_dir, filename), filename):
            if os.path.exists(path):
                pm = QPixmap(path)
                if not pm.isNull():
                    return pm
        return None

    def _init_ui(self):
        """Inicializa la estructura y disposición responsiva y táctil de los widgets."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Barra Superior Plana ---
        self.title_bar = QWidget()
        self.title_bar.setObjectName("titleBar")
        self.title_bar_layout = QHBoxLayout(self.title_bar)
        self.title_bar_layout.setContentsMargins(16, 6, 16, 6)
        self.title_bar_layout.setSpacing(10)

        self.logo_left_pixmap = self._load_logo_pixmap("logo_left.png")
        self.logo_right_pixmap = self._load_logo_pixmap("logo_right.png")

        self.logo_left = QLabel()
        self.logo_left.setObjectName("logoBox")
        if self.logo_left_pixmap is None:
            self.logo_left.setText("Logo Izquierdo")

        self.title_label = QLabel("Sistema de Inspección de Fibra de Alpaca")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setWordWrap(True)

        self.logo_right = QLabel()
        self.logo_right.setObjectName("logoBox")
        if self.logo_right_pixmap is None:
            self.logo_right.setText("Logo Derecho")

        # Botón de pantalla completa para pantallas táctiles/empotradas sin teclado
        self.toggle_fullscreen_btn = QPushButton("⛶")
        self.toggle_fullscreen_btn.setObjectName("fullscreenBtn")
        self.toggle_fullscreen_btn.setFixedSize(38, 32)
        self.toggle_fullscreen_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_fullscreen_btn.setToolTip("Pantalla completa (F11)")
        self.toggle_fullscreen_btn.clicked.connect(self._toggle_fullscreen)

        # Pequeña barrita superior para plegar/desplegar el panel de configuración
        self.toggle_sidebar_btn = QPushButton("❯")
        self.toggle_sidebar_btn.setObjectName("toggleSidebarBtn")
        self.toggle_sidebar_btn.setFixedSize(44, 32)
        self.toggle_sidebar_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_sidebar_btn.setToolTip("Ocultar panel de configuración (Ctrl+B)")
        self.toggle_sidebar_btn.clicked.connect(self._toggle_sidebar)

        self.title_bar_layout.addWidget(self.logo_left)
        self.title_bar_layout.addStretch(1)
        self.title_bar_layout.addWidget(self.title_label, 2)
        self.title_bar_layout.addStretch(1)
        self.title_bar_layout.addWidget(self.logo_right)
        self.title_bar_layout.addWidget(self.toggle_fullscreen_btn, 0, Qt.AlignVCenter)
        self.title_bar_layout.addWidget(self.toggle_sidebar_btn, 0, Qt.AlignVCenter)

        # --- Área de Contenido Principal con QSplitter adaptable ---
        content_area = QWidget()
        content_area.setObjectName("contentArea")
        self.content_layout = QHBoxLayout(content_area)
        self.content_layout.setContentsMargins(12, 12, 12, 12)
        self.content_layout.setSpacing(0)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(10)
        self.main_splitter.splitterMoved.connect(self._on_splitter_moved)

        # Visor de cámara (Izquierda) - sin mínimos rígidos que rompan pantallas pequeñas
        self.video_label = QLabel("Seleccione una etapa e inicie la detección o cargue una imagen de prueba")
        self.video_label.setObjectName("videoLabel")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setWordWrap(True)
        self.video_label.setMinimumSize(320, 220)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.main_splitter.addWidget(self.video_label)

        # Panel lateral derecho con scroll suave y táctil
        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setObjectName("sidebarScroll")
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sidebar_scroll.setMinimumWidth(250)
        self.sidebar_scroll.setMaximumWidth(460)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        self.sidebar_layout = QVBoxLayout(sidebar)
        self.sidebar_layout.setContentsMargins(14, 14, 14, 14)
        self.sidebar_layout.setSpacing(10)

        # --- Panel de Selección de Etapa / Modelo ---
        model_title = QLabel("Etapa del proceso")
        model_title.setObjectName("sectionLabel")
        self.sidebar_layout.addWidget(model_title)

        stage_row = QHBoxLayout()
        stage_row.setSpacing(5)
        self.stage_buttons = {}
        for key, cfg in PROCESS_MODELS.items():
            btn = QPushButton(cfg["label"])
            btn.setCheckable(True)
            btn.setObjectName("stageBtn")
            btn.setMinimumHeight(36)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, k=key: self.select_process_stage(k, load_now=True))
            self.stage_buttons[key] = btn
            stage_row.addWidget(btn)
        self.sidebar_layout.addLayout(stage_row)

        # Barra desplegable exclusiva de Cardado (Segmentación / Homogeneidad vs Imperfecciones BB)
        self.cardado_mode_container = QWidget()
        cardado_mode_layout = QHBoxLayout(self.cardado_mode_container)
        cardado_mode_layout.setContentsMargins(0, 0, 0, 0)
        cardado_mode_layout.setSpacing(6)
        cardado_mode_lbl = QLabel("Modo Cardado")
        cardado_mode_lbl.setObjectName("mutedLabel")
        self.combo_cardado_mode = QComboBox()
        self.combo_cardado_mode.setMinimumHeight(34)
        for mode_key, mode_cfg in CARDADO_MODES.items():
            self.combo_cardado_mode.addItem(mode_cfg["label"], mode_key)
        self.combo_cardado_mode.currentIndexChanged.connect(self._on_cardado_mode_changed)
        cardado_mode_layout.addWidget(cardado_mode_lbl)
        cardado_mode_layout.addWidget(self.combo_cardado_mode, 1)
        self.cardado_mode_container.setVisible(False)
        self.sidebar_layout.addWidget(self.cardado_mode_container)

        self.model_info_label = QLabel("Modelo: sin cargar")
        self.model_info_label.setObjectName("monoMuted")
        self.model_info_label.setWordWrap(True)
        self.sidebar_layout.addWidget(self.model_info_label)

        # --- Parámetros graduables de inferencia (compartidos) ---
        params_row = QHBoxLayout()
        params_row.setSpacing(5)

        conf_lbl = QLabel("Conf.")
        conf_lbl.setObjectName("mutedLabel")

        self.btn_conf_minus = QPushButton("-")
        self.btn_conf_minus.setObjectName("touchStepBtn")
        self.btn_conf_minus.setFixedSize(30, 34)
        self.btn_conf_minus.setCursor(Qt.PointingHandCursor)
        self.btn_conf_minus.clicked.connect(lambda: self._step_conf(-0.05))

        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.05, 0.95)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setValue(self.infer_conf)
        self.spin_conf.setMinimumHeight(34)
        self.spin_conf.setKeyboardTracking(False)
        self.spin_conf.valueChanged.connect(self._on_infer_params_changed)

        self.btn_conf_plus = QPushButton("+")
        self.btn_conf_plus.setObjectName("touchStepBtn")
        self.btn_conf_plus.setFixedSize(30, 34)
        self.btn_conf_plus.setCursor(Qt.PointingHandCursor)
        self.btn_conf_plus.clicked.connect(lambda: self._step_conf(0.05))

        imgsz_lbl = QLabel("Imgsz")
        imgsz_lbl.setObjectName("mutedLabel")
        self.combo_imgsz = QComboBox()
        for sz in [320, 416, 512, 640, 800, 960, 1024, 1280, 1600]:
            self.combo_imgsz.addItem(str(sz), sz)
        self.combo_imgsz.setCurrentText(str(self.infer_imgsz))
        self.combo_imgsz.setMinimumHeight(34)
        self.combo_imgsz.currentIndexChanged.connect(self._on_infer_params_changed)

        params_row.addWidget(conf_lbl)
        params_row.addWidget(self.btn_conf_minus)
        params_row.addWidget(self.spin_conf, 1)
        params_row.addWidget(self.btn_conf_plus)
        params_row.addWidget(imgsz_lbl)
        params_row.addWidget(self.combo_imgsz, 1)
        self.sidebar_layout.addLayout(params_row)

        # Botón principal de cámara en tiempo real
        self.detect_button = QPushButton("Iniciar detección")
        self.detect_button.setObjectName("detectButton")
        self.detect_button.setMinimumHeight(42)
        self.detect_button.setCursor(Qt.PointingHandCursor)
        self.detect_button.clicked.connect(self.start_detection)
        self.sidebar_layout.addWidget(self.detect_button)

        # === MÓDULO TEMPORAL DE PRUEBA CON IMÁGENES (UI) ===
        test_img_row = QHBoxLayout()
        test_img_row.setSpacing(6)
        self.btn_test_image = QPushButton("Probar imagen...")
        self.btn_test_image.setMinimumHeight(36)
        self.btn_test_image.setCursor(Qt.PointingHandCursor)
        self.btn_test_image.clicked.connect(self._open_test_image)

        self.btn_clear_test_image = QPushButton("Limpiar")
        self.btn_clear_test_image.setMinimumHeight(36)
        self.btn_clear_test_image.setCursor(Qt.PointingHandCursor)
        self.btn_clear_test_image.clicked.connect(self._clear_test_image)
        self.btn_clear_test_image.setEnabled(False)

        test_img_row.addWidget(self.btn_test_image, 2)
        test_img_row.addWidget(self.btn_clear_test_image, 1)
        self.sidebar_layout.addLayout(test_img_row)
        # === FIN MÓDULO TEMPORAL DE PRUEBA CON IMÁGENES (UI) ===

        self.sidebar_layout.addWidget(self._make_divider())

        # Estado de la fibra / Homogeneidad
        self.status_section_label = QLabel("Estado de la fibra")
        self.status_section_label.setObjectName("sectionLabel")
        self.sidebar_layout.addWidget(self.status_section_label)

        self.status_label = QLabel("ESPERANDO")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.sidebar_layout.addWidget(self.status_label)

        # Barra horizontal (Impurezas en BB / Homogeneidad en Segmentación de Cardado)
        gauge_header = QHBoxLayout()
        self.gauge_title_label = QLabel("Nivel de impurezas")
        self.gauge_title_label.setObjectName("mutedLabel")
        self.gauge_value_label = QLabel("0 / 60")
        self.gauge_value_label.setObjectName("monoValue")
        gauge_header.addWidget(self.gauge_title_label)
        gauge_header.addStretch()
        gauge_header.addWidget(self.gauge_value_label)
        self.sidebar_layout.addLayout(gauge_header)

        self.gauge_figure = Figure(figsize=(3.5, 0.5), dpi=100)
        self.gauge_figure.patch.set_facecolor("#FFFFFF")
        self.gauge_canvas = FigureCanvas(self.gauge_figure)
        self.gauge_canvas.setFixedHeight(24)
        self.sidebar_layout.addWidget(self.gauge_canvas)

        # Panel de métricas detalladas de homogeneidad (visible solo en Cardado -> Segmentación)
        self.homogeneity_panel = QFrame()
        self.homogeneity_panel.setObjectName("homogeneityPanel")
        homog_layout = QVBoxLayout(self.homogeneity_panel)
        homog_layout.setContentsMargins(8, 7, 8, 7)
        homog_layout.setSpacing(4)

        self.homog_votes_label = QLabel("Votos: Alta 0 · Media 0 · Baja 0")
        self.homog_votes_label.setObjectName("monoValue")
        homog_layout.addWidget(self.homog_votes_label)

        self.homog_m1_label = QLabel("M1 Int/Ent: --")
        self.homog_m2_label = QLabel("M2 Perfil:  --")
        self.homog_m3_label = QLabel("M3 GLCM:    --")
        self.homog_m4_label = QLabel("M4 Bloques: --")
        for lbl in (self.homog_m1_label, self.homog_m2_label, self.homog_m3_label, self.homog_m4_label):
            lbl.setObjectName("monoMuted")
            lbl.setWordWrap(True)
            homog_layout.addWidget(lbl)

        self.homogeneity_panel.setVisible(False)
        self.sidebar_layout.addWidget(self.homogeneity_panel)

        self.sidebar_layout.addWidget(self._make_divider())

        # Control de motor Arduino/L298N
        motor_title = QLabel("Control de motor")
        motor_title.setObjectName("sectionLabel")
        self.sidebar_layout.addWidget(motor_title)

        port_row = QHBoxLayout()
        port_row.setSpacing(5)
        self.motor_port_combo = QComboBox()
        self.motor_port_combo.setMinimumHeight(34)
        self.motor_refresh_button = QPushButton("Buscar")
        self.motor_refresh_button.setMinimumHeight(34)
        self.motor_connect_button = QPushButton("Conectar")
        self.motor_connect_button.setMinimumHeight(34)
        self.motor_refresh_button.clicked.connect(self._find_motor_ports)
        self.motor_connect_button.clicked.connect(self._toggle_motor_connection)
        port_row.addWidget(self.motor_port_combo, 1)
        port_row.addWidget(self.motor_refresh_button)
        port_row.addWidget(self.motor_connect_button)
        self.sidebar_layout.addLayout(port_row)

        self.motor_connection_label = QLabel("Arduino: desconectado")
        self.motor_connection_label.setObjectName("mutedLabel")
        self.sidebar_layout.addWidget(self.motor_connection_label)

        run_row = QHBoxLayout()
        run_row.setSpacing(6)
        self.motor_start_button = QPushButton("Iniciar motor")
        self.motor_stop_button = QPushButton("Detener motor")
        self.motor_start_button.setObjectName("motorActionBtn")
        self.motor_stop_button.setObjectName("motorStopBtn")
        self.motor_start_button.setMinimumHeight(36)
        self.motor_stop_button.setMinimumHeight(36)
        self.motor_start_button.clicked.connect(self._motor_start)
        self.motor_stop_button.clicked.connect(self._motor_stop)
        run_row.addWidget(self.motor_start_button)
        run_row.addWidget(self.motor_stop_button)
        self.sidebar_layout.addLayout(run_row)

        direction_row = QHBoxLayout()
        direction_row.setSpacing(6)
        self.motor_reverse_button = QPushButton("Reversa")
        self.motor_forward_button = QPushButton("Adelante")
        self.motor_reverse_button.setMinimumHeight(36)
        self.motor_forward_button.setMinimumHeight(36)
        self.motor_reverse_button.clicked.connect(self._motor_reverse)
        self.motor_forward_button.clicked.connect(self._motor_forward)
        direction_row.addWidget(self.motor_reverse_button)
        direction_row.addWidget(self.motor_forward_button)
        self.sidebar_layout.addLayout(direction_row)

        speed_row = QHBoxLayout()
        speed_title = QLabel("Velocidad")
        speed_title.setObjectName("mutedLabel")
        self.motor_speed_label = QLabel("PWM: 90  |  35 %")
        self.motor_speed_label.setObjectName("monoValue")
        speed_row.addWidget(speed_title)
        speed_row.addStretch()
        speed_row.addWidget(self.motor_speed_label)
        self.sidebar_layout.addLayout(speed_row)

        self.motor_speed_slider = QSlider(Qt.Horizontal)
        self.motor_speed_slider.setRange(0, 255)
        self.motor_speed_slider.setValue(90)
        self.motor_speed_slider.setMinimumHeight(32)
        self.motor_speed_slider.valueChanged.connect(self._motor_speed_changed)
        self.sidebar_layout.addWidget(self.motor_speed_slider)

        self.motor_response_label = QLabel("Respuesta Arduino: sin datos")
        self.motor_response_label.setObjectName("monoMuted")
        self.motor_response_label.setWordWrap(True)
        self.sidebar_layout.addWidget(self.motor_response_label)

        self._find_motor_ports()

        self.sidebar_layout.addWidget(self._make_divider())

        # Registro de eventos
        log_title = QLabel("Registro de eventos")
        log_title.setObjectName("sectionLabel")
        self.sidebar_layout.addWidget(log_title)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMinimumHeight(95)
        self.sidebar_layout.addWidget(self.log_area, 1)

        self.sidebar_scroll.setWidget(sidebar)
        self.main_splitter.addWidget(self.sidebar_scroll)
        self.main_splitter.setStretchFactor(0, 4)
        self.main_splitter.setStretchFactor(1, 1)

        self.content_layout.addWidget(self.main_splitter, 1)

        main_layout.addWidget(self.title_bar)
        main_layout.addWidget(content_area, 1)

        self.update_status_label(0)
        self.update_gauge_chart(0)

    def _step_conf(self, delta):
        new_val = round(max(0.05, min(0.95, self.spin_conf.value() + delta)), 2)
        self.spin_conf.setValue(new_val)

    def _on_splitter_moved(self, pos, index):
        if self._sidebar_visible:
            sizes = self.main_splitter.sizes()
            if len(sizes) == 2 and sizes[1] >= 200:
                self._saved_sidebar_width = sizes[1]
        if not self.timer.isActive():
            self._render_last_pixmap()

    def _apply_responsive_layout(self, width, height):
        """Ajusta proporciones de cabecera, logos, márgenes y panel según el tamaño de pantalla."""
        if width < 1080 or height < 650:
            mode = "compact"
        elif width < 1480:
            mode = "medium"
        else:
            mode = "large"

        if mode == self._layout_mode:
            return
        self._layout_mode = mode

        if mode == "compact":
            bar_h = 52
            logo_l_size = QSize(100, 32)
            logo_r_size = QSize(90, 32)
            title_px = 13
            margin = 8
            side_margin = 10
            side_spacing = 8
            sidebar_w = 285
            log_min_h = 80
            toggle_size = QSize(42, 34)
            fs_size = QSize(36, 34)
        elif mode == "medium":
            bar_h = 60
            logo_l_size = QSize(170, 44)
            logo_r_size = QSize(155, 44)
            title_px = 16
            margin = 12
            side_margin = 13
            side_spacing = 9
            sidebar_w = 330
            log_min_h = 105
            toggle_size = QSize(44, 32)
            fs_size = QSize(38, 32)
        else:
            bar_h = 74
            logo_l_size = QSize(240, 56)
            logo_r_size = QSize(220, 56)
            title_px = 19
            margin = 16
            side_margin = 16
            side_spacing = 11
            sidebar_w = 380
            log_min_h = 130
            toggle_size = QSize(46, 34)
            fs_size = QSize(40, 34)

        self.title_bar.setFixedHeight(bar_h)
        self.title_bar_layout.setContentsMargins(margin, 4, margin, 4)
        self.title_label.setStyleSheet(
            f"font-size: {title_px}px; font-weight: 600; color: #1A1A1A; background: transparent;"
        )
        self.toggle_sidebar_btn.setFixedSize(toggle_size)
        self.toggle_fullscreen_btn.setFixedSize(fs_size)

        if self.logo_left_pixmap is not None:
            self.logo_left.setPixmap(
                self.logo_left_pixmap.scaled(logo_l_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        if self.logo_right_pixmap is not None:
            self.logo_right.setPixmap(
                self.logo_right_pixmap.scaled(logo_r_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

        self.content_layout.setContentsMargins(margin, margin, margin, margin)
        self.sidebar_layout.setContentsMargins(side_margin, side_margin, side_margin, side_margin)
        self.sidebar_layout.setSpacing(side_spacing)
        self.log_area.setMinimumHeight(log_min_h)
        self._saved_sidebar_width = sidebar_w

        # Repartir espacio inicial entre visor y barra lateral solo si está visible
        if self._sidebar_visible:
            video_w = max(320, width - sidebar_w - (margin * 2) - 10)
            self.main_splitter.setSizes([video_w, sidebar_w])

    # ------------------------------------------------------------------
    # GESTIÓN DE MODELOS BAJO DEMANDA Y PARÁMETROS DE INFERENCIA
    # ------------------------------------------------------------------
    def _get_stage_config(self, stage_key=None, cardado_mode=None):
        key = stage_key or self.current_stage_key or "escarmenado"
        base_cfg = PROCESS_MODELS.get(key, {})
        if key == "cardado":
            mode = cardado_mode or self.current_cardado_mode or "segmentacion"
            mode_cfg = CARDADO_MODES.get(mode, CARDADO_MODES["segmentacion"])
            return {**base_cfg, **mode_cfg}
        return base_cfg

    def _get_model_id(self, stage_key=None, cardado_mode=None):
        key = stage_key or self.current_stage_key or "escarmenado"
        if key == "cardado":
            mode = cardado_mode or self.current_cardado_mode or "segmentacion"
            return f"cardado:{mode}"
        return key

    def _is_homogeneity_mode(self):
        stage_key = self.loaded_model_key or self.current_stage_key
        cardado_mode = self.loaded_cardado_mode if self.loaded_model_key == "cardado" else self.current_cardado_mode
        return stage_key == "cardado" and cardado_mode == "segmentacion"

    def _sync_stage_ui_mode(self):
        """Adapta la visibilidad del selector de Cardado y la barra de diagnóstico según el modo activo."""
        is_cardado = self.current_stage_key == "cardado"
        self.cardado_mode_container.setVisible(is_cardado)

        is_homog = is_cardado and self.current_cardado_mode == "segmentacion"
        self.homogeneity_panel.setVisible(is_homog)

        if is_homog:
            self.status_section_label.setText("Homogeneidad de la fibra")
            self.gauge_title_label.setText("Evaluación de homogeneidad")
            self.update_homogeneity_status(None, waiting=True)
            self.update_homogeneity_gauge(None)
            self.update_homogeneity_details(None)
        else:
            self.status_section_label.setText("Estado de la fibra")
            self.gauge_title_label.setText("Nivel de impurezas")
            self.update_status_label(0)
            self.update_gauge_chart(0)

    def _resolve_model_path(self, filename):
        """Busca el archivo de pesos primero en la carpeta del script."""
        direct_path = os.path.join(self.base_dir, filename)
        if os.path.exists(direct_path):
            return direct_path
        cwd_path = os.path.join(os.getcwd(), filename)
        if os.path.exists(cwd_path):
            return cwd_path
        raise FileNotFoundError(f"No se encontró '{filename}' en '{self.base_dir}'.")

    def select_process_stage(self, stage_key, load_now=True):
        """Selecciona la etapa (Escarmenado, Cardado o Hilado) y carga el modelo bajo demanda."""
        if stage_key not in PROCESS_MODELS:
            return

        self.current_stage_key = stage_key
        for key, btn in self.stage_buttons.items():
            btn.setChecked(key == stage_key)

        self._sync_stage_ui_mode()
        cfg = self._get_stage_config(stage_key)
        if not load_now:
            self.model_info_label.setText(f"{cfg['filename']} · {cfg['desc']} (en espera)")
            return

        ok = self._ensure_model_loaded(stage_key)
        if ok and self.test_image_bgr is not None and not self.timer.isActive():
            self._run_test_image_inference()

    def _on_cardado_mode_changed(self):
        """Alterna entre Segmentación (Homogeneidad) e Imperfecciones (BB) dentro de Cardado."""
        mode_key = self.combo_cardado_mode.currentData()
        if not mode_key or mode_key not in CARDADO_MODES:
            return

        self.current_cardado_mode = mode_key
        if self.current_stage_key != "cardado":
            return

        self._sync_stage_ui_mode()
        ok = self._ensure_model_loaded("cardado")
        if ok and self.test_image_bgr is not None and not self.timer.isActive():
            self._run_test_image_inference()

    def _ensure_model_loaded(self, stage_key=None):
        """Carga bajo demanda el modelo correspondiente y libera el anterior."""
        if stage_key is None:
            stage_key = self.current_stage_key or "escarmenado"

        cardado_mode = self.current_cardado_mode if stage_key == "cardado" else None
        target_model_id = self._get_model_id(stage_key, cardado_mode)

        if self.model is not None and self.loaded_model_id == target_model_id:
            return True

        if YOLO is None:
            QMessageBox.critical(
                self,
                "Error de Modelo",
                f"No se pudo importar Ultralytics YOLO.\n{YOLO_IMPORT_ERROR}"
            )
            return False

        cfg = self._get_stage_config(stage_key, cardado_mode)
        was_running = self.timer.isActive()
        if was_running:
            self.timer.stop()

        try:
            model_path = self._resolve_model_path(cfg["filename"])
            self.model_info_label.setText(f"Cargando {cfg['filename']}...")
            QApplication.processEvents()

            # Liberar modelo previo antes de cargar el nuevo
            if self.model is not None:
                del self.model
                self.model = None
                self.loaded_model_key = None
                self.loaded_cardado_mode = None
                self.loaded_model_id = None
                gc.collect()
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.model = YOLO(model_path)
            self.loaded_model_key = stage_key
            self.loaded_cardado_mode = cardado_mode
            self.loaded_model_id = target_model_id
            self.model_info_label.setText(f"{cfg['filename']} · {cfg['desc']}")
            print(f"Modelo cargado [{cfg['label']}]: {model_path}")
            return True

        except Exception as e:
            self.model = None
            self.loaded_model_key = None
            self.loaded_cardado_mode = None
            self.loaded_model_id = None
            self.model_info_label.setText(f"Error al cargar {cfg['filename']}")
            QMessageBox.critical(
                self,
                "Error de Modelo",
                f"No se pudo cargar el modelo '{cfg['filename']}'.\n{e}"
            )
            return False

        finally:
            if was_running and self.model is not None:
                self.timer.start(33)

    def _on_infer_params_changed(self):
        """Actualiza conf e imgsz compartidos por todos los modelos."""
        self.infer_conf = float(self.spin_conf.value())
        self.infer_imgsz = int(self.combo_imgsz.currentData())

        if self.test_image_bgr is not None and not self.timer.isActive():
            self._run_test_image_inference()

    def _active_stage_config(self):
        key = self.loaded_model_key or self.current_stage_key
        mode = self.loaded_cardado_mode if self.loaded_model_key == "cardado" else self.current_cardado_mode
        return self._get_stage_config(key, mode)

    def _is_segmentation_stage(self):
        return self._active_stage_config().get("task") == "segment"

    def _get_stage_class_names(self, stage_cfg):
        """Devuelve el mapa de nombres de clases aplicando overrides de la etapa si existen."""
        base_names = self.model.names if self.model is not None else {}
        override = stage_cfg.get("class_names_override")
        if not override:
            return base_names

        if isinstance(base_names, dict):
            merged = dict(base_names)
            merged.update(override)
            return merged

        merged = {idx: name for idx, name in enumerate(base_names)}
        merged.update(override)
        return merged

    def _run_inference_on_frame(self, frame):
        """
        Ejecuta la inferencia YOLO compartida tanto para video en tiempo real
        como para la verificación temporal con imágenes.
        """
        if self.model is not None:
            results = self.model(
                frame,
                conf=self.infer_conf,
                imgsz=self.infer_imgsz,
                verbose=False
            )

            detection_count = len(results[0].boxes) if results[0].boxes is not None else 0
            stage_cfg = self._active_stage_config()
            class_names = self._get_stage_class_names(stage_cfg)
            annotated_frame = draw_custom_predictions(
                frame.copy(),
                results,
                class_names,
                draw_masks=self._is_segmentation_stage(),
                color_map=stage_cfg.get("colors")
            )

            speed = results[0].speed
            stage_label = stage_cfg.get("label", "Modelo")

            if self._is_homogeneity_mode():
                mascara_640 = self.homogeneity_service.extraer_mascara_yolo(results[0], frame.shape)
                resultado = (
                    self.homogeneity_service.analizar_con_mascara(frame, mascara_640)
                    if mascara_640 is not None
                    else None
                )
                self.update_homogeneity_status(resultado)
                self.update_homogeneity_gauge(resultado)
                self.update_homogeneity_details(resultado)

                if resultado is not None:
                    log_msg = (
                        f"[Cardado · Homogeneidad] {resultado.decision} | "
                        f"Votos(A:{resultado.votos_alta} M:{resultado.votos_media} B:{resultado.votos_baja}) | "
                        f"M1(CV={resultado.cv1:.3f},H={resultado.entropia1:.2f}) "
                        f"M2(CV={resultado.cv2:.3f}) "
                        f"M3(GLCM={resultado.homogeneidad3:.3f}) "
                        f"M4(CV={resultado.cv4:.3f}) | "
                        f"{speed['inference']:.1f}ms inf"
                    )
                else:
                    log_msg = (
                        f"[Cardado · Homogeneidad] Sin velo/cinta segmentada "
                        f"(conf={self.infer_conf:.2f}, imgsz={self.infer_imgsz}): "
                        f"{speed['inference']:.1f}ms inf"
                    )
                print(log_msg)
            else:
                self.update_status_label(detection_count)
                self.update_gauge_chart(detection_count)
                log_msg = (
                    f"[{stage_label}] {detection_count} detections "
                    f"(conf={self.infer_conf:.2f}, imgsz={self.infer_imgsz}): "
                    f"{speed['preprocess']:.1f}ms pre, {speed['inference']:.1f}ms inf, {speed['postprocess']:.1f}ms post"
                )
                print(log_msg)
        else:
            annotated_frame = frame.copy()

        self._display_bgr_frame(annotated_frame)

    def _display_bgr_frame(self, bgr_frame):
        """Convierte un frame BGR de OpenCV a QPixmap y lo muestra escalado en el visor."""
        rgb_image = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self.last_rendered_pixmap = QPixmap.fromImage(qt_image)
        self._render_last_pixmap()

    def _render_last_pixmap(self):
        if self.last_rendered_pixmap is None:
            return
        target = self.video_label.size()
        if target.width() > 10 and target.height() > 10:
            self.video_label.setPixmap(
                self.last_rendered_pixmap.scaled(
                    target,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_layout(self.width(), self.height())
        if not self.timer.isActive():
            self._render_last_pixmap()

    # ------------------------------------------------------------------
    # === MÓDULO TEMPORAL DE PRUEBA CON IMÁGENES ===
    # (Se puede eliminar luego sin afectar la detección en tiempo real)
    # ------------------------------------------------------------------
    def _open_test_image(self):
        """Carga una imagen local para verificar el modelo y parámetros sin cámara."""
        if self.timer.isActive():
            QMessageBox.information(
                self,
                "Prueba con imagen",
                "Detén la cámara en tiempo real antes de cargar una imagen de prueba."
            )
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar imagen de prueba",
            self.base_dir,
            "Imágenes (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not file_path:
            return

        try:
            data = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("No se pudo decodificar la imagen seleccionada.")

            self.test_image_bgr = img
            self.test_image_path = file_path
            self.btn_clear_test_image.setEnabled(True)
            print(f"Imagen de prueba cargada: {os.path.basename(file_path)} ({img.shape[1]}x{img.shape[0]})")

            self._run_test_image_inference()
        except Exception as e:
            QMessageBox.warning(self, "Error de imagen", f"No se pudo abrir la imagen.\n{e}")

    def _run_test_image_inference(self):
        """Ejecuta inferencia sobre la imagen de prueba cargada usando el modelo seleccionado."""
        if self.test_image_bgr is None:
            return
        if not self._ensure_model_loaded(self.current_stage_key):
            return
        try:
            self._run_inference_on_frame(self.test_image_bgr)
        except Exception as e:
            print(f"Error al procesar imagen de prueba: {e}")

    def _clear_test_image(self):
        """Limpia la imagen de prueba cargada."""
        self.test_image_bgr = None
        self.test_image_path = None
        self.last_rendered_pixmap = None
        self.btn_clear_test_image.setEnabled(False)
        if not self.timer.isActive():
            self.video_label.setPixmap(QPixmap())
            self.video_label.setText("Inicie la detección para ver la cámara o cargue una imagen de prueba")
            self._sync_stage_ui_mode()
        print("Imagen de prueba limpiada.")
    # === FIN MÓDULO TEMPORAL DE PRUEBA CON IMÁGENES ===

    # ------------------------------------------------------------------
    # CALIBRACIÓN DE COLOR (se conservan todos los métodos existentes)
    # ------------------------------------------------------------------
    def load_calibration_profile(self):
        """Carga parámetros de calibración desde JSON. Si no existe, usa valores por defecto."""
        params = DEFAULT_COLOR_PARAMS.copy()

        if os.path.exists(CALIBRATION_PROFILE_PATH):
            try:
                with open(CALIBRATION_PROFILE_PATH, "r", encoding="utf-8") as f:
                    loaded = json.load(f)

                for key in params:
                    if key in loaded:
                        params[key] = loaded[key]

                print(f"Perfil de calibración cargado desde {CALIBRATION_PROFILE_PATH}")
            except Exception as e:
                print(f"No se pudo cargar el perfil de calibración. Se usarán valores por defecto. Error: {e}")
        else:
            print("No existe perfil de calibración. Se usarán valores por defecto.")

        return params

    def save_calibration_profile(self):
        """Guarda en JSON los parámetros actuales de calibración."""
        try:
            self.on_color_param_changed()

            data = self.params.copy()
            data["color_fix_enabled"] = self.color_fix_enabled

            with open(CALIBRATION_PROFILE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

            self.log(f"Perfil de calibración guardado en {CALIBRATION_PROFILE_PATH}")
            QMessageBox.information(
                self,
                "Perfil guardado",
                f"Perfil de calibración guardado en:\n{CALIBRATION_PROFILE_PATH}"
            )
        except Exception as e:
            self.log(f"Error al guardar perfil de calibración: {e}")
            QMessageBox.warning(
                self,
                "Error",
                f"No se pudo guardar el perfil de calibración.\n{e}"
            )

    def reset_calibration_profile(self):
        """Restaura los controles a los valores recomendados por defecto."""
        self.apply_profile_to_controls(DEFAULT_COLOR_PARAMS)
        self.on_color_param_changed()
        self.log("Parámetros de calibración restaurados a valores por defecto.")

    def apply_profile_to_controls(self, params):
        """Aplica un diccionario de parámetros a los controles de la interfaz."""
        self.chk_color_fix.setChecked(bool(params.get("color_fix_enabled", True)))
        self.spin_blue.setValue(float(params.get("blue_scale", DEFAULT_COLOR_PARAMS["blue_scale"])))
        self.spin_green.setValue(float(params.get("green_scale", DEFAULT_COLOR_PARAMS["green_scale"])))
        self.spin_red.setValue(float(params.get("red_scale", DEFAULT_COLOR_PARAMS["red_scale"])))
        self.spin_direct_beta.setValue(float(params.get("direct_beta", DEFAULT_COLOR_PARAMS["direct_beta"])))
        self.spin_gray_strength.setValue(float(params.get("gray_strength", DEFAULT_COLOR_PARAMS["gray_strength"])))
        self.spin_gray_max.setValue(float(params.get("gray_max_gain", DEFAULT_COLOR_PARAMS["gray_max_gain"])))
        self.spin_gray_min.setValue(float(params.get("gray_min_gain", DEFAULT_COLOR_PARAMS["gray_min_gain"])))
        self.spin_vis_alpha.setValue(float(params.get("vis_alpha", DEFAULT_COLOR_PARAMS["vis_alpha"])))
        self.spin_vis_beta.setValue(float(params.get("vis_beta", DEFAULT_COLOR_PARAMS["vis_beta"])))

    def auto_calibrate_white_reference(self):
        """
        Calibración primaria usando un fondo blanco o gris claro.
        Captura varios frames crudos, calcula el promedio BGR en la zona central
        y ajusta los factores Blue/Green/Red para neutralizar la dominante de color.
        """
        if self.cap is None or not self.cap.isOpened():
            QMessageBox.warning(
                self,
                "Calibración",
                "Primero inicia la detección para abrir la cámara."
            )
            return

        reply = QMessageBox.question(
            self,
            "Calibración con fondo blanco",
            "Coloca una hoja blanca o fondo gris claro frente a la cámara, "
            "con iluminación uniforme y la apertura de la lente en un valor medio "
            "por ejemplo f/5.6 o f/8.\n\n"
            "Luego presiona Yes para calibrar.",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        self.log("Iniciando calibración automática con referencia blanca...")

        was_active = self.timer.isActive()
        if was_active:
            self.timer.stop()

        try:
            frames = []
            n_frames = 20

            for _ in range(5):
                self.cap.capture_raw_frame_for_calibration(timeout_ms=1000)

            for _ in range(n_frames):
                ok, raw = self.cap.capture_raw_frame_for_calibration(timeout_ms=1000)
                if ok and raw is not None:
                    h, w = raw.shape[:2]
                    y1, y2 = int(h * 0.30), int(h * 0.70)
                    x1, x2 = int(w * 0.30), int(w * 0.70)
                    roi = raw[y1:y2, x1:x2]
                    frames.append(roi.astype(np.float32))

            if len(frames) < 5:
                raise RuntimeError("No se pudieron capturar suficientes frames para calibrar.")

            stack = np.concatenate([f.reshape(-1, 3) for f in frames], axis=0)

            valid = (
                (stack[:, 0] > 10) & (stack[:, 1] > 10) & (stack[:, 2] > 10) &
                (stack[:, 0] < 245) & (stack[:, 1] < 245) & (stack[:, 2] < 245)
            )

            if valid.sum() < 1000:
                raise RuntimeError(
                    "La referencia blanca está demasiado oscura o saturada. "
                    "Ajusta iluminación/apertura y vuelve a intentar."
                )

            pixels = stack[valid]
            mean_bgr = pixels.mean(axis=0)
            b_mean, g_mean, r_mean = mean_bgr
            global_mean = float(mean_bgr.mean())

            if global_mean < 15:
                raise RuntimeError(
                    "La imagen está demasiado oscura para calibrar. "
                    "Aumenta iluminación o abre ligeramente la lente."
                )

            if max(mean_bgr) > 240:
                raise RuntimeError(
                    "La imagen está demasiado saturada para calibrar. "
                    "Cierra un poco la apertura o reduce iluminación."
                )

            b_gain = global_mean / max(b_mean, 1.0)
            g_gain = global_mean / max(g_mean, 1.0)
            r_gain = global_mean / max(r_mean, 1.0)

            b_gain = float(np.clip(b_gain, 0.55, 2.20))
            g_gain = float(np.clip(g_gain, 0.45, 1.80))
            r_gain = float(np.clip(r_gain, 0.55, 2.20))

            blend = 0.85
            b_gain = 1.0 + blend * (b_gain - 1.0)
            g_gain = 1.0 + blend * (g_gain - 1.0)
            r_gain = 1.0 + blend * (r_gain - 1.0)

            self.spin_blue.setValue(round(b_gain, 2))
            self.spin_green.setValue(round(g_gain, 2))
            self.spin_red.setValue(round(r_gain, 2))

            self.spin_direct_beta.setValue(4)
            self.spin_gray_strength.setValue(0.10)
            self.spin_gray_max.setValue(1.30)
            self.spin_gray_min.setValue(0.80)
            self.spin_vis_alpha.setValue(1.05)
            self.spin_vis_beta.setValue(6)

            self.on_color_param_changed()

            ok, raw_debug = self.cap.capture_raw_frame_for_calibration(timeout_ms=1000)
            if ok and raw_debug is not None:
                corrected_debug = suppress_green_cast_bgr(
                    raw_debug,
                    green_scale=self.params["green_scale"],
                    red_scale=self.params["red_scale"],
                    blue_scale=self.params["blue_scale"],
                    beta=self.params["direct_beta"]
                )
                corrected_debug = gray_world_color_correction_bgr(
                    corrected_debug,
                    strength=self.params["gray_strength"],
                    max_gain=self.params["gray_max_gain"],
                    min_gain=self.params["gray_min_gain"]
                )
                corrected_debug = enhance_visibility_bgr(
                    corrected_debug,
                    alpha=self.params["vis_alpha"],
                    beta=self.params["vis_beta"]
                )

                cv2.imwrite("calibration_white_raw.jpg", raw_debug)
                cv2.imwrite("calibration_white_corrected.jpg", corrected_debug)

            self.save_calibration_profile()

            self.log(
                "Calibración terminada. "
                f"Promedio crudo BGR: B={b_mean:.1f}, G={g_mean:.1f}, R={r_mean:.1f}. "
                f"Ganancias aplicadas: B={b_gain:.2f}, G={g_gain:.2f}, R={r_gain:.2f}"
            )

            QMessageBox.information(
                self,
                "Calibración terminada",
                "Calibración primaria completada y perfil guardado.\n\n"
                f"Promedio crudo BGR:\n"
                f"B={b_mean:.1f}, G={g_mean:.1f}, R={r_mean:.1f}\n\n"
                f"Ganancias aplicadas:\n"
                f"B={b_gain:.2f}, G={g_gain:.2f}, R={r_gain:.2f}\n\n"
                "Se guardaron:\n"
                "calibration_white_raw.jpg\n"
                "calibration_white_corrected.jpg"
            )

        except Exception as e:
            self.log(f"Error durante calibración automática: {e}")
            QMessageBox.warning(self, "Error de calibración", str(e))

        finally:
            if was_active:
                self.timer.start(33)

    def create_spinbox(self, min_val, max_val, step, value, decimals=2):
        box = QDoubleSpinBox()
        box.setRange(min_val, max_val)
        box.setSingleStep(step)
        box.setDecimals(decimals)
        box.setValue(value)
        box.setKeyboardTracking(False)
        return box

    def on_color_param_changed(self):
        self.color_fix_enabled = self.chk_color_fix.isChecked()
        self.params = {
            "blue_scale": self.spin_blue.value(),
            "green_scale": self.spin_green.value(),
            "red_scale": self.spin_red.value(),
            "direct_beta": int(self.spin_direct_beta.value()),
            "gray_strength": self.spin_gray_strength.value(),
            "gray_max_gain": self.spin_gray_max.value(),
            "gray_min_gain": self.spin_gray_min.value(),
            "vis_alpha": self.spin_vis_alpha.value(),
            "vis_beta": int(self.spin_vis_beta.value()),
        }
        self.lbl_params_summary.setText(
            f"B:{self.params['blue_scale']:.2f} | G:{self.params['green_scale']:.2f} | R:{self.params['red_scale']:.2f} | Gray:{self.params['gray_strength']:.2f}"
        )
        if self.cap is not None:
            self.cap.color_fix = self.color_fix_enabled
            self.cap.blue_scale = self.params["blue_scale"]
            self.cap.green_scale = self.params["green_scale"]
            self.cap.red_scale = self.params["red_scale"]
            self.cap.direct_beta = self.params["direct_beta"]
            self.cap.gray_strength = self.params["gray_strength"]
            self.cap.gray_max_gain = self.params["gray_max_gain"]
            self.cap.gray_min_gain = self.params["gray_min_gain"]
            self.cap.vis_alpha = self.params["vis_alpha"]
            self.cap.vis_beta = self.params["vis_beta"]

    def apply_once_wb(self):
        try:
            if (
                self.cap is not None
                and getattr(self.cap, "hCamera", None) is not None
                and mvsdk is not None
                and hasattr(mvsdk, "CameraSetOnceWB")
            ):
                mvsdk.CameraSetOnceWB(self.cap.hCamera)
                self.log("Balance de blancos automático ejecutado.")
            else:
                self.log("La cámara no está abierta o el SDK no soporta CameraSetOnceWB.")
        except Exception as e:
            self.log(f"Error al aplicar balance de blancos: {e}")

    # ------------------------------------------------------------------
    # ESTILOS
    # ------------------------------------------------------------------
    def _apply_styles(self):
        """Estilo neutro y plano de instrumento industrial, sin saturación y táctil."""
        self.setStyleSheet("""
            * {
                font-family: 'Segoe UI', 'Ubuntu', 'Arial', sans-serif;
                color: #1F1F1F;
            }
            QMainWindow, #contentArea {
                background-color: #EFEFEF;
            }
            QSplitter::handle {
                background-color: #EFEFEF;
            }
            QSplitter::handle:hover {
                background-color: #D6D6D6;
            }
            #titleBar {
                background-color: #FFFFFF;
                border-bottom: 1px solid #D6D6D6;
            }
            #logoBox {
                background: transparent;
                color: #666666;
                font-size: 11px;
            }
            #videoLabel {
                background-color: #161616;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                color: #A3A3A3;
                font-size: 14px;
                padding: 12px;
            }
            #sidebarScroll {
                background-color: #FFFFFF;
                border: 1px solid #D6D6D6;
                border-radius: 3px;
            }
            #sidebar {
                background-color: #FFFFFF;
                border: none;
            }
            #toggleSidebarBtn {
                background-color: #9E2A2B;
                border: 1px solid #7D1F20;
                border-radius: 4px;
                color: #FFFFFF;
                font-size: 14px;
                font-weight: 700;
                padding: 0px;
            }
            #toggleSidebarBtn:hover {
                background-color: #B43536;
                border-color: #8A2425;
            }
            #toggleSidebarBtn:pressed {
                background-color: #7D1F20;
            }
            #fullscreenBtn {
                background-color: #FAFAFA;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                color: #262626;
                font-size: 14px;
                font-weight: 600;
                padding: 0px;
            }
            #fullscreenBtn:hover {
                background-color: #EFEFEF;
                border-color: #B3B3B3;
            }
            #touchStepBtn {
                background-color: #F4F4F4;
                border: 1px solid #C8C8C8;
                border-radius: 4px;
                font-size: 14px;
                font-weight: 700;
                padding: 0px;
            }
            #touchStepBtn:pressed {
                background-color: #DFDFDF;
            }
            #divider {
                color: #E5E5E5;
                background-color: #E5E5E5;
                max-height: 1px;
                border: none;
            }
            #sectionLabel {
                font-size: 12px;
                font-weight: 600;
                color: #262626;
            }
            #mutedLabel {
                font-size: 11px;
                color: #5E5E5E;
            }
            #monoValue {
                font-family: 'Consolas', 'DejaVu Sans Mono', 'Courier New', monospace;
                font-size: 11px;
                font-weight: 600;
                color: #262626;
            }
            #monoMuted {
                font-family: 'Consolas', 'DejaVu Sans Mono', 'Courier New', monospace;
                font-size: 10px;
                color: #666666;
            }
            #stageBtn {
                background-color: #FAFAFA;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                padding: 6px 4px;
                font-size: 11px;
                font-weight: 500;
                color: #262626;
            }
            #stageBtn:hover {
                background-color: #EFEFEF;
            }
            #stageBtn:checked {
                background-color: #262626;
                color: #FFFFFF;
                border: 1px solid #1A1A1A;
                font-weight: 600;
            }
            #detectButton {
                background-color: #262626;
                color: #FFFFFF;
                font-size: 13px;
                font-weight: 600;
                padding: 10px 12px;
                border: 1px solid #1A1A1A;
                border-radius: 3px;
            }
            #detectButton:hover {
                background-color: #383838;
            }
            #detectButton:pressed {
                background-color: #171717;
            }
            #detectButton[running="true"] {
                background-color: #F4F4F4;
                color: #8A1F1D;
                border: 1px solid #C98A88;
            }
            #detectButton[running="true"]:hover {
                background-color: #FBECEB;
            }
            #statusLabel {
                font-size: 12px;
                font-weight: 600;
                padding: 8px 10px;
                border-radius: 3px;
            }
            #statusLabel[status="clean"] {
                background-color: #EEF5F0;
                color: #235334;
                border: 1px solid #C6DEC9;
                border-left: 4px solid #3A7D54;
            }
            #statusLabel[status="warning"] {
                background-color: #FBF5E8;
                color: #7A520B;
                border: 1px solid #E8D2A6;
                border-left: 4px solid #C28522;
            }
            #statusLabel[status="danger"] {
                background-color: #FAEEEE;
                color: #7D2322;
                border: 1px solid #E3C0BF;
                border-left: 4px solid #B03A38;
            }
            #statusLabel[status="neutral"] {
                background-color: #F4F4F4;
                color: #4A4A4A;
                border: 1px solid #D6D6D6;
                border-left: 4px solid #8A8A8A;
            }
            #homogeneityPanel {
                background-color: #FAFAFA;
                border: 1px solid #DEDEDE;
                border-radius: 3px;
            }
            QComboBox, QDoubleSpinBox {
                background-color: #FAFAFA;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                padding: 4px 5px;
                font-size: 11px;
                color: #1F1F1F;
            }
            QPushButton {
                background-color: #FAFAFA;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                padding: 6px 8px;
                font-size: 11px;
                font-weight: 500;
                color: #1F1F1F;
            }
            QPushButton:hover {
                background-color: #EFEFEF;
                border-color: #B3B3B3;
            }
            QPushButton:pressed {
                background-color: #E5E5E5;
            }
            QPushButton:disabled {
                color: #A0A0A0;
                background-color: #F5F5F5;
                border-color: #E0E0E0;
            }
            #motorActionBtn {
                border-left: 3px solid #3A7D54;
            }
            #motorStopBtn {
                border-left: 3px solid #B03A38;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #D9D9D9;
                border-radius: 4px;
            }
            QSlider::sub-page:horizontal {
                background: #595959;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #262626;
                border: 1px solid #1A1A1A;
                width: 24px;
                height: 24px;
                margin: -8px 0;
                border-radius: 12px;
            }
            QTextEdit {
                background-color: #F7F7F7;
                border: 1px solid #D9D9D9;
                border-radius: 3px;
                padding: 5px;
                font-family: 'Consolas', 'DejaVu Sans Mono', 'Courier New', monospace;
                font-size: 10px;
                color: #333333;
            }
        """)

    def _set_running_visual_state(self, running):
        self.detect_button.setProperty("running", "true" if running else "false")
        self.detect_button.style().unpolish(self.detect_button)
        self.detect_button.style().polish(self.detect_button)

    def log(self, message):
        """Añade un mensaje al área de texto y hace scroll automático al final."""
        self.log_area.append(str(message))
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

    # ------------------------------------------------------------------
    # LÓGICA DE DETECCIÓN EN TIEMPO REAL (sin alterar su flujo original)
    # ------------------------------------------------------------------
    def start_detection(self):
        """Alterna el estado de la detección (Iniciar/Detener)."""
        if self.timer.isActive():
            self.timer.stop()
            if self.cap:
                self.cap.release()
            self.detect_button.setText("Iniciar detección")
            self._set_running_visual_state(False)
            self.last_rendered_pixmap = None
            self.video_label.setPixmap(QPixmap())
            self.video_label.setText("Detección detenida.")
            print("Detección detenida por el usuario.")
            return

        # Cargar el modelo de la etapa seleccionada bajo demanda antes de abrir la cámara
        if not self._ensure_model_loaded(self.current_stage_key):
            return

        # Limpiar cualquier imagen temporal cargada antes de entrar en tiempo real
        if self.test_image_bgr is not None:
            self.test_image_bgr = None
            self.test_image_path = None
            self.btn_clear_test_image.setEnabled(False)

        try:
            self.cap = USBCamera(
                device="/dev/video0",
                width=800,
                height=600,
                fps=30,
                fourcc="YUYV"
            )
            self.cap.open()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error de Cámara USB",
                f"No se puede abrir la cámara USB.\n{e}"
            )
            self.cap = None
            self._set_running_visual_state(False)
            return

        self.detect_button.setText("Detener detección")
        self._set_running_visual_state(True)
        self.timer.start(33)
        print("Cámara USB abierta. Iniciando detección...")

    def update_status_label(self, count):
        """
        Actualiza la etiqueta de estado basándose en umbrales de conteo de impurezas.
        Utiliza propiedades dinámicas de Qt para cambiar el estilo (colores).
        """
        if count < 5:
            status = "clean"
            text = "FIBRA LIMPIA"
        elif count <= 20:
            status = "warning"
            text = "FIBRA CON IMPUREZAS"
        else:
            status = "danger"
            text = "FIBRA SUCIA"

        self.status_label.setText(text)
        self.status_label.setProperty("status", status)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def update_gauge_chart(self, count):
        """Renderiza una barra horizontal limpia y sobria para el nivel de impurezas."""
        max_count = 60
        clamped_count = max(0, min(count, max_count))
        count_norm = clamped_count / max_count

        if hasattr(self, "gauge_value_label"):
            self.gauge_value_label.setText(f"{count} / {max_count}")

        self.gauge_figure.clear()
        ax = self.gauge_figure.add_subplot(111)
        ax.set_facecolor("#FFFFFF")

        cmap = mcolors.LinearSegmentedColormap.from_list("grad", ["#3A7D54", "#C28522", "#B03A38"])
        bar_color = cmap(count_norm)

        ax.barh([0], [max_count], color="#EAEAEA", height=0.62, edgecolor="#D4D4D4", linewidth=0.7)
        if clamped_count > 0:
            ax.barh([0], [clamped_count], color=bar_color, height=0.62, edgecolor="none")

        ax.set_xlim(0, max_count)
        ax.set_ylim(-0.55, 0.55)
        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        self.gauge_figure.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.08)
        self.gauge_canvas.draw()

    def update_homogeneity_status(self, resultado, waiting=False):
        """Actualiza la etiqueta principal de estado para Cardado -> Segmentación."""
        if resultado is None:
            status = "neutral"
            text = "ESPERANDO" if waiting else "SIN VELO DETECTADO"
        elif resultado.decision == "HOMOGÉNEO":
            status = "clean"
            text = "HOMOGÉNEO"
        elif resultado.decision == "MEDIA HOMOGENEIDAD":
            status = "warning"
            text = "MEDIA HOMOGENEIDAD"
        else:
            status = "danger"
            text = "BAJA HOMOGENEIDAD"

        self.status_label.setText(text)
        self.status_label.setProperty("status", status)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def update_homogeneity_gauge(self, resultado):
        """
        Reemplaza la barra de impurezas por una barra de homogeneidad
        (No homogénea <-> Homogénea) basada en los votos de los 4 métodos.
        """
        self.gauge_figure.clear()
        ax = self.gauge_figure.add_subplot(111)
        ax.set_facecolor("#FFFFFF")

        ax.barh([0], [100], color="#EAEAEA", height=0.62, edgecolor="#D4D4D4", linewidth=0.7)

        if resultado is None:
            if hasattr(self, "gauge_value_label"):
                self.gauge_value_label.setText("Sin velo")
        else:
            score_pct = ((resultado.votos_alta * 1.0 + resultado.votos_media * 0.5) / 4.0) * 100.0
            score_pct = max(8.0 if resultado.votos_baja > 0 else 0.0, min(100.0, score_pct))

            if resultado.decision == "HOMOGÉNEO":
                bar_color = "#3A7D54"
                etiqueta = "HOMOGÉNEA"
            elif resultado.decision == "MEDIA HOMOGENEIDAD":
                bar_color = "#C28522"
                etiqueta = "MEDIA"
            else:
                bar_color = "#B03A38"
                etiqueta = "NO HOMOGÉNEA"

            if hasattr(self, "gauge_value_label"):
                self.gauge_value_label.setText(
                    f"{etiqueta} ({resultado.votos_alta}A·{resultado.votos_media}M·{resultado.votos_baja}B)"
                )

            ax.barh([0], [score_pct], color=bar_color, height=0.62, edgecolor="none")

        for mark in (25, 50, 75):
            ax.axvline(mark, color="#FFFFFF", linewidth=1.0, alpha=0.8)

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.55, 0.55)
        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        self.gauge_figure.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.08)
        self.gauge_canvas.draw()

    def update_homogeneity_details(self, resultado):
        """Muestra el desglose completo de votos y métricas de los 4 métodos."""
        if resultado is None:
            self.homog_votes_label.setText("Votos: Alta 0 · Media 0 · Baja 0")
            self.homog_m1_label.setText("M1 Int/Entropía: --")
            self.homog_m2_label.setText("M2 Perfil velo:  --")
            self.homog_m3_label.setText("M3 Textura GLCM: --")
            self.homog_m4_label.setText("M4 Bloques 4x4:  --")
            return

        self.homog_votes_label.setText(
            f"Votos: Alta {resultado.votos_alta} · Media {resultado.votos_media} · Baja {resultado.votos_baja}"
        )

        def _short(cls_text):
            if "ALTA" in cls_text:
                return "ALTA"
            if "MEDIA" in cls_text:
                return "MEDIA"
            return "BAJA"

        self.homog_m1_label.setText(
            f"M1 Int/Entropía: {_short(resultado.metodo1)} | CV={resultado.cv1:.3f} H={resultado.entropia1:.2f}"
        )
        self.homog_m2_label.setText(
            f"M2 Perfil velo:  {_short(resultado.metodo2)} | CV={resultado.cv2:.3f}"
        )
        self.homog_m3_label.setText(
            f"M3 Textura GLCM: {_short(resultado.metodo3)} | Hom={resultado.homogeneidad3:.3f}"
        )
        self.homog_m4_label.setText(
            f"M4 Bloques 4x4:  {_short(resultado.metodo4)} | CV={resultado.cv4:.3f}"
        )

    def update_frame(self):
        """Ciclo principal: Captura frame -> Inferencia YOLO -> Actualización UI."""
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Error al leer el frame. Deteniendo.")
            self.timer.stop()
            if self.cap:
                self.cap.release()
            self.cap = None
            self.detect_button.setText("Iniciar detección")
            self._set_running_visual_state(False)
            QMessageBox.warning(self, "Advertencia", "Se perdió la conexión con la cámara.")
            return

        try:
            self._run_inference_on_frame(frame)
        except Exception as e:
            print(f"Error en el ciclo de actualización: {str(e)}")

    # ------------------------------------------------------------------
    # CONTROL DE MOTOR ARDUINO / L298N
    # ------------------------------------------------------------------
    def _find_motor_ports(self):
        self.motor_port_combo.clear()

        if serial is None:
            self.motor_port_combo.addItem('PySerial no disponible')
            return

        ports = sorted(glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*'))
        if os.name == 'nt':
            try:
                from serial.tools import list_ports
                ports = [p.device for p in list_ports.comports()]
            except Exception:
                ports = []

        if ports:
            self.motor_port_combo.addItems(ports)
        else:
            self.motor_port_combo.addItem('No encontrado')

    def _toggle_motor_connection(self):
        if serial is None:
            detail = str(SERIAL_IMPORT_ERROR) if SERIAL_IMPORT_ERROR else 'PySerial no está instalado.'
            QMessageBox.warning(
                self,
                'Arduino',
                f'No se puede abrir la conexión serial.\n{detail}\n\n'
                'La detección por visión puede seguir funcionando normalmente.'
            )
            return

        if self.serial_port is not None:
            self._disconnect_motor_serial()
            return

        port = self.motor_port_combo.currentText()
        if not port or port == 'No encontrado':
            QMessageBox.warning(self, 'Arduino', 'No se encontró un puerto Arduino.')
            return

        try:
            self.serial_port = serial.Serial(port, 115200, timeout=0.05)
            time.sleep(1.8)
            self.serial_port.reset_input_buffer()
            self.motor_connect_button.setText('Desconectar')
            self.motor_connection_label.setText(f'Arduino: conectado ({port})')
            self._send_motor_command('STOP')
            self.log(f'Arduino conectado en {port} a 115200 baud.')
        except Exception as e:
            self.serial_port = None
            QMessageBox.critical(self, 'Error de conexión Arduino', str(e))

    def _disconnect_motor_serial(self):
        try:
            if self.serial_port is not None:
                self._send_motor_command('STOP')
                self.serial_port.close()
        except Exception:
            pass

        self.serial_port = None
        self.motor_running = False
        self.motor_connect_button.setText('Conectar')
        self.motor_connection_label.setText('Arduino: desconectado')
        self.log('Arduino desconectado.')

    def _send_motor_command(self, command):
        if self.serial_port is None or not self.serial_port.is_open:
            return False

        try:
            self.serial_port.write((command + '\n').encode('utf-8'))
            self.serial_port.flush()
            return True
        except Exception as e:
            self.log(f'Error serial Arduino: {e}')
            return False

    def _motor_start(self):
        if self.serial_port is None or not self.serial_port.is_open:
            QMessageBox.warning(self, 'Arduino', 'Primero conecta el Arduino.')
            return

        self._send_motor_command(f'SPEED:{self.motor_speed}')
        self._send_motor_command(self.motor_direction)
        self._send_motor_command('START')
        self.motor_running = True
        self.log(f'Motor iniciado. Dirección={self.motor_direction}, PWM={self.motor_speed}.')

    def _motor_stop(self):
        self._send_motor_command('STOP')
        self.motor_running = False
        self.log('Motor detenido.')

    def _motor_forward(self):
        self.motor_direction = 'FORWARD'
        self._send_motor_command('FORWARD')
        self.log('Dirección del motor: ADELANTE.')

    def _motor_reverse(self):
        self.motor_direction = 'REVERSE'
        self._send_motor_command('REVERSE')
        self.log('Dirección del motor: REVERSA.')

    def _motor_speed_changed(self, value):
        self.motor_speed = int(value)
        pct = int(round(self.motor_speed * 100 / 255))
        self.motor_speed_label.setText(f'PWM: {self.motor_speed}  |  {pct} %')

        if self.motor_running:
            self._send_motor_command(f'SPEED:{self.motor_speed}')

    def _read_motor_serial(self):
        if self.serial_port is None or not self.serial_port.is_open:
            return

        try:
            while self.serial_port.in_waiting:
                line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    self.motor_response_label.setText(f'Respuesta Arduino: {line}')
        except Exception as e:
            self.log(f'Error leyendo Arduino: {e}')

    def closeEvent(self, event):
        """Limpieza de recursos al cerrar la ventana."""
        print("Cerrando aplicación...")
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            print("Cámara liberada.")

        try:
            self.motor_serial_timer.stop()
            if self.serial_port is not None:
                self._send_motor_command("STOP")
                self.serial_port.close()
                self.serial_port = None
        except Exception:
            pass

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        event.accept()


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    QApplication.setAttribute(Qt.AA_SynthesizeMouseForUnhandledTabletEvents, True)
    QApplication.setAttribute(Qt.AA_SynthesizeTouchForUnhandledMouseEvents, True)
    App = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(App.exec())