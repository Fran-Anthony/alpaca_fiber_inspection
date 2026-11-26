"""Homogeneidad de la cinta a partir de las máscaras de Cardado."""
from dataclasses import dataclass

import cv2
import numpy as np


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

    @property
    def puntaje(self):
        return (self.votos_alta + 0.5 * self.votos_media) * 25


class ServicioHomogeneidad:
    """
    Evalúa la homogeneidad del velo segmentado en 640x640 mediante 4 métodos
    (intensidad/entropía, perfil del velo, textura GLCM y bloques 4x4) y votación.
    """

    SIZE = 640

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

    def analizar_con_mascara(self, frame_bgr, mascara_velo):
        """Evalúa los cuatro métodos sobre la máscara de la cinta."""
        if frame_bgr is None or frame_bgr.size == 0 or mascara_velo is None:
            return None
        if not np.any(mascara_velo > 0.5):
            return None

        redimensionada = cv2.resize(frame_bgr, (self.SIZE, self.SIZE), interpolation=cv2.INTER_LINEAR)
        gris = self._a_gris(redimensionada)
        if mascara_velo.shape != (self.SIZE, self.SIZE):
            mascara_velo = cv2.resize(
                mascara_velo.astype(np.float32), (self.SIZE, self.SIZE), interpolation=cv2.INTER_NEAREST,
            )
        mascara_limpia = self._limpiar_mascara(mascara_velo)
        roi_gris = self._aplicar_mascara(gris, mascara_limpia)

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

        if cv < 0.25 and entropia < 6.4:
            clase = "ALTA HOMOGENEIDAD"
        elif cv < 0.375 and entropia < 8.0:
            clase = "MEDIA HOMOGENEIDAD"
        else:
            clase = "BAJA HOMOGENEIDAD"

        return {"cv": cv, "entropia": entropia, "clase": clase}

    def _metodo2_perfil_velo(self, mascara):
        perfil = np.sum(mascara > 0.5, axis=0).astype(np.float64)
        media = float(np.mean(perfil))
        std = float(np.sqrt(np.mean((perfil - media) ** 2)))
        cv = std / media if media > 0.0 else 0.0

        if cv < 0.25:
            clase = "ALTA HOMOGENEIDAD"
        elif cv < 0.50:
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

        if homogeneidad > 0.54:
            clase = "ALTA HOMOGENEIDAD"
        elif homogeneidad > 0.16:
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

        if cv < 0.20:
            clase = "ALTA HOMOGENEIDAD"
        elif cv < 0.375:
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
