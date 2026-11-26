"""Perfil de color y calibración de referencia blanca."""
import json
import os

import cv2
import numpy as np
from PyQt5.QtWidgets import QDoubleSpinBox, QMessageBox

from .paths import RUNTIME_DIR

CALIBRATION_PROFILE_PATH = RUNTIME_DIR / "mindvision_color_profile.json"

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


class ColorCalibration:
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

            RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
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

                RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(RUNTIME_DIR / "calibration_white_raw.jpg"), raw_debug)
                cv2.imwrite(str(RUNTIME_DIR / "calibration_white_corrected.jpg"), corrected_debug)

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
                "Archivos en runtime/\n"
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
            apply_wb = getattr(self.cap, "apply_once_wb", None)
            if apply_wb is not None and self.cap.isOpened():
                apply_wb()
                self.log("Balance de blancos automático ejecutado.")
            else:
                self.log("La cámara no está abierta o el SDK no soporta CameraSetOnceWB.")
        except Exception as e:
            self.log(f"Error al aplicar balance de blancos: {e}")

