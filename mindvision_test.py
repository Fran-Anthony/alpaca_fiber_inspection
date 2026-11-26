#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mindvision_white_calibration.py

Calibración objetiva de color usando una hoja blanca mate en la ROI central.

Estado fijo de cámara:
    - Exposure = 8.33 ms
    - Auto Exposure = OFF
    - Gray-World = OFF
    - Once WB = OFF
    - Ajuste de B/G/R únicamente MANUAL y FIJO

Objetivo:
    En vez de calibrar "a ojo", medir cuantitativamente qué tan neutra
    y bien expuesta está la hoja blanca.

Controles:
    1 / 2 : Blue  - / +
    3 / 4 : Green - / +
    5 / 6 : Red   - / +
    7 / 8 : Beta  - / +

    c     : corrección ON/OFF
    r     : reset
    s     : guardar RAW + corregida
    q/ESC : salir

Interpretación:
    - B, G y R deben quedar muy próximos.
    - No buscamos 255,255,255: eso sería saturación.
    - Idealmente la hoja debe quedar luminosa, pero con detalle.
"""

import time
import cv2
import numpy as np
import mvsdk


# ============================================================
# CÁMARA
# ============================================================

CAMERA_INDEX = 0
EXPOSURE_MS = 8.33


# ============================================================
# AJUSTES MANUALES DE COLOR
# ============================================================

INITIAL_BLUE_SCALE = 1.49
INITIAL_GREEN_SCALE = 0.85
INITIAL_RED_SCALE = 1.72
INITIAL_BETA = -1

COLOR_STEP = 0.02
BETA_STEP = 1

MIN_COLOR_SCALE = 0.20
MAX_COLOR_SCALE = 2.50
MIN_BETA = -30
MAX_BETA = 30


# ============================================================
# ROI / CRITERIOS DE BLANCO
# ============================================================

# ROI central: 30% del ancho y 30% del alto.
ROI_FRACTION = 0.04

# Un píxel se considera "casi neutro" si la diferencia
# entre su canal máximo y mínimo no supera este valor.
NEUTRAL_TOLERANCE = 15

# Rango de luminancia útil para una hoja blanca.
# Evitamos exigir 255 porque sería clipping/saturación.
WHITE_MIN_BRIGHTNESS = 180
WHITE_MAX_BRIGHTNESS = 245

# Consideramos saturado un canal >= este valor.
SATURATION_THRESHOLD = 250

# Para el indicador de luminosidad, el centro recomendado.
TARGET_BRIGHTNESS = 220.0


# ============================================================
# VISUALIZACIÓN
# ============================================================

WINDOW_NAME = "MindVision - calibracion objetiva de blanco"
SCREEN_USAGE = 0.82

FALLBACK_SCREEN_WIDTH = 1280
FALLBACK_SCREEN_HEIGHT = 720


def get_screen_size():
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()

        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()

        root.destroy()

        return int(width), int(height)

    except Exception as exc:
        print(
            "No se pudo detectar el monitor. "
            f"Usando {FALLBACK_SCREEN_WIDTH}x{FALLBACK_SCREEN_HEIGHT}. "
            f"Detalle: {exc}"
        )

        return FALLBACK_SCREEN_WIDTH, FALLBACK_SCREEN_HEIGHT


def calculate_display_size(frame_width, frame_height, screen_width, screen_height):
    max_width = int(screen_width * SCREEN_USAGE)
    max_height = int(screen_height * SCREEN_USAGE)

    scale_w = max_width / float(frame_width)
    scale_h = max_height / float(frame_height)

    scale = min(scale_w, scale_h, 1.0)

    return (
        max(1, int(frame_width * scale)),
        max(1, int(frame_height * scale)),
        scale,
    )


def fixed_color_correction_bgr(frame, blue_scale, green_scale, red_scale, beta):
    """Corrección manual completamente fija."""
    if frame is None or frame.size == 0:
        return frame

    image = frame.astype(np.float32)

    image[:, :, 0] *= blue_scale
    image[:, :, 1] *= green_scale
    image[:, :, 2] *= red_scale
    image += beta

    return np.clip(image, 0, 255).astype(np.uint8)


def center_roi(frame, fraction=ROI_FRACTION):
    """Devuelve ROI central y sus coordenadas."""
    h, w = frame.shape[:2]

    roi_w = max(1, int(w * fraction))
    roi_h = max(1, int(h * fraction))

    x1 = (w - roi_w) // 2
    y1 = (h - roi_h) // 2
    x2 = x1 + roi_w
    y2 = y1 + roi_h

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def analyze_white_roi(roi):
    """
    Analiza objetivamente qué tan cerca del blanco neutro está la ROI.

    IMPORTANTE:
    Este score es un indicador práctico de ingeniería, NO una medición
    colorimétrica certificada.

    Métricas:
      - número total de píxeles
      - B/G/R promedio
      - separación máxima entre canales
      - error relativo de neutralidad
      - luminancia promedio
      - porcentaje de píxeles neutros
      - porcentaje de píxeles "blancos útiles"
      - porcentaje de píxeles saturados
      - score 0..100
    """

    if roi is None or roi.size == 0:
        return None

    pixels = roi.reshape(-1, 3).astype(np.float32)
    total_pixels = pixels.shape[0]

    # Promedios BGR
    channel_means = pixels.mean(axis=0)
    b_mean, g_mean, r_mean = map(float, channel_means)

    global_mean = float(channel_means.mean())

    channel_range = float(channel_means.max() - channel_means.min())

    if global_mean > 1e-6:
        neutral_error_pct = 100.0 * channel_range / global_mean
    else:
        neutral_error_pct = 100.0

    # Brightness simple: promedio de los tres canales por píxel.
    brightness = pixels.mean(axis=1)
    mean_brightness = float(brightness.mean())

    # Diferencia de color por píxel.
    pixel_max = pixels.max(axis=1)
    pixel_min = pixels.min(axis=1)
    pixel_channel_range = pixel_max - pixel_min

    neutral_mask = pixel_channel_range <= NEUTRAL_TOLERANCE

    brightness_mask = (
        (brightness >= WHITE_MIN_BRIGHTNESS)
        & (brightness <= WHITE_MAX_BRIGHTNESS)
    )

    useful_white_mask = neutral_mask & brightness_mask

    # Saturación: al menos un canal demasiado cerca de 255.
    saturated_mask = pixel_max >= SATURATION_THRESHOLD

    neutral_pct = 100.0 * float(neutral_mask.mean())
    useful_white_pct = 100.0 * float(useful_white_mask.mean())
    saturated_pct = 100.0 * float(saturated_mask.mean())

    # ------------------------------------------------------------
    # SCORE DE NEUTRALIDAD
    # ------------------------------------------------------------
    # 0 diferencia -> 100
    # >= 30 niveles de diferencia promedio -> 0
    neutrality_score = 100.0 * (
        1.0 - min(channel_range / 30.0, 1.0)
    )

    # ------------------------------------------------------------
    # SCORE DE BRILLO
    # ------------------------------------------------------------
    # Máximo cerca de TARGET_BRIGHTNESS.
    # Una desviación >= 80 niveles da 0.
    brightness_error = abs(mean_brightness - TARGET_BRIGHTNESS)

    brightness_score = 100.0 * (
        1.0 - min(brightness_error / 80.0, 1.0)
    )

    # ------------------------------------------------------------
    # PENALIZACIÓN POR SATURACIÓN
    # ------------------------------------------------------------
    saturation_score = max(
        0.0,
        100.0 - saturated_pct * 5.0
    )

    # Score práctico:
    # Neutralidad tiene más peso que brillo.
    score = (
        0.60 * neutrality_score
        + 0.25 * brightness_score
        + 0.15 * saturation_score
    )

    # ------------------------------------------------------------
    # ESTADO / RECOMENDACIÓN
    # ------------------------------------------------------------
    if saturated_pct > 5.0:
        status = "SATURADO"
        recommendation = "Demasiada luz: evita clipping antes de ajustar color."

    elif mean_brightness < WHITE_MIN_BRIGHTNESS:
        status = "OSCURO"
        recommendation = "La hoja esta oscura; no compenses color con Beta aun."

    elif neutral_error_pct <= 3.0 and saturated_pct < 1.0:
        status = "EXCELENTE"
        recommendation = "B/G/R muy equilibrados. Guarda estos parametros."

    elif neutral_error_pct <= 6.0:
        status = "MUY CERCA"
        recommendation = "Ajusta finamente el canal mas alto/bajo."

    elif neutral_error_pct <= 12.0:
        status = "MEJORANDO"
        recommendation = "Todavia hay una dominante de color visible."

    else:
        status = "LEJOS"
        recommendation = "Corrige primero el desequilibrio entre B, G y R."

    # Dirección sencilla para el usuario.
    labels = ["B", "G", "R"]

    min_idx = int(np.argmin(channel_means))
    max_idx = int(np.argmax(channel_means))

    direction = (
        f"Subir {labels[min_idx]} o bajar {labels[max_idx]}"
    )

    return {
        "total_pixels": total_pixels,
        "b_mean": b_mean,
        "g_mean": g_mean,
        "r_mean": r_mean,
        "mean_brightness": mean_brightness,
        "channel_range": channel_range,
        "neutral_error_pct": neutral_error_pct,
        "neutral_pct": neutral_pct,
        "useful_white_pct": useful_white_pct,
        "saturated_pct": saturated_pct,
        "score": float(np.clip(score, 0.0, 100.0)),
        "status": status,
        "recommendation": recommendation,
        "direction": direction,
    }


class MindVisionCamera:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index

        self.h_camera = None
        self.capability = None
        self.frame_buffer = None

        self.channels = 3
        self.mono = False
        self.opened = False

    def open(self):
        devices = mvsdk.CameraEnumerateDevice()

        if not devices:
            raise RuntimeError(
                "No se detecto ninguna camara MindVision."
            )

        if self.camera_index >= len(devices):
            raise RuntimeError(
                f"Indice invalido: {self.camera_index}. "
                f"Detectadas: {len(devices)}"
            )

        device = devices[self.camera_index]

        print("=" * 72)
        print("MINDVISION - CALIBRACION OBJETIVA DE BLANCO")
        print("=" * 72)
        print(f"Usando camara: {device}")

        self.h_camera = mvsdk.CameraInit(device, -1, -1)
        self.capability = mvsdk.CameraGetCapability(self.h_camera)

        self.mono = self.capability.sIspCapacity.bMonoSensor != 0

        if self.mono:
            mvsdk.CameraSetIspOutFormat(
                self.h_camera,
                mvsdk.CAMERA_MEDIA_TYPE_MONO8
            )
            self.channels = 1
        else:
            mvsdk.CameraSetIspOutFormat(
                self.h_camera,
                mvsdk.CAMERA_MEDIA_TYPE_BGR8
            )
            self.channels = 3

        mvsdk.CameraSetTriggerMode(self.h_camera, 0)

        # EXPOSICION TOTALMENTE FIJA.
        mvsdk.CameraSetAeState(self.h_camera, 0)
        mvsdk.CameraSetExposureTime(
            self.h_camera,
            EXPOSURE_MS * 1000.0
        )

        # Desactivar WB automatico si el wrapper lo soporta.
        if hasattr(mvsdk, "CameraSetWbMode"):
            try:
                mvsdk.CameraSetWbMode(
                    self.h_camera,
                    False
                )
                print("White Balance automatico: OFF")
            except Exception as exc:
                print(
                    "No se pudo forzar WB manual. "
                    f"Detalle: {exc}"
                )

        mvsdk.CameraPlay(self.h_camera)

        max_width = self.capability.sResolutionRange.iWidthMax
        max_height = self.capability.sResolutionRange.iHeightMax

        buffer_size = max_width * max_height * self.channels

        self.frame_buffer = mvsdk.CameraAlignMalloc(
            buffer_size,
            16
        )

        self.opened = True

        time.sleep(0.5)

        print("Auto Exposure: OFF")
        print(f"Exposure: {EXPOSURE_MS:.2f} ms FIJO")
        print("Gray-World: OFF")
        print("Once WB: NO")
        print("=" * 72)

    def read(self, timeout_ms=1000):
        if not self.opened or self.h_camera is None:
            return False, None

        raw_data = None

        try:
            raw_data, frame_head = mvsdk.CameraGetImageBuffer(
                self.h_camera,
                timeout_ms
            )

            mvsdk.CameraImageProcess(
                self.h_camera,
                raw_data,
                self.frame_buffer,
                frame_head
            )

            frame_data = (
                mvsdk.c_ubyte * frame_head.uBytes
            ).from_address(self.frame_buffer)

            frame = np.frombuffer(
                frame_data,
                dtype=np.uint8
            )

            if self.mono:
                frame = frame.reshape(
                    (frame_head.iHeight, frame_head.iWidth)
                )

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_GRAY2BGR
                )

            else:
                frame = frame.reshape(
                    (
                        frame_head.iHeight,
                        frame_head.iWidth,
                        3
                    )
                )

            return True, frame.copy()

        except mvsdk.CameraException as exc:
            error_code = getattr(
                exc,
                "error_code",
                None
            )

            timeout_code = getattr(
                mvsdk,
                "CAMERA_STATUS_TIME_OUT",
                None
            )

            if error_code == timeout_code:
                print("TIMEOUT esperando frame.")
            else:
                print(f"Error MVSDK: {exc}")

            return False, None

        except Exception as exc:
            print(f"Error leyendo frame: {exc}")
            return False, None

        finally:
            if raw_data is not None:
                try:
                    mvsdk.CameraReleaseImageBuffer(
                        self.h_camera,
                        raw_data
                    )
                except Exception:
                    pass

    def release(self):
        if self.frame_buffer is not None:
            try:
                mvsdk.CameraAlignFree(
                    self.frame_buffer
                )
            except Exception:
                pass

            self.frame_buffer = None

        if self.h_camera is not None:
            try:
                mvsdk.CameraUnInit(
                    self.h_camera
                )
            except Exception:
                pass

            self.h_camera = None

        self.opened = False

        print("Camara liberada.")


def clamp_color(value):
    return float(
        np.clip(
            value,
            MIN_COLOR_SCALE,
            MAX_COLOR_SCALE
        )
    )


def score_color(score):
    """Color BGR para visualizar el score."""
    if score >= 90:
        return (0, 255, 0)
    elif score >= 75:
        return (0, 220, 255)
    else:
        return (0, 0, 255)


def draw_info(
    frame,
    fps,
    enabled,
    blue_scale,
    green_scale,
    red_scale,
    beta,
    roi_rect,
    metrics,
):
    output = frame.copy()

    x1, y1, x2, y2 = roi_rect

    metric_color = score_color(metrics["score"])

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        metric_color,
        3
    )

    lines = [
        f"Exposure: {EXPOSURE_MS:.2f} ms FIXED | FPS: {fps:.1f}",
        f"Correction: {'ON' if enabled else 'OFF'}",
        (
            f"B:{blue_scale:.2f}  "
            f"G:{green_scale:.2f}  "
            f"R:{red_scale:.2f}  "
            f"Beta:{beta}"
        ),
        (
            f"ROI pixels: {metrics['total_pixels']:,}"
        ),
        (
            f"Mean BGR: "
            f"{metrics['b_mean']:.1f} / "
            f"{metrics['g_mean']:.1f} / "
            f"{metrics['r_mean']:.1f}"
        ),
        (
            f"Brightness: {metrics['mean_brightness']:.1f} "
            f"| channel spread: {metrics['channel_range']:.1f}"
        ),
        (
            f"Neutral error: {metrics['neutral_error_pct']:.2f}% "
            f"| neutral pixels: {metrics['neutral_pct']:.1f}%"
        ),
        (
            f"Useful white: {metrics['useful_white_pct']:.1f}% "
            f"| saturated: {metrics['saturated_pct']:.2f}%"
        ),
        (
            f"WHITE SCORE: {metrics['score']:.1f}/100 "
            f"[{metrics['status']}]"
        ),
        (
            f"Suggestion: {metrics['direction']}"
        ),
        "1/2 B-/B+ | 3/4 G-/G+ | 5/6 R-/R+",
        "7/8 Beta-/Beta+ | r reset | c compare | s save | q quit",
    ]

    y = 30

    for index, text in enumerate(lines):
        color = (
            metric_color
            if index in (8, 9)
            else (0, 255, 0)
        )

        cv2.putText(
            output,
            text,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            color,
            2,
            cv2.LINE_AA
        )

        y += 25

    return output


def print_metrics(metrics, blue_scale, green_scale, red_scale, beta):
    print()
    print("-" * 72)
    print("MEDICION ROI")
    print("-" * 72)
    print(f"Pixeles analizados : {metrics['total_pixels']:,}")
    print(
        "BGR promedio       : "
        f"B={metrics['b_mean']:.2f}, "
        f"G={metrics['g_mean']:.2f}, "
        f"R={metrics['r_mean']:.2f}"
    )
    print(
        f"Brillo promedio    : {metrics['mean_brightness']:.2f}"
    )
    print(
        f"Separacion canales : {metrics['channel_range']:.2f}"
    )
    print(
        f"Error neutralidad  : {metrics['neutral_error_pct']:.2f}%"
    )
    print(
        f"Pixeles neutros    : {metrics['neutral_pct']:.2f}%"
    )
    print(
        f"Blanco util        : {metrics['useful_white_pct']:.2f}%"
    )
    print(
        f"Pixeles saturados  : {metrics['saturated_pct']:.2f}%"
    )
    print(
        f"WHITE SCORE        : {metrics['score']:.2f}/100 "
        f"({metrics['status']})"
    )
    print(
        f"Sugerencia         : {metrics['direction']}"
    )
    print(
        "Parametros         : "
        f"B={blue_scale:.2f}, "
        f"G={green_scale:.2f}, "
        f"R={red_scale:.2f}, "
        f"Beta={beta}"
    )
    print("-" * 72)


def main():
    blue_scale = INITIAL_BLUE_SCALE
    green_scale = INITIAL_GREEN_SCALE
    red_scale = INITIAL_RED_SCALE
    beta = INITIAL_BETA

    correction_enabled = True

    camera = MindVisionCamera(
        camera_index=CAMERA_INDEX
    )

    try:
        camera.open()

        screen_width, screen_height = get_screen_size()

        print(
            f"Monitor: {screen_width}x{screen_height}"
        )

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL
        )

        last_time = time.perf_counter()
        fps_smoothed = 0.0

        consecutive_failures = 0
        window_configured = False

        last_print_time = 0.0

        print()
        print("Coloca una hoja BLANCA MATE cubriendo completamente")
        print("el rectangulo central.")
        print()
        print("Meta practica:")
        print("  - B ~= G ~= R")
        print("  - error neutralidad < 3-6%")
        print("  - saturacion < 1%")
        print("  - brillo aprox. 180-245")
        print()
        print("Controles:")
        print("  1 / 2 : Blue  - / +")
        print("  3 / 4 : Green - / +")
        print("  5 / 6 : Red   - / +")
        print("  7 / 8 : Beta  - / +")
        print("  r     : reset")
        print("  c     : correccion ON/OFF")
        print("  s     : guardar imagen")
        print("  q/ESC : salir")
        print()

        while True:
            ok, raw_frame = camera.read(
                timeout_ms=1000
            )

            if not ok or raw_frame is None:
                consecutive_failures += 1

                if consecutive_failures >= 10:
                    raise RuntimeError(
                        "10 frames consecutivos perdidos."
                    )

                continue

            consecutive_failures = 0

            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            if dt > 0:
                current_fps = 1.0 / dt

                if fps_smoothed == 0.0:
                    fps_smoothed = current_fps
                else:
                    fps_smoothed = (
                        0.90 * fps_smoothed
                        + 0.10 * current_fps
                    )

            if correction_enabled:
                corrected = fixed_color_correction_bgr(
                    raw_frame,
                    blue_scale,
                    green_scale,
                    red_scale,
                    beta
                )
            else:
                corrected = raw_frame.copy()

            roi, roi_rect = center_roi(
                corrected,
                fraction=ROI_FRACTION
            )

            metrics = analyze_white_roi(roi)

            display = draw_info(
                corrected,
                fps_smoothed,
                correction_enabled,
                blue_scale,
                green_scale,
                red_scale,
                beta,
                roi_rect,
                metrics
            )

            if not window_configured:
                frame_h, frame_w = display.shape[:2]

                display_w, display_h, scale = (
                    calculate_display_size(
                        frame_w,
                        frame_h,
                        screen_width,
                        screen_height
                    )
                )

                cv2.resizeWindow(
                    WINDOW_NAME,
                    display_w,
                    display_h
                )

                print(
                    f"Frame: {frame_w}x{frame_h}"
                )
                print(
                    f"ROI: {int(frame_w * ROI_FRACTION)}x"
                    f"{int(frame_h * ROI_FRACTION)} aprox."
                )
                print(
                    f"Ventana: {display_w}x{display_h} "
                    f"(scale={scale:.3f})"
                )

                window_configured = True

            cv2.imshow(
                WINDOW_NAME,
                display
            )

            # Imprimir una medicion resumida cada 2 segundos.
            if now - last_print_time >= 2.0:
                print(
                    f"SCORE {metrics['score']:5.1f} | "
                    f"BGR {metrics['b_mean']:6.1f} "
                    f"{metrics['g_mean']:6.1f} "
                    f"{metrics['r_mean']:6.1f} | "
                    f"err {metrics['neutral_error_pct']:5.2f}% | "
                    f"sat {metrics['saturated_pct']:5.2f}% | "
                    f"{metrics['direction']}"
                )
                last_print_time = now

            key = cv2.waitKey(1) & 0xFF

            changed = False

            if key in (27, ord("q")):
                break

            elif key == ord("1"):
                blue_scale = clamp_color(
                    blue_scale - COLOR_STEP
                )
                changed = True

            elif key == ord("2"):
                blue_scale = clamp_color(
                    blue_scale + COLOR_STEP
                )
                changed = True

            elif key == ord("3"):
                green_scale = clamp_color(
                    green_scale - COLOR_STEP
                )
                changed = True

            elif key == ord("4"):
                green_scale = clamp_color(
                    green_scale + COLOR_STEP
                )
                changed = True

            elif key == ord("5"):
                red_scale = clamp_color(
                    red_scale - COLOR_STEP
                )
                changed = True

            elif key == ord("6"):
                red_scale = clamp_color(
                    red_scale + COLOR_STEP
                )
                changed = True

            elif key == ord("7"):
                beta = int(
                    np.clip(
                        beta - BETA_STEP,
                        MIN_BETA,
                        MAX_BETA
                    )
                )
                changed = True

            elif key == ord("8"):
                beta = int(
                    np.clip(
                        beta + BETA_STEP,
                        MIN_BETA,
                        MAX_BETA
                    )
                )
                changed = True

            elif key == ord("r"):
                blue_scale = INITIAL_BLUE_SCALE
                green_scale = INITIAL_GREEN_SCALE
                red_scale = INITIAL_RED_SCALE
                beta = INITIAL_BETA
                changed = True

            elif key == ord("c"):
                correction_enabled = (
                    not correction_enabled
                )
                print(
                    "Correccion -> "
                    + (
                        "ON"
                        if correction_enabled
                        else "OFF"
                    )
                )

            elif key == ord("s"):
                timestamp = time.strftime(
                    "%Y%m%d_%H%M%S"
                )

                raw_name = (
                    f"mindvision_white_RAW_{timestamp}.png"
                )

                corrected_name = (
                    "mindvision_white_CORRECTED_"
                    f"{timestamp}_"
                    f"B{blue_scale:.2f}_"
                    f"G{green_scale:.2f}_"
                    f"R{red_scale:.2f}_"
                    f"Beta{beta}_"
                    f"Score{metrics['score']:.1f}.png"
                )

                cv2.imwrite(
                    raw_name,
                    raw_frame
                )

                cv2.imwrite(
                    corrected_name,
                    corrected
                )

                print_metrics(
                    metrics,
                    blue_scale,
                    green_scale,
                    red_scale,
                    beta
                )

                print(f"RAW guardado: {raw_name}")
                print(
                    f"Corregido guardado: {corrected_name}"
                )

            if changed:
                print(
                    "Parametros -> "
                    f"B={blue_scale:.2f}, "
                    f"G={green_scale:.2f}, "
                    f"R={red_scale:.2f}, "
                    f"Beta={beta}"
                )

    except KeyboardInterrupt:
        print("\nInterrumpido.")

    except Exception as exc:
        print(f"\nERROR: {exc}")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
