#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mindvision_camera_app_tablet_webm_vp8.py
========================================

Aplicación táctil simple para grabar video con cámara MindVision mediante MVSDK.

Diseñada para:
- Jetson / Ubuntu
- pantalla táctil tipo tablet landscape
- usuario no programador
- reproducción directa en Firefox
- buena calidad priorizando detalle sobre FPS

Formato de video:
- Contenedor: WebM
- Codec: VP8 (libvpx)
- FPS: 10
- Pixel format: yuv420p
- Bitrate: 12 Mbps
- CRF: 8

Calibración aplicada DIRECTAMENTE en el ISP MindVision:
- Exposure: 8.33 ms
- R gain: 142
- G gain: 71
- B gain: 120
- Auto Exposure: OFF
- Auto White Balance: OFF

No se aplica ninguna transformación B/G/R posterior con NumPy/OpenCV.

Requisito:
    ffmpeg con libvpx
    (verificar: ffmpeg -encoders | grep libvpx)

Salida:
    videos_mindvision/video_YYYYMMDD_HHMMSS.webm
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

# En este equipo PyQt5 antes de OpenCV/MVSDK ha sido más estable.
if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
    os.environ.setdefault("QT_QPA_PLATFORM", "wayland")

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QSizePolicy,
)

import cv2
import numpy as np
import mvsdk


# ============================================================
# PERFIL DE CÁMARA VALIDADO
# ============================================================

CAMERA_INDEX = 0

EXPOSURE_MS = 8.33

R_GAIN = 142
G_GAIN = 71
B_GAIN = 120


# ============================================================
# VIDEO
# ============================================================

OUTPUT_DIR = Path("videos_mindvision")

# Priorizamos calidad y resolución sobre fluidez.
RECORD_FPS = 10

# Calidad alta para VP8.
VP8_CRF = 8

# Bitrate generoso para preservar detalle fino.
VP8_BITRATE = "12M"

VIDEO_EXTENSION = ".webm"


# ============================================================
# CÁMARA MINDVISION
# ============================================================

class MindVisionCamera:
    def __init__(
        self,
        camera_index=0,
        exposure_ms=8.33,
        r_gain=142,
        g_gain=71,
        b_gain=120,
    ):
        self.camera_index = int(camera_index)
        self.exposure_ms = float(exposure_ms)
        self.r_gain = int(r_gain)
        self.g_gain = int(g_gain)
        self.b_gain = int(b_gain)

        self.h_camera = None
        self.capability = None
        self.frame_buffer = None

        self.channels = 3
        self.mono = False
        self.opened = False

        self.camera_description = "NA"
        self.applied = {}

    def open(self):
        devices = mvsdk.CameraEnumerateDevice()

        if not devices:
            raise RuntimeError(
                "No se detectó ninguna cámara MindVision.\n\n"
                "Verifique que la cámara esté conectada por USB."
            )

        if self.camera_index >= len(devices):
            raise RuntimeError(
                f"Índice de cámara inválido: {self.camera_index}.\n"
                f"Cámaras detectadas: {len(devices)}"
            )

        device = devices[self.camera_index]
        self.camera_description = str(device)

        self.h_camera = mvsdk.CameraInit(device, -1, -1)
        self.capability = mvsdk.CameraGetCapability(self.h_camera)

        self.mono = self.capability.sIspCapacity.bMonoSensor != 0

        if self.mono:
            self.channels = 1
            mvsdk.CameraSetIspOutFormat(
                self.h_camera,
                mvsdk.CAMERA_MEDIA_TYPE_MONO8
            )
        else:
            self.channels = 3
            mvsdk.CameraSetIspOutFormat(
                self.h_camera,
                mvsdk.CAMERA_MEDIA_TYPE_BGR8
            )

        # Captura continua.
        mvsdk.CameraSetTriggerMode(self.h_camera, 0)

        # Autoexposición OFF.
        mvsdk.CameraSetAeState(self.h_camera, 0)

        # Exposición fija.
        mvsdk.CameraSetExposureTime(
            self.h_camera,
            self.exposure_ms * 1000.0
        )

        # Auto white balance OFF.
        if hasattr(mvsdk, "CameraSetWbMode"):
            try:
                mvsdk.CameraSetWbMode(
                    self.h_camera,
                    0
                )
            except Exception:
                pass

        # Ganancias RGB directamente en el ISP.
        if hasattr(mvsdk, "CameraSetGain"):
            mvsdk.CameraSetGain(
                self.h_camera,
                self.r_gain,
                self.g_gain,
                self.b_gain
            )
        else:
            raise RuntimeError(
                "Esta versión de mvsdk no expone CameraSetGain()."
            )

        # Iniciar adquisición.
        mvsdk.CameraPlay(self.h_camera)

        max_width = self.capability.sResolutionRange.iWidthMax
        max_height = self.capability.sResolutionRange.iHeightMax

        buffer_size = max_width * max_height * self.channels

        self.frame_buffer = mvsdk.CameraAlignMalloc(
            buffer_size,
            16
        )

        self.opened = True
        time.sleep(0.25)

        self.applied = self.readback_settings()

    def readback_settings(self):
        data = {
            "camera_index": self.camera_index,
            "camera_description": self.camera_description,
            "exposure_ms": self.exposure_ms,
            "r_gain": self.r_gain,
            "g_gain": self.g_gain,
            "b_gain": self.b_gain,
        }

        if self.h_camera is None:
            return data

        try:
            if hasattr(mvsdk, "CameraGetExposureTime"):
                data["exposure_ms"] = round(
                    float(
                        mvsdk.CameraGetExposureTime(
                            self.h_camera
                        )
                    ) / 1000.0,
                    6
                )
        except Exception:
            pass

        try:
            if hasattr(mvsdk, "CameraGetGain"):
                rg, gg, bg = mvsdk.CameraGetGain(
                    self.h_camera
                )

                data["r_gain"] = int(rg)
                data["g_gain"] = int(gg)
                data["b_gain"] = int(bg)
        except Exception:
            pass

        return data

    def read(self, timeout_ms=200):
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
                    frame_head.iHeight,
                    frame_head.iWidth
                )

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_GRAY2BGR
                )
            else:
                frame = frame.reshape(
                    frame_head.iHeight,
                    frame_head.iWidth,
                    3
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
                -12
            )

            if error_code != timeout_code:
                print(
                    f"Error MindVision: {exc}"
                )

            return False, None

        except Exception as exc:
            print(
                f"Error leyendo cámara: {exc}"
            )

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


# ============================================================
# GRABADOR WEBM / VP8
# ============================================================

class VP8VideoWriter:
    """
    Envía frames BGR RAW por stdin a FFmpeg.
    FFmpeg genera WebM usando libvpx/VP8.
    """

    def __init__(
        self,
        filename,
        width,
        height,
        fps=10,
        crf=8,
        bitrate="12M",
    ):
        self.filename = str(filename)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.crf = int(crf)
        self.bitrate = str(bitrate)

        self.process = None

        ffmpeg = shutil.which("ffmpeg")

        if not ffmpeg:
            raise RuntimeError(
                "FFmpeg no está instalado."
            )

        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel", "error",

            # Entrada desde Python: BGR sin comprimir.
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", str(self.fps),
            "-i", "-",

            # Sin audio.
            "-an",

            # WebM / VP8.
            "-c:v", "libvpx",

            # Parámetros pensados para Jetson.
            "-deadline", "realtime",
            "-cpu-used", "4",

            # Calidad alta.
            "-crf", str(self.crf),
            "-b:v", self.bitrate,

            # Compatibilidad navegador.
            "-pix_fmt", "yuv420p",

            self.filename,
        ]

        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=10**7,
        )

        # Detectar un fallo inmediato del encoder.
        time.sleep(0.15)

        if self.process.poll() is not None:
            detail = ""

            if self.process.stderr is not None:
                detail = self.process.stderr.read().decode(
                    "utf-8",
                    errors="ignore"
                )

            raise RuntimeError(
                "FFmpeg no pudo iniciar la codificación WebM/VP8.\n\n"
                + detail[-1500:]
            )

    def write(self, frame):
        if (
            self.process is None
            or self.process.stdin is None
            or self.process.poll() is not None
        ):
            raise RuntimeError(
                "El grabador WebM/VP8 dejó de funcionar."
            )

        if (
            frame.shape[1] != self.width
            or frame.shape[0] != self.height
        ):
            raise RuntimeError(
                "Cambió la resolución de la cámara durante la grabación."
            )

        try:
            self.process.stdin.write(
                frame.tobytes()
            )

        except BrokenPipeError:
            detail = ""

            try:
                if self.process.stderr is not None:
                    detail = self.process.stderr.read().decode(
                        "utf-8",
                        errors="ignore"
                    )
            except Exception:
                pass

            raise RuntimeError(
                "FFmpeg interrumpió la grabación.\n\n"
                + detail[-1500:]
            )

    def release(self):
        if self.process is None:
            return

        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
        except Exception:
            pass

        try:
            self.process.wait(
                timeout=20
            )
        except subprocess.TimeoutExpired:
            self.process.terminate()

            try:
                self.process.wait(
                    timeout=5
                )
            except Exception:
                self.process.kill()

        return_code = self.process.returncode

        if return_code not in (0, None):
            detail = ""

            try:
                if self.process.stderr is not None:
                    detail = self.process.stderr.read().decode(
                        "utf-8",
                        errors="ignore"
                    )
            except Exception:
                pass

            raise RuntimeError(
                "FFmpeg terminó con error.\n\n"
                + detail[-1500:]
            )

        self.process = None


# ============================================================
# APLICACIÓN
# ============================================================

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.camera = MindVisionCamera(
            camera_index=CAMERA_INDEX,
            exposure_ms=EXPOSURE_MS,
            r_gain=R_GAIN,
            g_gain=G_GAIN,
            b_gain=B_GAIN,
        )

        self.timer = QTimer(self)
        self.timer.timeout.connect(
            self.update_frame
        )

        self.writer = None
        self.recording = False
        self.recording_filename = None

        self.last_frame = None

        self.record_start_time = None
        self.frame_counter = 0

        self.last_recorded_frame_time = None
        self.record_interval = 1.0 / RECORD_FPS

        screen = QApplication.primaryScreen()
        geometry = screen.availableGeometry()

        self.screen_width = geometry.width()
        self.screen_height = geometry.height()

        scale_w = self.screen_width / 1280.0
        scale_h = self.screen_height / 800.0

        self.ui_scale = max(
            0.75,
            min(
                scale_w,
                scale_h,
                1.60
            )
        )

        self.init_ui()
        self.open_camera()

    def px(self, value):
        return max(
            1,
            int(
                value * self.ui_scale
            )
        )

    def init_ui(self):
        self.setWindowTitle(
            "Cámara MindVision"
        )

        self.setMinimumSize(
            640,
            400
        )

        central = QWidget()
        self.setCentralWidget(
            central
        )

        main_layout = QVBoxLayout(
            central
        )

        main_layout.setContentsMargins(
            self.px(10),
            self.px(10),
            self.px(10),
            self.px(8)
        )

        main_layout.setSpacing(
            self.px(8)
        )

        # ----------------------------------------------------
        # VISTA PREVIA
        # ----------------------------------------------------

        self.video_label = QLabel(
            "Conectando cámara..."
        )

        self.video_label.setAlignment(
            Qt.AlignCenter
        )

        self.video_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        self.video_label.setStyleSheet(
            f"""
            QLabel {{
                background: #000000;
                color: white;
                border-radius: {self.px(8)}px;
                font-size: {self.px(22)}px;
            }}
            """
        )

        main_layout.addWidget(
            self.video_label,
            1
        )

        # ----------------------------------------------------
        # CONTROLES
        # ----------------------------------------------------

        bottom_bar = QWidget()

        bottom_layout = QHBoxLayout(
            bottom_bar
        )

        bottom_layout.setContentsMargins(
            self.px(6),
            self.px(4),
            self.px(6),
            self.px(4)
        )

        bottom_layout.setSpacing(
            self.px(12)
        )

        self.status_label = QLabel(
            "LISTO"
        )

        self.status_label.setAlignment(
            Qt.AlignCenter
        )

        self.status_label.setFont(
            QFont(
                "Arial",
                self.px(17),
                QFont.Bold
            )
        )

        self.status_label.setMinimumWidth(
            self.px(170)
        )

        self.timer_label = QLabel(
            "00:00"
        )

        self.timer_label.setAlignment(
            Qt.AlignCenter
        )

        self.timer_label.setFont(
            QFont(
                "Arial",
                self.px(23),
                QFont.Bold
            )
        )

        self.timer_label.setMinimumWidth(
            self.px(120)
        )

        self.record_button = QPushButton(
            "●  GRABAR"
        )

        self.record_button.setMinimumHeight(
            self.px(72)
        )

        self.record_button.setMinimumWidth(
            self.px(260)
        )

        self.record_button.setFont(
            QFont(
                "Arial",
                self.px(20),
                QFont.Bold
            )
        )

        self.record_button.clicked.connect(
            self.toggle_recording
        )

        self.set_record_button_style(
            False
        )

        bottom_layout.addWidget(
            self.status_label,
            1
        )

        bottom_layout.addWidget(
            self.timer_label,
            0
        )

        bottom_layout.addWidget(
            self.record_button,
            1
        )

        main_layout.addWidget(
            bottom_bar,
            0
        )

        self.file_label = QLabel(
            f"WebM · VP8 · {RECORD_FPS} FPS · videos_mindvision/"
        )

        self.file_label.setAlignment(
            Qt.AlignCenter
        )

        self.file_label.setWordWrap(
            True
        )

        self.file_label.setStyleSheet(
            f"font-size: {self.px(12)}px; "
            "color: #555555; "
            f"padding: {self.px(2)}px;"
        )

        main_layout.addWidget(
            self.file_label,
            0
        )

    def set_record_button_style(
        self,
        recording
    ):
        radius = self.px(34)

        if recording:
            self.record_button.setText(
                "■  DETENER"
            )

            self.record_button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: #C62828;
                    color: white;
                    border: none;
                    border-radius: {radius}px;
                    padding: {self.px(10)}px {self.px(24)}px;
                }}

                QPushButton:pressed {{
                    background-color: #8E0000;
                }}
                """
            )

        else:
            self.record_button.setText(
                "●  GRABAR"
            )

            self.record_button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: #E53935;
                    color: white;
                    border: none;
                    border-radius: {radius}px;
                    padding: {self.px(10)}px {self.px(24)}px;
                }}

                QPushButton:pressed {{
                    background-color: #B71C1C;
                }}
                """
            )

    def open_camera(self):
        try:
            self.camera.open()

            applied = self.camera.applied

            self.status_label.setText(
                "CÁMARA LISTA"
            )

            self.file_label.setText(
                f"{applied.get('exposure_ms')} ms · "
                f"R {applied.get('r_gain')} / "
                f"G {applied.get('g_gain')} / "
                f"B {applied.get('b_gain')} · "
                f"WebM / VP8 · {RECORD_FPS} FPS"
            )

            self.timer.start(10)

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Error de cámara",
                str(exc)
            )

            self.status_label.setText(
                "SIN CÁMARA"
            )

            self.record_button.setEnabled(
                False
            )

    def update_frame(self):
        ok, frame = self.camera.read(
            timeout_ms=200
        )

        if not ok or frame is None:
            return

        self.last_frame = frame
        now = time.perf_counter()

        # ----------------------------------------------------
        # GRABACIÓN A 10 FPS CONSERVANDO TIEMPO REAL
        # ----------------------------------------------------

        if (
            self.recording
            and self.writer is not None
        ):
            should_write = False

            if self.last_recorded_frame_time is None:
                should_write = True
            elif (
                now - self.last_recorded_frame_time
                >= self.record_interval
            ):
                should_write = True

            if should_write:
                try:
                    self.writer.write(
                        frame
                    )

                    self.frame_counter += 1

                    if self.last_recorded_frame_time is None:
                        self.last_recorded_frame_time = now
                    else:
                        self.last_recorded_frame_time += (
                            self.record_interval
                        )

                        if (
                            now - self.last_recorded_frame_time
                            > self.record_interval * 2
                        ):
                            self.last_recorded_frame_time = now

                except Exception as exc:
                    self.abort_recording(
                        exc
                    )

            if self.recording:
                elapsed = (
                    time.time()
                    - self.record_start_time
                )

                minutes = int(
                    elapsed // 60
                )

                seconds = int(
                    elapsed % 60
                )

                self.timer_label.setText(
                    f"{minutes:02d}:{seconds:02d}"
                )

        # ----------------------------------------------------
        # VISTA PREVIA
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        h, w, ch = rgb.shape

        image = QImage(
            rgb.data,
            w,
            h,
            ch * w,
            QImage.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(
            image
        )

        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if self.last_frame is None:
            QMessageBox.warning(
                self,
                "Cámara",
                "Todavía no se recibió imagen de la cámara."
            )
            return

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        h, w = self.last_frame.shape[:2]

        timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = OUTPUT_DIR / (
            f"video_{timestamp}{VIDEO_EXTENSION}"
        )

        try:
            writer = VP8VideoWriter(
                filename=filename,
                width=w,
                height=h,
                fps=RECORD_FPS,
                crf=VP8_CRF,
                bitrate=VP8_BITRATE,
            )

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Error de grabación",
                str(exc)
            )
            return

        self.writer = writer
        self.recording = True
        self.recording_filename = filename

        self.record_start_time = time.time()
        self.frame_counter = 0
        self.last_recorded_frame_time = None

        self.timer_label.setText(
            "00:00"
        )

        self.status_label.setText(
            "● GRABANDO"
        )

        self.status_label.setStyleSheet(
            "color: #D32F2F;"
        )

        self.file_label.setText(
            f"Grabando WebM/VP8 · {filename.name}"
        )

        self.set_record_button_style(
            True
        )

    def stop_recording(self):
        filename = self.recording_filename

        # Detener escritura antes de cerrar FFmpeg.
        self.recording = False

        writer = self.writer
        self.writer = None

        self.recording_filename = None
        self.last_recorded_frame_time = None

        error = None

        if writer is not None:
            try:
                writer.release()
            except Exception as exc:
                error = exc

        self.set_record_button_style(
            False
        )

        if error is not None:
            self.status_label.setText(
                "ERROR AL GUARDAR"
            )

            self.status_label.setStyleSheet(
                "color: #B71C1C;"
            )

            QMessageBox.critical(
                self,
                "Error finalizando video",
                str(error)
            )

            return

        self.status_label.setText(
            "VIDEO GUARDADO"
        )

        self.status_label.setStyleSheet(
            "color: #1B5E20;"
        )

        if filename is not None:
            self.file_label.setText(
                f"Guardado: {filename.name}"
            )

    def abort_recording(
        self,
        error
    ):
        self.recording = False

        writer = self.writer
        self.writer = None

        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass

        self.set_record_button_style(
            False
        )

        self.status_label.setText(
            "ERROR GRABACIÓN"
        )

        self.status_label.setStyleSheet(
            "color: #B71C1C;"
        )

        QMessageBox.critical(
            self,
            "Error durante la grabación",
            str(error)
        )

    def keyPressEvent(
        self,
        event
    ):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(
                event
            )

    def closeEvent(
        self,
        event
    ):
        if self.recording:
            self.stop_recording()

        self.timer.stop()
        self.camera.release()

        event.accept()


def main():
    app = QApplication(
        sys.argv
    )

    app.setStyle(
        "Fusion"
    )

    window = CameraApp()
    window.showMaximized()

    sys.exit(
        app.exec_()
    )


if __name__ == "__main__":
    main()