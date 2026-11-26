#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
camera_app_con_faja.py
======================

Aplicación táctil para:
  1) visualizar y grabar video MindVision;
  2) controlar una faja transportadora mediante Arduino + L298N.

VIDEO
-----
- MindVision mediante MVSDK
- Calibración directamente en ISP:
    Exposure = 8.33 ms
    R = 142
    G = 71
    B = 120
    AE = OFF
    AWB = OFF
- Grabación: WebM / VP8 / libvpx
- 10 FPS, alta calidad

FAJA
----
Comunicación serial con Arduino a 115200 baud.
Comandos enviados:
    STOP
    START
    FORWARD
    REVERSE
    SPEED:<0..255>

IMPORTANTE:
El programa Python NO controla directamente los pines del L298N.
Envía comandos de texto al Arduino. El firmware del Arduino debe interpretar
estos comandos y accionar el L298N.

Salida:
    videos_mindvision/video_YYYYMMDD_HHMMSS.webm
"""

import os
import sys
import time
import glob
import shutil
import subprocess
from pathlib import Path

if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
    os.environ.setdefault("QT_QPA_PLATFORM", "wayland")

from PyQt5.QtCore import Qt, QTimer, QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QMessageBox, QSizePolicy,
    QComboBox, QSlider, QGroupBox
)

import cv2
import numpy as np
import mvsdk

try:
    from ultralytics import YOLO
    YOLO_IMPORT_ERROR = None
except Exception as exc:
    YOLO = None
    YOLO_IMPORT_ERROR = exc

try:
    import serial
    SERIAL_IMPORT_ERROR = None
except Exception as exc:
    serial = None
    SERIAL_IMPORT_ERROR = exc


# ============================================================
# CÁMARA
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
RECORD_FPS = 10
VP8_CRF = 8
VP8_BITRATE = "12M"
VIDEO_EXTENSION = ".webm"


# ============================================================
# DETECCIÓN VISUAL YOLO
# ============================================================

DETECTION_CONF = 0.50
DETECTION_IMGSZ = 416

# En la instalación actual el modelo está en ~/alpaca_usb_demo/best.pt.
# También se busca un best.pt junto a esta aplicación por portabilidad.
YOLO_MODEL_CANDIDATES = [
    Path(__file__).resolve().parent / "best.pt",
    Path("/home/cite/alpaca_usb_demo/best.pt"),
]


# ============================================================
# FAJA / MOTOR
# ============================================================

SERIAL_BAUD = 115200
DEFAULT_PWM = 90
DEFAULT_DIRECTION = "FORWARD"


class MindVisionCamera:
    def __init__(
        self,
        camera_index=CAMERA_INDEX,
        exposure_ms=EXPOSURE_MS,
        r_gain=R_GAIN,
        g_gain=G_GAIN,
        b_gain=B_GAIN,
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
        self.applied = {}

    def open(self):
        devices = mvsdk.CameraEnumerateDevice()

        if not devices:
            raise RuntimeError("No se detectó ninguna cámara MindVision.")

        if self.camera_index >= len(devices):
            raise RuntimeError(
                f"Índice de cámara inválido: {self.camera_index}. "
                f"Detectadas: {len(devices)}"
            )

        device = devices[self.camera_index]

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

        # Captura continua + automatismos desactivados.
        mvsdk.CameraSetTriggerMode(self.h_camera, 0)
        mvsdk.CameraSetAeState(self.h_camera, 0)

        mvsdk.CameraSetExposureTime(
            self.h_camera,
            self.exposure_ms * 1000.0
        )

        if hasattr(mvsdk, "CameraSetWbMode"):
            try:
                mvsdk.CameraSetWbMode(self.h_camera, 0)
            except Exception:
                pass

        if not hasattr(mvsdk, "CameraSetGain"):
            raise RuntimeError("mvsdk no expone CameraSetGain().")

        mvsdk.CameraSetGain(
            self.h_camera,
            self.r_gain,
            self.g_gain,
            self.b_gain
        )

        mvsdk.CameraPlay(self.h_camera)

        max_width = self.capability.sResolutionRange.iWidthMax
        max_height = self.capability.sResolutionRange.iHeightMax

        self.frame_buffer = mvsdk.CameraAlignMalloc(
            max_width * max_height * self.channels,
            16
        )

        self.opened = True
        time.sleep(0.25)
        self.applied = self.readback_settings()

    def readback_settings(self):
        data = {
            "exposure_ms": self.exposure_ms,
            "r_gain": self.r_gain,
            "g_gain": self.g_gain,
            "b_gain": self.b_gain,
        }

        try:
            if hasattr(mvsdk, "CameraGetExposureTime"):
                data["exposure_ms"] = round(
                    float(mvsdk.CameraGetExposureTime(self.h_camera)) / 1000.0,
                    6
                )
        except Exception:
            pass

        try:
            if hasattr(mvsdk, "CameraGetGain"):
                rg, gg, bg = mvsdk.CameraGetGain(self.h_camera)
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

            frame = np.frombuffer(frame_data, dtype=np.uint8)

            if self.mono:
                frame = frame.reshape(
                    frame_head.iHeight,
                    frame_head.iWidth
                )
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame = frame.reshape(
                    frame_head.iHeight,
                    frame_head.iWidth,
                    3
                )

            return True, frame.copy()

        except mvsdk.CameraException as exc:
            if getattr(exc, "error_code", None) != getattr(
                mvsdk, "CAMERA_STATUS_TIME_OUT", -12
            ):
                print(f"Error MindVision: {exc}")
            return False, None

        except Exception as exc:
            print(f"Error leyendo cámara: {exc}")
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
                mvsdk.CameraAlignFree(self.frame_buffer)
            except Exception:
                pass
            self.frame_buffer = None

        if self.h_camera is not None:
            try:
                mvsdk.CameraUnInit(self.h_camera)
            except Exception:
                pass
            self.h_camera = None

        self.opened = False


class VP8VideoWriter:
    def __init__(
        self,
        filename,
        width,
        height,
        fps=RECORD_FPS,
        crf=VP8_CRF,
        bitrate=VP8_BITRATE,
    ):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("FFmpeg no está instalado.")

        self.filename = str(filename)
        self.width = int(width)
        self.height = int(height)

        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", str(int(fps)),
            "-i", "-",
            "-an",
            "-c:v", "libvpx",
            "-deadline", "realtime",
            "-cpu-used", "4",
            "-crf", str(int(crf)),
            "-b:v", str(bitrate),
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

        time.sleep(0.15)

        if self.process.poll() is not None:
            detail = self.process.stderr.read().decode(
                "utf-8", errors="ignore"
            )
            raise RuntimeError(
                "FFmpeg no pudo iniciar VP8:\n" + detail[-1200:]
            )

    def write(self, frame):
        if (
            self.process is None
            or self.process.stdin is None
            or self.process.poll() is not None
        ):
            raise RuntimeError("El grabador VP8 dejó de funcionar.")

        self.process.stdin.write(frame.tobytes())

    def release(self):
        if self.process is None:
            return

        try:
            if self.process.stdin:
                self.process.stdin.close()
        except Exception:
            pass

        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()

        if self.process.returncode not in (0, None):
            detail = ""
            try:
                detail = self.process.stderr.read().decode(
                    "utf-8", errors="ignore"
                )
            except Exception:
                pass
            raise RuntimeError(
                "FFmpeg terminó con error:\n" + detail[-1200:]
            )

        self.process = None


def draw_visual_detections(frame, detections):
    """
    Dibuja detecciones SOLO sobre una copia para la GUI.
    El frame original usado por el grabador nunca pasa por esta función.
    """
    colors = {
        0: (0, 0, 255),    # rojo
        1: (255, 0, 0),    # azul
    }

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        cls_id = det["class_id"]
        label = det["label"]
        conf = det["confidence"]

        color = colors.get(cls_id, (0, 255, 0))

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            2
        )

        text_y = max(th + 6, y1)

        cv2.rectangle(
            frame,
            (x1, text_y - th - 6),
            (x1 + tw + 6, text_y),
            color,
            -1
        )

        cv2.putText(
            frame,
            text,
            (x1 + 3, text_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return frame


class DetectorWorker(QObject):
    """
    YOLO corre en un QThread independiente.

    La GUI nunca espera a que termine una inferencia. Mientras YOLO está
    ocupado, la cámara sigue capturando y el grabador sigue recibiendo frames.
    """
    ready = pyqtSignal(str)
    result_ready = pyqtSignal(object, float)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.model = None

    @pyqtSlot()
    def initialize(self):
        if YOLO is None:
            detail = (
                str(YOLO_IMPORT_ERROR)
                if YOLO_IMPORT_ERROR
                else "Ultralytics no está disponible."
            )
            self.error.emit(
                "No se pudo importar Ultralytics/YOLO: " + detail
            )
            return

        model_path = None

        for candidate in YOLO_MODEL_CANDIDATES:
            if candidate.exists():
                model_path = candidate
                break

        if model_path is None:
            self.error.emit(
                "No se encontró best.pt. Rutas buscadas: "
                + ", ".join(str(p) for p in YOLO_MODEL_CANDIDATES)
            )
            return

        try:
            self.model = YOLO(str(model_path))
            self.ready.emit(str(model_path))
        except Exception as exc:
            self.model = None
            self.error.emit(
                f"No se pudo cargar el modelo YOLO: {exc}"
            )

    @pyqtSlot(object)
    def detect(self, frame):
        if self.model is None:
            self.error.emit("El modelo YOLO no está inicializado.")
            return

        try:
            t0 = time.perf_counter()

            results = self.model(
                frame,
                conf=DETECTION_CONF,
                imgsz=DETECTION_IMGSZ,  
                verbose=False
            )

            elapsed_ms = (
                time.perf_counter() - t0
            ) * 1000.0

            detections = []

            if results and len(results) > 0:
                names = results[0].names

                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    if isinstance(names, dict):
                        label = str(
                            names.get(cls_id, cls_id)
                        )
                    else:
                        label = str(names[cls_id])

                    detections.append(
                        {
                            "xyxy": (x1, y1, x2, y2),
                            "class_id": cls_id,
                            "label": label,
                            "confidence": conf,
                        }
                    )

            self.result_ready.emit(
                detections,
                elapsed_ms
            )

        except Exception as exc:
            self.error.emit(
                f"Error durante inferencia YOLO: {exc}"
            )


class CameraBeltApp(QMainWindow):
    detect_request = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        # Cámara
        self.camera = MindVisionCamera()
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_frame)

        # Video
        self.writer = None
        self.recording = False
        self.recording_filename = None
        self.record_start_time = None
        self.last_recorded_frame_time = None
        self.record_interval = 1.0 / RECORD_FPS
        self.last_frame = None

        # Detección visual: completamente independiente de la grabación.
        self.latest_detections = []
        self.last_detection_time = 0.0
        self.detector_ready = False
        self.detector_busy = False
        self.detector_thread = None
        self.detector_worker = None

        # Faja
        self.serial_port = None
        self.motor_running = False
        self.motor_speed = DEFAULT_PWM
        self.motor_direction = DEFAULT_DIRECTION

        self.motor_serial_timer = QTimer(self)
        self.motor_serial_timer.timeout.connect(self.read_motor_serial)
        self.motor_serial_timer.start(50)

        # Escala UI
        screen = QApplication.primaryScreen()
        geometry = screen.availableGeometry()

        scale_w = geometry.width() / 1280.0
        scale_h = geometry.height() / 800.0
        self.ui_scale = max(0.75, min(scale_w, scale_h, 1.60))

        self.build_ui()
        self.start_detector()
        self.find_motor_ports()
        self.open_camera()

    def px(self, value):
        return max(1, int(value * self.ui_scale))

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def build_ui(self):
        self.setWindowTitle("MindVision + Faja Transportadora")
        self.setMinimumSize(800, 480)

        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        # Márgenes compactos para pantallas tablet landscape (p.ej. 1024x600).
        root.setContentsMargins(
            self.px(5), self.px(5),
            self.px(5), self.px(5)
        )
        root.setSpacing(self.px(6))

        # ===================== IZQUIERDA: VIDEO =====================
        left = QVBoxLayout()
        left.setSpacing(self.px(5))

        self.video_label = QLabel("Conectando cámara...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        self.video_label.setStyleSheet(
            f"""
            QLabel {{
                background:#000;
                color:white;
                border-radius:{self.px(8)}px;
                font-size:{self.px(20)}px;
            }}
            """
        )
        left.addWidget(self.video_label, 1)

        video_bottom = QHBoxLayout()

        self.record_status = QLabel("CÁMARA LISTA")
        self.record_status.setAlignment(Qt.AlignCenter)
        self.record_status.setFont(
            QFont("Arial", self.px(15), QFont.Bold)
        )

        self.record_time = QLabel("00:00")
        self.record_time.setAlignment(Qt.AlignCenter)
        self.record_time.setFont(
            QFont("Arial", self.px(17), QFont.Bold)
        )

        self.record_button = QPushButton("●  GRABAR")
        self.record_button.setMinimumHeight(self.px(48))
        self.record_button.setMaximumHeight(self.px(54))
        self.record_button.setFont(
            QFont("Arial", self.px(15), QFont.Bold)
        )
        self.record_button.clicked.connect(self.toggle_recording)
        self.set_record_style(False)

        video_bottom.addWidget(self.record_status, 1)
        video_bottom.addWidget(self.record_time, 0)
        video_bottom.addWidget(self.record_button, 1)

        left.addLayout(video_bottom)

        self.video_info = QLabel(
            "WebM / VP8 · 10 FPS · videos_mindvision/"
        )
        self.video_info.setAlignment(Qt.AlignCenter)
        left.addWidget(self.video_info)

        root.addLayout(left, 4)

        # ===================== DERECHA: DETECCIÓN + FAJA =====================
        right = QVBoxLayout()
        right.setSpacing(self.px(4))
        right.setContentsMargins(0, 0, 0, 0)

        detection_box = QGroupBox("DETECCIÓN VISUAL — NO SE GRABA")
        detection_box.setFont(
            QFont("Arial", self.px(10), QFont.Bold)
        )
        detection_layout = QVBoxLayout(detection_box)
        detection_layout.setContentsMargins(
            self.px(6), self.px(5),
            self.px(6), self.px(5)
        )
        detection_layout.setSpacing(self.px(2))

        self.detection_status_label = QLabel("Cargando modelo YOLO...")
        self.detection_status_label.setAlignment(Qt.AlignCenter)
        self.detection_status_label.setWordWrap(True)
        self.detection_status_label.setStyleSheet(
            f"font-weight:bold;color:#455A64;font-size:{self.px(11)}px;"
        )
        detection_layout.addWidget(self.detection_status_label)

        self.detection_count_label = QLabel("Detecciones: --")
        self.detection_count_label.setAlignment(Qt.AlignCenter)
        self.detection_count_label.setFont(
            QFont("Arial", self.px(12), QFont.Bold)
        )
        detection_layout.addWidget(self.detection_count_label)

        self.detection_speed_label = QLabel(
            f"Conf. {DETECTION_CONF:.2f} · imgsz {DETECTION_IMGSZ}"
        )
        self.detection_speed_label.setAlignment(Qt.AlignCenter)
        self.detection_speed_label.setStyleSheet(
            f"font-size:{self.px(10)}px;"
        )
        detection_layout.addWidget(self.detection_speed_label)

        right.addWidget(detection_box)

        belt_box = QGroupBox("FAJA TRANSPORTADORA")
        belt_box.setMinimumWidth(self.px(285))
        belt_box.setFont(
            QFont("Arial", self.px(10), QFont.Bold)
        )
        belt = QVBoxLayout(belt_box)
        belt.setContentsMargins(
            self.px(6), self.px(6),
            self.px(6), self.px(6)
        )
        belt.setSpacing(self.px(4))

        title = QLabel("CONTROL DE MOVIMIENTO")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(
            QFont("Arial", self.px(13), QFont.Bold)
        )
        belt.addWidget(title)

        # Puerto
        port_row = QHBoxLayout()

        self.port_combo = QComboBox()
        self.port_combo.setMinimumHeight(self.px(32))
        self.port_combo.setMaximumHeight(self.px(36))
        self.port_combo.setFont(
            QFont("Arial", self.px(10))
        )

        self.refresh_port_button = QPushButton("BUSCAR")
        self.refresh_port_button.setMinimumHeight(self.px(32))
        self.refresh_port_button.setMaximumHeight(self.px(36))
        self.refresh_port_button.setFont(
            QFont("Arial", self.px(10), QFont.Bold)
        )
        self.refresh_port_button.clicked.connect(self.find_motor_ports)

        port_row.addWidget(self.port_combo, 1)
        port_row.addWidget(self.refresh_port_button)

        belt.addLayout(port_row)

        self.connect_button = QPushButton("CONECTAR ARDUINO")
        self.connect_button.setMinimumHeight(self.px(34))
        self.connect_button.setMaximumHeight(self.px(38))
        self.connect_button.setFont(
            QFont("Arial", self.px(11), QFont.Bold)
        )
        self.connect_button.clicked.connect(self.toggle_motor_connection)
        belt.addWidget(self.connect_button)

        self.connection_label = QLabel("Arduino: DESCONECTADO")
        self.connection_label.setAlignment(Qt.AlignCenter)
        self.connection_label.setWordWrap(True)
        self.connection_label.setStyleSheet(
            f"font-size:{self.px(10)}px;"
        )
        belt.addWidget(self.connection_label)

        # Iniciar / detener
        run_row = QHBoxLayout()

        self.start_belt_button = QPushButton("▶ INICIAR")
        self.stop_belt_button = QPushButton("■ DETENER")

        for btn in (self.start_belt_button, self.stop_belt_button):
            btn.setMinimumHeight(self.px(42))
            btn.setMaximumHeight(self.px(46))
            btn.setFont(
                QFont("Arial", self.px(13), QFont.Bold)
            )

        self.start_belt_button.clicked.connect(self.motor_start)
        self.stop_belt_button.clicked.connect(self.motor_stop)

        self.start_belt_button.setStyleSheet(
            "background:#2E7D32;color:white;border-radius:8px;"
        )
        self.stop_belt_button.setStyleSheet(
            "background:#B71C1C;color:white;border-radius:8px;"
        )

        run_row.addWidget(self.start_belt_button)
        run_row.addWidget(self.stop_belt_button)
        belt.addLayout(run_row)

        # Dirección
        direction_title = QLabel("DIRECCIÓN")
        direction_title.setAlignment(Qt.AlignCenter)
        direction_title.setFont(
            QFont("Arial", self.px(10), QFont.Bold)
        )
        direction_title.setMaximumHeight(self.px(18))
        belt.addWidget(direction_title)

        direction_row = QHBoxLayout()

        self.reverse_button = QPushButton("◀ REVERSA")
        self.forward_button = QPushButton("ADELANTE ▶")

        self.reverse_button.setMinimumHeight(self.px(34))
        self.reverse_button.setMaximumHeight(self.px(38))
        self.forward_button.setMinimumHeight(self.px(34))
        self.forward_button.setMaximumHeight(self.px(38))

        self.reverse_button.setFont(
            QFont("Arial", self.px(10), QFont.Bold)
        )
        self.forward_button.setFont(
            QFont("Arial", self.px(10), QFont.Bold)
        )

        self.reverse_button.clicked.connect(self.motor_reverse)
        self.forward_button.clicked.connect(self.motor_forward)

        direction_row.addWidget(self.reverse_button)
        direction_row.addWidget(self.forward_button)
        belt.addLayout(direction_row)

        # Velocidad
        speed_title = QLabel("VELOCIDAD")
        speed_title.setAlignment(Qt.AlignCenter)
        speed_title.setFont(
            QFont("Arial", self.px(10), QFont.Bold)
        )
        speed_title.setMaximumHeight(self.px(18))
        belt.addWidget(speed_title)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 255)
        self.speed_slider.setValue(DEFAULT_PWM)
        self.speed_slider.setMinimumHeight(self.px(26))
        self.speed_slider.setMaximumHeight(self.px(30))
        self.speed_slider.valueChanged.connect(self.motor_speed_changed)

        belt.addWidget(self.speed_slider)

        self.speed_label = QLabel()
        self.speed_label.setAlignment(Qt.AlignCenter)
        self.speed_label.setFont(
            QFont("Arial", self.px(10))
        )
        self.speed_label.setMaximumHeight(self.px(20))
        belt.addWidget(self.speed_label)
        self.motor_speed_changed(DEFAULT_PWM)

        # Estado motor
        self.motor_state_label = QLabel("FAJA DETENIDA")
        self.motor_state_label.setAlignment(Qt.AlignCenter)
        self.motor_state_label.setFont(
            QFont("Arial", self.px(12), QFont.Bold)
        )
        self.motor_state_label.setMinimumHeight(self.px(32))
        self.motor_state_label.setMaximumHeight(self.px(44))
        self.motor_state_label.setStyleSheet(
            "background:#ECEFF1;padding:4px;border-radius:6px;"
        )
        belt.addWidget(self.motor_state_label)

        self.motor_response_label = QLabel("Arduino: sin respuesta")
        self.motor_response_label.setAlignment(Qt.AlignCenter)
        self.motor_response_label.setWordWrap(True)
        self.motor_response_label.setFont(
            QFont("Arial", self.px(9))
        )
        self.motor_response_label.setMaximumHeight(self.px(34))
        belt.addWidget(self.motor_response_label)

        belt.addStretch(1)

        right.addWidget(belt_box, 1)
        root.addLayout(right, 1)

    def set_record_style(self, recording):
        if recording:
            self.record_button.setText("■  DETENER GRABACIÓN")
            self.record_button.setStyleSheet(
                "background:#B71C1C;color:white;border-radius:25px;"
            )
        else:
            self.record_button.setText("●  GRABAR")
            self.record_button.setStyleSheet(
                "background:#E53935;color:white;border-radius:25px;"
            )

    # --------------------------------------------------------
    # DETECCIÓN VISUAL
    # --------------------------------------------------------

    def start_detector(self):
        """
        Arranca YOLO al iniciar la aplicación.

        La inferencia vive en otro hilo para que una inferencia lenta no
        bloquee la captura MindVision ni la escritura del video.
        """
        self.detector_thread = QThread(self)
        self.detector_worker = DetectorWorker()

        self.detector_worker.moveToThread(
            self.detector_thread
        )

        self.detector_thread.started.connect(
            self.detector_worker.initialize
        )

        self.detector_worker.ready.connect(
            self.on_detector_ready
        )

        self.detector_worker.result_ready.connect(
            self.on_detection_result
        )

        self.detector_worker.error.connect(
            self.on_detector_error
        )

        self.detect_request.connect(
            self.detector_worker.detect
        )

        self.detector_thread.start()

    def on_detector_ready(self, model_path):
        self.detector_ready = True
        self.detector_busy = False

        self.detection_status_label.setText(
            "YOLO ACTIVO"
        )
        self.detection_status_label.setStyleSheet(
            "font-weight:bold;color:#1B5E20;"
        )

        self.detection_speed_label.setText(
            f"{Path(model_path).name} · "
            f"conf {DETECTION_CONF:.2f} · "
            f"imgsz {DETECTION_IMGSZ}"
        )

    def on_detection_result(
        self,
        detections,
        elapsed_ms
    ):
        self.latest_detections = detections
        self.last_detection_time = time.perf_counter()
        self.detector_busy = False

        self.detection_count_label.setText(
            f"Detecciones: {len(detections)}"
        )

        if elapsed_ms > 0:
            inference_fps = 1000.0 / elapsed_ms

            self.detection_speed_label.setText(
                f"Inferencia: {elapsed_ms:.0f} ms "
                f"({inference_fps:.1f} FPS YOLO)"
            )

    def on_detector_error(self, message):
        self.detector_busy = False

        # Si falló la carga inicial, no seguir enviando frames.
        if not self.detector_ready:
            self.detection_status_label.setText(
                "YOLO NO DISPONIBLE"
            )
            self.detection_status_label.setStyleSheet(
                "font-weight:bold;color:#B71C1C;"
            )

        self.detection_speed_label.setText(
            str(message)
        )

        print(message)

    def submit_frame_for_detection(self, frame):
        """
        No se crea una cola de frames.

        Si YOLO sigue procesando el frame anterior, este frame se omite para
        detección. Así la predicción siempre se mantiene cerca del tiempo real
        y nunca retrasa la grabación.
        """
        if (
            not self.detector_ready
            or self.detector_busy
            or self.detector_worker is None
        ):
            return

        self.detector_busy = True
        self.detect_request.emit(
            frame.copy()
        )

    # --------------------------------------------------------
    # CÁMARA / VIDEO
    # --------------------------------------------------------

    def open_camera(self):
        try:
            self.camera.open()
            applied = self.camera.applied

            self.video_info.setText(
                f"{applied.get('exposure_ms')} ms · "
                f"R {applied.get('r_gain')} / "
                f"G {applied.get('g_gain')} / "
                f"B {applied.get('b_gain')} · "
                f"WebM / VP8 · {RECORD_FPS} FPS"
            )

            self.camera_timer.start(10)

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Error de cámara",
                str(exc)
            )
            self.record_status.setText("SIN CÁMARA")
            self.record_button.setEnabled(False)

    def update_frame(self):
        ok, frame = self.camera.read(timeout_ms=200)

        if not ok or frame is None:
            return

        self.last_frame = frame
        now = time.perf_counter()

        # Grabación a 10 FPS manteniendo duración real.
        if self.recording and self.writer is not None:
            should_write = (
                self.last_recorded_frame_time is None
                or now - self.last_recorded_frame_time >= self.record_interval
            )

            if should_write:
                try:
                    self.writer.write(frame)

                    if self.last_recorded_frame_time is None:
                        self.last_recorded_frame_time = now
                    else:
                        self.last_recorded_frame_time += self.record_interval

                        if (
                            now - self.last_recorded_frame_time
                            > self.record_interval * 2
                        ):
                            self.last_recorded_frame_time = now

                except Exception as exc:
                    self.recording = False
                    QMessageBox.critical(
                        self,
                        "Error de grabación",
                        str(exc)
                    )

            elapsed = time.time() - self.record_start_time
            self.record_time.setText(
                f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
            )

        # Enviar el frame LIMPIO a YOLO en segundo plano.
        # La grabación ya se realizó arriba con ese mismo frame sin anotaciones.
        self.submit_frame_for_detection(frame)

        # Para la GUI usamos una COPIA sobre la que sí se dibujan boxes.
        display_frame = frame.copy()

        # Si por algún motivo la inferencia se detuvo hace varios segundos,
        # no dejamos boxes antiguos indefinidamente sobre la imagen.
        detections_for_display = self.latest_detections
        if (
            self.last_detection_time > 0
            and time.perf_counter() - self.last_detection_time > 2.0
        ):
            detections_for_display = []

        if detections_for_display:
            draw_visual_detections(
                display_frame,
                detections_for_display
            )

        rgb = cv2.cvtColor(
            display_frame,
            cv2.COLOR_BGR2RGB
        )
        h, w, ch = rgb.shape

        qimg = QImage(
            rgb.data,
            w, h,
            ch * w,
            QImage.Format_RGB888
        ).copy()

        pix = QPixmap.fromImage(qimg)

        self.video_label.setPixmap(
            pix.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if self.last_frame is None:
            QMessageBox.warning(
                self,
                "Cámara",
                "Todavía no hay una imagen válida."
            )
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        h, w = self.last_frame.shape[:2]

        filename = OUTPUT_DIR / (
            "video_"
            + time.strftime("%Y%m%d_%H%M%S")
            + VIDEO_EXTENSION
        )

        try:
            self.writer = VP8VideoWriter(
                filename,
                width=w,
                height=h
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Error de grabación",
                str(exc)
            )
            return

        self.recording = True
        self.recording_filename = filename
        self.record_start_time = time.time()
        self.last_recorded_frame_time = None

        self.record_time.setText("00:00")
        self.record_status.setText("● GRABANDO")
        self.record_status.setStyleSheet("color:#D32F2F;")
        self.video_info.setText(f"Grabando: {filename.name}")
        self.set_record_style(True)

    def stop_recording(self):
        self.recording = False

        writer = self.writer
        filename = self.recording_filename

        self.writer = None
        self.recording_filename = None
        self.last_recorded_frame_time = None

        try:
            if writer:
                writer.release()
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Error cerrando video",
                str(exc)
            )

        self.record_status.setText("VIDEO GUARDADO")
        self.record_status.setStyleSheet("color:#1B5E20;")
        self.set_record_style(False)

        if filename:
            self.video_info.setText(f"Guardado: {filename.name}")

    # --------------------------------------------------------
    # FAJA / ARDUINO
    # --------------------------------------------------------

    def find_motor_ports(self):
        self.port_combo.clear()

        if serial is None:
            self.port_combo.addItem("PySerial no disponible")
            return

        ports = sorted(
            glob.glob("/dev/ttyACM*")
            + glob.glob("/dev/ttyUSB*")
        )

        if ports:
            self.port_combo.addItems(ports)
        else:
            self.port_combo.addItem("No encontrado")

    def toggle_motor_connection(self):
        if serial is None:
            detail = (
                str(SERIAL_IMPORT_ERROR)
                if SERIAL_IMPORT_ERROR
                else "PySerial no está instalado."
            )

            QMessageBox.warning(
                self,
                "Arduino",
                "No se puede abrir el puerto serial.\n\n" + detail
            )
            return

        if self.serial_port is not None:
            self.disconnect_motor()
            return

        port = self.port_combo.currentText()

        if (
            not port
            or port in ("No encontrado", "PySerial no disponible")
        ):
            QMessageBox.warning(
                self,
                "Arduino",
                "No se encontró un puerto Arduino."
            )
            return

        try:
            self.serial_port = serial.Serial(
                port,
                SERIAL_BAUD,
                timeout=0.05
            )

            # Muchos Arduino reinician al abrir el puerto.
            time.sleep(1.8)

            self.serial_port.reset_input_buffer()

            self.connect_button.setText("DESCONECTAR")
            self.connection_label.setText(
                f"Arduino: CONECTADO\n{port}"
            )
            self.connection_label.setStyleSheet("color:#1B5E20;")

            # Seguridad: conectar siempre con la faja detenida.
            self.send_motor_command("STOP")
            self.motor_running = False
            self.update_motor_state()

        except Exception as exc:
            self.serial_port = None

            QMessageBox.critical(
                self,
                "Error conectando Arduino",
                str(exc)
            )

    def disconnect_motor(self):
        try:
            if self.serial_port is not None:
                # Seguridad: detener antes de desconectar.
                self.send_motor_command("STOP")
                self.serial_port.close()
        except Exception:
            pass

        self.serial_port = None
        self.motor_running = False

        self.connect_button.setText("CONECTAR ARDUINO")
        self.connection_label.setText("Arduino: DESCONECTADO")
        self.connection_label.setStyleSheet("")
        self.update_motor_state()

    def send_motor_command(self, command):
        if (
            self.serial_port is None
            or not self.serial_port.is_open
        ):
            return False

        try:
            self.serial_port.write(
                (command + "\n").encode("utf-8")
            )
            self.serial_port.flush()
            return True

        except Exception as exc:
            QMessageBox.warning(
                self,
                "Error serial",
                str(exc)
            )
            return False

    def require_motor_connection(self):
        if (
            self.serial_port is None
            or not self.serial_port.is_open
        ):
            QMessageBox.warning(
                self,
                "Arduino",
                "Primero conecte el Arduino."
            )
            return False

        return True

    def motor_start(self):
        if not self.require_motor_connection():
            return

        # El orden replica el programa recibido:
        # 1) velocidad
        # 2) dirección
        # 3) START
        self.send_motor_command(
            f"SPEED:{self.motor_speed}"
        )
        self.send_motor_command(
            self.motor_direction
        )
        self.send_motor_command(
            "START"
        )

        self.motor_running = True
        self.update_motor_state()

    def motor_stop(self):
        self.send_motor_command("STOP")
        self.motor_running = False
        self.update_motor_state()

    def motor_forward(self):
        self.motor_direction = "FORWARD"

        if self.require_motor_connection():
            self.send_motor_command("FORWARD")

        self.update_motor_state()

    def motor_reverse(self):
        self.motor_direction = "REVERSE"

        if self.require_motor_connection():
            self.send_motor_command("REVERSE")

        self.update_motor_state()

    def motor_speed_changed(self, value):
        self.motor_speed = int(value)
        pct = int(round(self.motor_speed * 100 / 255))

        self.speed_label.setText(
            f"PWM: {self.motor_speed}  |  {pct} %"
        )

        # En el programa original, si el motor ya está funcionando,
        # el nuevo PWM se envía inmediatamente.
        if (
            getattr(self, "motor_running", False)
            and self.serial_port is not None
        ):
            self.send_motor_command(
                f"SPEED:{self.motor_speed}"
            )

    def update_motor_state(self):
        if self.motor_running:
            direction_text = (
                "ADELANTE"
                if self.motor_direction == "FORWARD"
                else "REVERSA"
            )

            self.motor_state_label.setText(
                f"FAJA EN MOVIMIENTO\n{direction_text}"
            )

            self.motor_state_label.setStyleSheet(
                "background:#C8E6C9;"
                "color:#1B5E20;"
                "padding:4px;"
                "border-radius:6px;"
            )
        else:
            self.motor_state_label.setText(
                "FAJA DETENIDA"
            )

            self.motor_state_label.setStyleSheet(
                "background:#ECEFF1;"
                "color:#263238;"
                "padding:4px;"
                "border-radius:6px;"
            )

    def read_motor_serial(self):
        if (
            self.serial_port is None
            or not self.serial_port.is_open
        ):
            return

        try:
            while self.serial_port.in_waiting:
                line = self.serial_port.readline().decode(
                    "utf-8",
                    errors="ignore"
                ).strip()

                if line:
                    self.motor_response_label.setText(
                        f"Arduino: {line}"
                    )

        except Exception as exc:
            self.motor_response_label.setText(
                f"Error serial: {exc}"
            )

    # --------------------------------------------------------
    # CIERRE SEGURO
    # --------------------------------------------------------

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        # Detener grabación.
        if self.recording:
            self.stop_recording()

        # Detener faja ANTES de cerrar serial.
        try:
            self.motor_serial_timer.stop()

            if self.serial_port is not None:
                self.send_motor_command("STOP")
                self.serial_port.close()
                self.serial_port = None

        except Exception:
            pass

        self.camera_timer.stop()
        self.camera.release()

        # Finalizar el hilo de inferencia. Si está terminando una predicción,
        # esperamos brevemente para salir de forma limpia.
        try:
            if self.detector_thread is not None:
                self.detector_thread.quit()
                self.detector_thread.wait(5000)
        except Exception:
            pass

        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = CameraBeltApp()
    window.showMaximized()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()