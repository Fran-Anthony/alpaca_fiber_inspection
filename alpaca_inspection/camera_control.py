"""Selección y control de la cámara de inspección."""
import os
import platform
import subprocess

import cv2


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


def create_camera():
    if os.environ.get("ALPACA_CAMERA_BACKEND", "mindvision").lower() == "uvc":
        return USBCamera()
    from .mindvision_video import MindVisionCamera

    return MindVisionCamera(rgb_gain=(142, 71, 120))
