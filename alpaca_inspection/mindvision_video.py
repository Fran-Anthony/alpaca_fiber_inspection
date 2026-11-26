#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mindvision_video.py

Visor simple de video para una cámara MindVision usando MVSDK.

No incluye:
- YOLO
- Arduino / motor
- PyQt5
- calibración automática
- corrección de color

Controles:
- q o ESC: salir
- s: guardar una captura PNG
"""

import os
import time
from ctypes import c_char_p, c_int
import cv2
import numpy as np
from .paths import RUNTIME_DIR

try:
    import mvsdk
except ModuleNotFoundError as exc:
    if exc.name != "mvsdk":
        raise
    from . import mvsdk

CAMERA_INDEX = 0
EXPOSURE_MS = 8.33
WINDOW_NAME = "MindVision - Video"


class MindVisionCamera:
    def __init__(self, camera_index=0, exposure_ms=8.33, rgb_gain=None):
        self.camera_index = camera_index
        self.exposure_ms = exposure_ms
        self.rgb_gain = rgb_gain

        self.h_camera = None
        self.capability = None
        self.frame_buffer = None

        self.channels = 3
        self.mono = False
        self.opened = False

    def open(self):
        if self.is_opened():
            return True
        try:
            self._open()
        except Exception:
            self.release()
            raise
        return True

    def _open(self):
        set_data_directory = getattr(getattr(mvsdk, "_sdk", None), "CameraSetDataDirectory", None)
        if set_data_directory is not None:
            data_dir = RUNTIME_DIR / "Camera"
            data_dir.mkdir(parents=True, exist_ok=True)
            set_data_directory.argtypes = [c_char_p]
            set_data_directory.restype = c_int
            status = set_data_directory(os.fsencode(data_dir))
            if status != 0:
                raise mvsdk.CameraException(status)

        print("Buscando cámaras MindVision...")

        devices = mvsdk.CameraEnumerateDevice()

        if not devices:
            raise RuntimeError(
                "No se detectó ninguna cámara MindVision mediante MVSDK. "
                "Comprueba primero que Linux la detecte con lsusb."
            )

        print(f"Cámaras detectadas: {len(devices)}")
        for i, device in enumerate(devices):
            print(f"  [{i}] {device}")

        if not 0 <= self.camera_index < len(devices):
            raise RuntimeError(
                f"Índice inválido: {self.camera_index}. "
                f"Detectadas: {len(devices)}"
            )

        device = devices[self.camera_index]
        print(f"Abriendo cámara [{self.camera_index}]...")

        self.h_camera = mvsdk.CameraInit(device, -1, -1)
        self.capability = mvsdk.CameraGetCapability(self.h_camera)

        self.mono = self.capability.sIspCapacity.bMonoSensor != 0

        if self.mono:
            self.channels = 1
            mvsdk.CameraSetIspOutFormat(
                self.h_camera,
                mvsdk.CAMERA_MEDIA_TYPE_MONO8
            )
            print("Sensor monocromático")
        else:
            self.channels = 3
            mvsdk.CameraSetIspOutFormat(
                self.h_camera,
                mvsdk.CAMERA_MEDIA_TYPE_BGR8
            )
            print("Sensor color BGR8")

        # Modo continuo
        mvsdk.CameraSetTriggerMode(self.h_camera, 0)

        # Exposición manual
        mvsdk.CameraSetAeState(self.h_camera, 0)
        mvsdk.CameraSetExposureTime(
            self.h_camera,
            self.exposure_ms * 1000.0
        )

        if not self.mono and self.rgb_gain is not None:
            if hasattr(mvsdk, "CameraSetWbMode"):
                try:
                    mvsdk.CameraSetWbMode(self.h_camera, 0)
                except mvsdk.CameraException as exc:
                    print(f"No se pudo fijar el balance de blancos manual: {exc}")
            mvsdk.CameraSetGain(self.h_camera, *self.rgb_gain)

        mvsdk.CameraPlay(self.h_camera)

        max_width = self.capability.sResolutionRange.iWidthMax
        max_height = self.capability.sResolutionRange.iHeightMax
        buffer_size = max_width * max_height * self.channels

        self.frame_buffer = mvsdk.CameraAlignMalloc(
            buffer_size,
            16
        )

        if not self.frame_buffer:
            raise RuntimeError("MVSDK no pudo reservar el búfer de captura.")

        self.opened = True
        time.sleep(0.3)

        print(f"Resolución máxima: {max_width} x {max_height}")
        print(f"Exposición: {self.exposure_ms:.2f} ms")
        print("Cámara abierta correctamente.")

    def is_opened(self):
        return self.opened and self.h_camera is not None

    def isOpened(self):
        return self.is_opened()

    def capture_raw_frame_for_calibration(self, timeout_ms=1000):
        return self.read(timeout_ms)

    def apply_once_wb(self):
        mvsdk.CameraSetOnceWB(self.h_camera)

    def read(self, timeout_ms=1000):
        if not self.is_opened():
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
            error_code = getattr(exc, "error_code", None)
            timeout_code = getattr(
                mvsdk,
                "CAMERA_STATUS_TIME_OUT",
                None
            )

            if error_code == timeout_code:
                print("Timeout esperando frame.")
            else:
                print(f"Error MVSDK: {exc}")

            return False, None

        except Exception as exc:
            print(f"Error procesando frame: {exc}")
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
        if self.frame_buffer:
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
        print("Cámara MindVision liberada.")


def main():
    camera = MindVisionCamera(
        camera_index=CAMERA_INDEX,
        exposure_ms=EXPOSURE_MS
    )

    try:
        camera.open()

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL
        )

        last_time = time.perf_counter()
        fps_smoothed = 0.0
        failures = 0
        first_frame = True

        print()
        print("Video iniciado.")
        print("Controles:")
        print("  q / ESC : salir")
        print("  s       : guardar captura")
        print()

        while True:
            ok, frame = camera.read(timeout_ms=1000)

            if not ok or frame is None:
                failures += 1

                if failures >= 10:
                    raise RuntimeError(
                        "Se perdieron 10 frames consecutivos."
                    )

                continue

            failures = 0

            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            if dt > 0:
                fps = 1.0 / dt
                if fps_smoothed == 0.0:
                    fps_smoothed = fps
                else:
                    fps_smoothed = (
                        0.90 * fps_smoothed
                        + 0.10 * fps
                    )

            if first_frame:
                h, w = frame.shape[:2]
                print(f"Frame recibido: {w} x {h}")

                max_display_w = 1280
                max_display_h = 800

                scale = min(
                    max_display_w / w,
                    max_display_h / h,
                    1.0
                )

                cv2.resizeWindow(
                    WINDOW_NAME,
                    max(1, int(w * scale)),
                    max(1, int(h * scale))
                )

                first_frame = False

            display = frame.copy()

            cv2.putText(
                display,
                f"MindVision | {fps_smoothed:.1f} FPS | Exposure {EXPOSURE_MS:.2f} ms",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

            elif key == ord("s"):
                filename = time.strftime(
                    "mindvision_capture_%Y%m%d_%H%M%S.png"
                )

                RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
                filename = str(RUNTIME_DIR / filename)
                if cv2.imwrite(filename, frame):
                    print(f"Captura guardada: {filename}")
                else:
                    print("No se pudo guardar la captura.")

    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")

    except Exception as exc:
        print(f"\nERROR: {exc}")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
