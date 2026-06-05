from __future__ import annotations

import os
from dataclasses import dataclass
from glob import glob
from typing import Iterator, Union

import cv2

CameraSource = Union[int, str]


@dataclass
class CameraSession:
    """Small wrapper around ``cv2.VideoCapture`` with the resolved source."""

    capture: cv2.VideoCapture
    source: CameraSource
    backend: int

    def read(self):
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()


def parse_camera_source(value: CameraSource) -> CameraSource:
    """Normalize a CLI camera value into an integer index, device path, or ``auto``."""

    if isinstance(value, int):
        return value

    stripped = value.strip()
    if not stripped:
        return "auto"

    if stripped.lower() == "auto":
        return "auto"

    if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
        return int(stripped)

    return stripped


def _camera_backend(source: CameraSource) -> int:
    if isinstance(source, int):
        return cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY

    if isinstance(source, str) and source.startswith("/dev/"):
        return cv2.CAP_V4L2

    return cv2.CAP_ANY


def _open_single_source(source: CameraSource) -> CameraSession | None:
    backend = _camera_backend(source)
    capture = cv2.VideoCapture(
        source, backend) if backend != cv2.CAP_ANY else cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        return None

    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        return None

    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return CameraSession(capture=capture, source=source, backend=backend)


def _auto_candidates(max_index: int) -> Iterator[CameraSource]:
    if os.name != "nt":
        for device_path in sorted(glob("/dev/video*")):
            yield device_path

    for index in range(max_index + 1):
        yield index


def _resolve_board_camera(source: str) -> CameraSource:
    devices = sorted(glob("/sys/class/video4linux/video*"))
    csi_device = None
    usb_devices = []

    for dev_path in devices:
        try:
            name_file = os.path.join(dev_path, "name")
            if os.path.exists(name_file):
                with open(name_file, "r") as f:
                    name = f.read().strip()
                dev_num = os.path.basename(dev_path)
                full_path = f"/dev/{dev_num}"

                is_csi = (
                    "dcmi" in name.lower()
                    or "stm" in name.lower()
                    or "gc2145" in name.lower()
                    or "ov5640" in name.lower()
                    or "imx335" in name.lower()
                    or name == "dcmipp_dump_capture"
                    or name == "dcmipp_main_capture"
                )
                if is_csi:
                    csi_device = full_path
                else:
                    if "meta" not in name.lower():
                        usb_devices.append(full_path)
        except Exception:
            continue

    if source.lower() == "csi":
        if csi_device is not None:
            return csi_device
        if os.path.exists("/dev/video0"):
            return "/dev/video0"
        raise RuntimeError("CSI / hardware camera not found")

    if source.lower() == "usb":
        if usb_devices:
            return usb_devices[0]
        if os.path.exists("/dev/video1"):
            return "/dev/video1"
        if os.path.exists("/dev/video7"):
            return "/dev/video7"
        raise RuntimeError("USB camera not found")

    return source


def open_camera(source: CameraSource, auto_max_index: int = 10) -> CameraSession:
    """Open the requested camera source.

    ``source`` may be an integer index, a Linux V4L2 device path, or ``auto``.
    """

    resolved = parse_camera_source(source)
    if isinstance(resolved, str) and resolved.lower() in ("csi", "usb") and os.name != "nt":
        resolved = _resolve_board_camera(resolved)

    if resolved == "auto":
        for candidate in _auto_candidates(auto_max_index):
            session = _open_single_source(candidate)
            if session is not None:
                return session
        raise RuntimeError("No working camera source was found")

    session = _open_single_source(resolved)
    if session is None:
        raise RuntimeError(f"Unable to open camera source: {resolved}")
    return session


def configure_capture(
    capture: cv2.VideoCapture,
    width: int,
    height: int,
    fps: int | None = None,
) -> None:
    """Apply common camera settings without failing hard if a backend ignores them."""

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps is not None:
        capture.set(cv2.CAP_PROP_FPS, float(fps))
    capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
