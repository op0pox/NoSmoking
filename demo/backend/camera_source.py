"""
demo/backend/camera_source.py — 실시간 입력 소스 (이 PC 카메라 / 네트워크 카메라)

- 이 PC 카메라: 번호(0, 1, ...)로 연다. macOS 권한 요청 때문에 메인 스레드에서 열어야 한다.
- 네트워크 카메라: rtsp:// 또는 http(s):// 주소로 연다. IP 카메라·CCTV는 대부분 RTSP,
  폰 IP 카메라 앱은 HTTP(MJPEG)를 쓴다.

네트워크 카메라는 두 가지를 따로 처리한다.
1) 지연 누적 방지: 판정이 프레임 속도를 못 따라가면 카메라가 보낸 프레임이 쌓여 화면이 점점 늦어진다.
   별도 스레드가 계속 읽어 "가장 최근 프레임 한 장"만 남기고 나머지는 버린다.
2) 자동 재연결: 와이파이가 잠깐 끊겨도 몇 번 다시 연결해 본다.

개인정보 원칙: 같은 공유기 안(사설 IP)의 카메라만 허용해 영상이 인터넷의 외부 서버를 거치지 않게 한다.
프레임은 저장하지 않고, 최신 한 장만 메모리에 두고 덮어쓴다.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import threading
import time
from typing import Optional
from urllib.parse import urlparse, urlunparse

import cv2

# RTSP는 기본이 UDP라 와이파이에서 패킷이 빠지면 화면이 깨진다. TCP로 받는다.
# (OpenCV가 네트워크 영상을 열 때마다 이 환경 변수를 읽는다)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

ALLOWED_SCHEMES = ("rtsp", "http", "https")
OPEN_TIMEOUT_MSEC = 5000       # 네트워크 카메라 연결을 기다리는 최대 시간
READ_TIMEOUT_MSEC = 5000       # 프레임 하나를 기다리는 최대 시간
RECONNECT_ATTEMPTS = 5         # 끊겼을 때 다시 연결해 볼 횟수
RECONNECT_DELAY_SEC = 2.0      # 다시 연결하기 전 기다리는 시간


class SourceError(ValueError):
    """화면에 그대로 보여줄 수 있는 입력 소스 오류."""


def is_network_url(source: str) -> bool:
    return urlparse(source).scheme.lower() in ALLOWED_SCHEMES


def mask_credentials(url: str) -> str:
    """rtsp://아이디:비밀번호@주소 → rtsp://아이디:****@주소 (화면 표시용)."""
    parsed = urlparse(url)
    if parsed.password is None:
        return url
    netloc = f"{parsed.username}:****@{parsed.hostname}" + (f":{parsed.port}" if parsed.port else "")
    return urlunparse(parsed._replace(netloc=netloc))


def validate_network_url(url: str):
    """주소 형식과 "같은 공유기 안의 카메라인지"를 확인한다. 문제가 있으면 SourceError."""
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise SourceError("카메라 주소는 rtsp:// 또는 http:// 로 시작해야 합니다.")
    if not parsed.hostname:
        raise SourceError("카메라 주소에 IP나 호스트 이름이 없습니다.")
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, None)}
    except socket.gaierror:
        raise SourceError(f"카메라 주소의 호스트({parsed.hostname})를 찾을 수 없습니다.") from None
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if not (ip.is_private or ip.is_loopback or ip.is_link_local):
            raise SourceError(
                f"같은 공유기 안의 카메라만 연결할 수 있습니다 ({parsed.hostname} → {address}). "
                "영상이 인터넷의 외부 서버를 거치지 않게 하기 위한 제한입니다."
            )


def open_local_camera(index: int) -> cv2.VideoCapture:
    """이 PC 카메라를 연다. macOS 권한 요청 때문에 메인 스레드(이벤트 루프)에서 호출할 것."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        raise SourceError(
            f"카메라 {index}번을 열 수 없습니다. 시스템 설정 > 개인정보 보호 및 보안 > 카메라에서 "
            "백엔드를 실행한 터미널(또는 IDE)에 권한이 있는지 확인하고, 다른 카메라 번호도 시도해 보세요."
        )
    return cap


def _open_network_capture(url: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MSEC,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MSEC,
    ])


class NetworkCamera:
    """네트워크 카메라를 백그라운드에서 계속 읽어 최신 프레임만 넘겨준다 (cv2.VideoCapture처럼 read/release)."""

    def __init__(self, url: str):
        validate_network_url(url)
        self.display_url = mask_credentials(url)
        self._url = url
        cap = _open_network_capture(url)
        if not cap.isOpened():
            cap.release()
            raise SourceError(
                f"네트워크 카메라에 연결할 수 없습니다: {self.display_url} — 주소·계정, 같은 와이파이인지, "
                "VLC 같은 플레이어에서 열리는지 확인해 주세요."
            )
        self._cap = cap
        self._lock = threading.Condition()
        self._frame = None
        self._frame_id = 0
        self._returned_id = 0
        self._stopped = False
        self.reconnecting = False
        self.failed: Optional[str] = None   # 재연결까지 실패하면 이유를 남긴다
        self._thread = threading.Thread(target=self._reader, name="network-camera", daemon=True)
        self._thread.start()

    def read(self, timeout: float = 1.0):
        """아직 넘겨주지 않은 최신 프레임을 돌려준다. 없으면 (False, None)."""
        with self._lock:
            if self._frame_id == self._returned_id and not self._stopped:
                self._lock.wait(timeout)
            if self._frame_id == self._returned_id or self._frame is None:
                return False, None
            self._returned_id = self._frame_id
            return True, self._frame

    def release(self):
        with self._lock:
            self._stopped = True
            self._frame = None
            self._lock.notify_all()
        self._thread.join(timeout=OPEN_TIMEOUT_MSEC / 1000 + 2)

    def _reader(self):
        try:
            while not self._stopped:
                ok, frame = self._cap.read()
                if ok and frame is not None:
                    with self._lock:
                        self._frame = frame          # 이전 프레임은 덮어써서 버린다
                        self._frame_id += 1
                        self._lock.notify_all()
                    continue
                if not self._reconnect():
                    return
        finally:
            self._cap.release()

    def _reconnect(self) -> bool:
        self.reconnecting = True
        self._cap.release()
        for _ in range(RECONNECT_ATTEMPTS):
            if self._stopped:
                return False
            time.sleep(RECONNECT_DELAY_SEC)
            cap = _open_network_capture(self._url)
            if cap.isOpened():
                self._cap = cap
                self.reconnecting = False
                return True
            cap.release()
        self.failed = f"네트워크 카메라 연결이 끊겨 {RECONNECT_ATTEMPTS}번 다시 시도했지만 실패했습니다."
        self.reconnecting = False
        with self._lock:
            self._stopped = True
            self._lock.notify_all()
        return False
