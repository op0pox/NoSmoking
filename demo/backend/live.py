"""
demo/backend/live.py — 웹 데모 실시간(웹캠) 세션

백엔드가 직접 웹캠을 읽고 감지 파이프라인을 돌린다. 판정 결과를 그린 화면은
브라우저에 MJPEG로 보여주기만 한다.

개인정보 원칙:
- 프레임은 저장하지 않는다. 메모리에는 화면 표시용 "가장 최근 프레임 한 장"(JPEG)만 두고
  다음 프레임이 오면 덮어쓴다. 세션을 끝내면 그것도 지운다.
- 화면을 보는 곳(스트림·상태 요청)이 IDLE_TIMEOUT_SEC 동안 없으면 카메라를 스스로 끈다
  (브라우저를 그냥 닫아도 카메라가 켜진 채 남지 않게).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import cv2

from overlay import render_overlay
from pipeline import SmokingPipeline

IDLE_TIMEOUT_SEC = 10.0      # 화면을 보는 곳이 이 시간 이상 없으면 자동 정지
JPEG_QUALITY = 80            # 화면 표시용 JPEG 품질
MAX_READ_FAILURES = 30       # 프레임을 연속으로 이만큼 못 읽으면 정지
MAX_KEPT_EVENTS = 50         # 화면에 보여줄 최근 이벤트 수


class LiveSession:
    """웹캠 1대에 대한 실시간 감지 세션. 한 번에 하나만 만든다.

    start()는 카메라를 연 캡처 객체를 받는다 — macOS는 카메라 권한 요청을 메인 스레드에서
    해야 해서, 카메라 열기는 호출하는 쪽(요청 처리 이벤트 루프)에서 한다.
    """

    def __init__(self, cap: cv2.VideoCapture, pipe: SmokingPipeline, mirror: bool = True):
        self._cap = cap
        self._pipe = pipe
        self._mirror = mirror   # 웹캠은 셀카처럼 좌우 반전 (motion_detector.py와 같음)

        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._latest_jpeg: Optional[bytes] = None
        self._frame_id = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="live-session", daemon=True)

        self.started_at = time.time()
        self.fps = 0.0
        self.error: Optional[str] = None
        self.stop_reason: Optional[str] = None
        self._last_viewer_seen = time.monotonic()
        self._people: list = []
        self._events: deque = deque(maxlen=MAX_KEPT_EVENTS)

    # ------------------------------------------------------------------ 제어
    def start(self):
        self._thread.start()

    def stop(self, reason: str = "사용자가 정지"):
        if self.stop_reason is None:
            self.stop_reason = reason
        self._stop.set()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=5)

    @property
    def running(self) -> bool:
        return self._thread.is_alive() and not self._stop.is_set()

    def touch(self):
        """화면을 보고 있다는 신호 (스트림·상태 요청마다 호출)."""
        self._last_viewer_seen = time.monotonic()

    # ------------------------------------------------------------------ 화면·상태
    def wait_for_frame(self, last_id: int, timeout: float = 1.0) -> tuple:
        """last_id 이후의 새 프레임을 기다려 (frame_id, jpeg)를 반환한다. 없으면 (last_id, None)."""
        with self._new_frame:
            if self._frame_id == last_id and self.running:
                self._new_frame.wait(timeout)
            if self._frame_id == last_id:
                return last_id, None
            return self._frame_id, self._latest_jpeg

    def state(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "fps": round(self.fps, 1),
                "error": self.error,
                "stop_reason": self.stop_reason,
                "started_at": self.started_at,
                "people": list(self._people),
                "events": list(self._events),
            }

    # ------------------------------------------------------------------ 작업 스레드
    def _run(self):
        failures = 0
        prev = time.monotonic()
        try:
            while not self._stop.is_set():
                if time.monotonic() - self._last_viewer_seen > IDLE_TIMEOUT_SEC:
                    self.stop_reason = f"{IDLE_TIMEOUT_SEC:.0f}초 동안 화면을 보는 곳이 없어 자동 정지"
                    break

                ok, frame = self._cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= MAX_READ_FAILURES:
                        self.error = "카메라에서 프레임을 읽지 못했습니다 (권한·연결 확인)"
                        break
                    time.sleep(0.05)
                    continue
                failures = 0
                if self._mirror:
                    frame = cv2.flip(frame, 1)

                now = time.monotonic()
                result = self._pipe.process(frame, now)
                dt = now - prev
                prev = now
                if dt > 0:
                    self.fps = self.fps * 0.9 + (1.0 / dt) * 0.1

                people = [self._person_state(p) for p in result.persons]
                canvas = render_overlay(frame, result, self._pipe, now, self.fps)
                ok, jpeg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

                with self._new_frame:
                    if ok:
                        self._latest_jpeg = jpeg.tobytes()   # 이전 프레임은 덮어써서 버린다
                        self._frame_id += 1
                    self._people = people
                    for e in result.events:
                        self._events.appendleft({
                            "clock": time.strftime("%H:%M:%S"),
                            "track_id": e.track_id,
                            "kind": e.kind,
                            "detail": e.detail,
                        })
                    self._new_frame.notify_all()
        except Exception as e:  # 화면에 원인을 보여준다
            self.error = f"실시간 처리 중 오류: {e}"
        finally:
            self._stop.set()
            self._cap.release()
            with self._new_frame:
                self._latest_jpeg = None   # 세션이 끝나면 마지막 프레임도 지운다
                self._new_frame.notify_all()

    def _person_state(self, person) -> dict:
        tid = person.track_id
        scores = self._pipe.object_scores.get(tid)
        ef = self._pipe.event_filters.get(tid)
        announcer = self._pipe.announcers.get(tid)
        detector = person.detector
        return {
            "track_id": tid,
            "state": detector.state,
            "puff_count": detector.puff_count,
            "dist": round(detector.last_dist, 2) if detector.last_dist is not None else None,
            "cig": round(scores["cigarette"], 2) if scores else None,
            "smoke": round(scores["smoke"], 2) if scores else None,
            "mouth_cig": round(scores["mouth_cig"], 2) if scores else None,
            "ratio": round(ef.last_ratio, 2) if ef else None,
            "confirmed": bool(ef and ef.confirmed),
            "announcer_state": announcer.state if announcer else None,
        }

