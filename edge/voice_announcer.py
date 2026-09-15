"""
edge/voice_announcer.py — 사전 녹음 mp3 안내 음성 상태 머신

이벤트가 확정되면(event_filter.EventTimeFilter.confirmed) 스피커로
사전 녹음된 mp3를 재생한다. 실시간 TTS 없음, 마이크 입력 코드 없음
(오디오 입력은 법적으로 금지 — 여기서는 재생만 한다).

흐름: 이벤트 확정 → 1차 완곡 안내 재생 → 30초 뒤에도 여전히(또는 다시)
흡연으로 확정 중이면 2차(법적 고지 포함) 안내 재생, 아니면 그냥
쿨다운 → 쿨다운 동안은 재방송 안 함(중복 방지) → 쿨다운 끝나면 IDLE로
복귀해 다시 감지를 기다린다.

재생은 OS 기본 플레이어를 subprocess로 실행해 논블로킹으로 돌린다.
외부 파이썬 오디오 라이브러리를 추가하지 않는다 — 파이 포팅 시에도
가볍게 유지하기 위함(playsound/pygame 등은 의존성이 무겁거나 플랫폼별
버그가 많다).

VoiceAnnouncer 클래스로 분리되어 있어 사람(track_id)별로 인스턴스를
하나씩 만들어 쓴다.
"""

from __future__ import annotations

import platform
import subprocess
import time
from pathlib import Path
from typing import Optional

SECOND_ANNOUNCE_WINDOW_SEC = 30.0   # 1차 안내 후 이 시간이 지난 시점에 여전히 확정 중이면 2차 안내
COOLDOWN_SEC = 60.0                  # 안내(1차 또는 2차) 후 재방송을 막는 쿨다운 길이

STATE_IDLE = "IDLE"
STATE_FIRST_ANNOUNCED = "FIRST_ANNOUNCED"
STATE_COOLDOWN = "COOLDOWN"


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _play_async(path: str) -> Optional[subprocess.Popen]:
    """OS 기본 플레이어로 mp3를 논블로킹 재생한다.

    재생 실패(파일 없음, 플레이어 없음 등)해도 예외를 삼키고 경고만
    출력한다 — 안내 음성 하나 때문에 감지 파이프라인 전체가 멈추면 안 된다.
    """
    if not Path(path).is_file():
        print(f"[경고] 안내 음성 파일을 찾을 수 없습니다: {path} (재생 건너뜀)")
        return None

    system = platform.system()
    try:
        if system == "Darwin":
            return subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif system == "Linux":
            # 라즈베리파이 등 — mpg123을 가장 가벼운 mp3 전용 플레이어로 우선 시도
            return subprocess.Popen(
                ["mpg123", "-q", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:
            print(f"[경고] {system} 플랫폼용 오디오 재생 명령이 정의되지 않았습니다: {path}")
            return None
    except FileNotFoundError:
        print(f"[경고] 오디오 재생 명령을 찾을 수 없습니다({system}). mp3: {path}")
        return None


class VoiceAnnouncer:
    """사람 1명에 대한 음성 안내 상태 머신.

    매 프레임 update(now, event_confirmed)를 호출한다.
    event_confirmed는 EventTimeFilter.confirmed 값을 그대로 넣으면 된다.
    """

    def __init__(self, track_id: int, first_mp3: str, second_mp3: str):
        self.track_id = track_id
        self.first_mp3 = first_mp3
        self.second_mp3 = second_mp3

        self.state = STATE_IDLE
        self._first_announced_at: Optional[float] = None
        self._cooldown_until: Optional[float] = None

    def reset(self):
        self.state = STATE_IDLE
        self._first_announced_at = None
        self._cooldown_until = None

    def update(self, now: float, event_confirmed: bool):
        """상태를 한 단계 진행한다. 상태당 한 번만 처리하고 바로 반환한다."""
        if self.state == STATE_COOLDOWN:
            if self._cooldown_until is not None and now >= self._cooldown_until:
                self.state = STATE_IDLE
                self._first_announced_at = None
                self._cooldown_until = None
            return  # 쿨다운 중에는 새 이벤트를 무시한다 (중복 방송 방지)

        if self.state == STATE_IDLE:
            if event_confirmed:
                self._announce(self.first_mp3, "1차")
                self.state = STATE_FIRST_ANNOUNCED
                self._first_announced_at = now
            return

        if self.state == STATE_FIRST_ANNOUNCED:
            elapsed = now - self._first_announced_at
            if elapsed < SECOND_ANNOUNCE_WINDOW_SEC:
                return  # 아직 30초가 안 지났으면 대기만 한다

            # 30초가 지난 시점 — 그때도 여전히 흡연으로 확정 중이면 2차, 아니면 조용히 쿨다운
            if event_confirmed:
                self._announce(self.second_mp3, "2차")
            self._enter_cooldown(now)

    def _announce(self, mp3_path: str, label: str):
        print(f"[{_ts()}] track {self.track_id}: {label} 안내 재생 — {mp3_path}")
        _play_async(mp3_path)

    def _enter_cooldown(self, now: float):
        self.state = STATE_COOLDOWN
        self._cooldown_until = now + COOLDOWN_SEC
