"""
edge/video_analysis.py — 영상 파일 하나를 파이프라인으로 끝까지 분석하는 공용 함수

평가 도구(eval/run.py)와 웹 데모의 영상 분석 페이지(demo/backend)가 같은 로직을 쓴다.
프레임은 추론에만 쓰고 저장하지 않는다. 결과에는 시각·점수 같은 숫자만 남는다.

record_timeline=True면 사람별·프레임별 판정 값(타임라인)을 함께 기록한다 —
"언제, 어느 단계에서 막혔는지"를 그래프로 보기 위함이다. 평가 도구처럼 요약만
필요하면 False로 두어 메모리를 아낀다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2

import event_filter
import pipeline as pipeline_module
from pipeline import EVENT_ANNOUNCED, EVENT_CONFIRMED, SmokingPipeline


@dataclass
class TrackSample:
    """사람 1명의 한 프레임 판정 값."""
    t: float                          # 영상 속 시각(초)
    state: str                        # 1차 게이트 상태 (IDLE / PUFF / SMOKING_SUSPECTED)
    puff_count: int
    dist: Optional[float] = None      # 손-입 정규화 거리 (이번 프레임에 계산됐을 때만)
    checked: bool = False             # 이번 프레임에 2차 확인을 돌렸나
    # 아래는 2차 확인을 돌린 프레임에만 채워진다
    cig: Optional[float] = None
    smoke: Optional[float] = None
    combined: Optional[float] = None
    mouth_cig: Optional[float] = None
    smoking_frame: Optional[bool] = None   # 이 프레임을 "흡연"으로 판정했나 (시간 필터 입력)
    ratio: Optional[float] = None           # 시간 필터의 최근 10초 판정 비율
    confirmed: Optional[bool] = None


@dataclass
class TrackSummary:
    track_id: int
    first_t: float
    last_t: float
    max_puffs: int
    checks: int                  # 2차 확인 횟수
    smoking_frames: int          # 그중 "흡연"으로 판정된 횟수
    max_cig: float
    max_smoke: float
    max_mouth_cig: float
    max_ratio: float
    confirmed: bool
    reason: str                  # 어느 단계까지 갔는지 / 어디서 막혔는지 (사람이 읽는 설명)


@dataclass
class AnalysisResult:
    duration_sec: float = 0.0
    fps: float = 0.0
    frames: int = 0
    wall_sec: float = 0.0
    events: list = field(default_factory=list)        # PipelineEvent 목록
    max_ratio: float = 0.0                            # 영상 전체 시간 필터 비율 최댓값
    timeline: dict = field(default_factory=dict)      # track_id -> [TrackSample] (record_timeline일 때만)

    def confirmed_times(self) -> list:
        return [e.time for e in self.events if e.kind == EVENT_CONFIRMED]

    def first_announce_times(self) -> list:
        return [e.time for e in self.events if e.kind == EVENT_ANNOUNCED and e.detail == "1차"]


def analyze_video(
    video_path: str,
    pipe: SmokingPipeline,
    record_timeline: bool = False,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> AnalysisResult:
    """영상을 처음부터 끝까지 pipe로 처리한다. on_progress(처리한 프레임 수, 전체 프레임 수)."""
    result = AnalysisResult()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"영상 파일을 열 수 없습니다: {video_path}")
    result.fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    result.duration_sec = total_frames / result.fps if result.fps else 0.0

    started = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # motion_detector.py와 같은 방식으로 영상 재생 위치를 시간 기준으로 쓴다
            now = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_result = pipe.process(frame, now)
            result.frames += 1
            result.events.extend(frame_result.events)
            for ef in pipe.event_filters.values():
                result.max_ratio = max(result.max_ratio, ef.last_ratio)
            if record_timeline:
                _record_frame(result.timeline, pipe, frame_result.persons, now)
            if on_progress is not None:
                on_progress(result.frames, total_frames)
    finally:
        cap.release()
    result.wall_sec = time.monotonic() - started
    return result


def _record_frame(timeline: dict, pipe: SmokingPipeline, persons: list, now: float):
    for person in persons:
        detector = person.detector
        # 손-입 거리는 이번 프레임에 실제로 계산됐을 때만 기록한다 (history 마지막 항목의 시각으로 판단)
        dist = None
        if detector.history and detector.history[-1][0] == now:
            dist = detector.history[-1][1]
        sample = TrackSample(t=now, state=detector.state, puff_count=detector.puff_count, dist=dist)

        if pipe.last_object_check.get(person.track_id) == now:
            scores = pipe.object_scores[person.track_id]
            ef = pipe.event_filters[person.track_id]
            sample.checked = True
            sample.cig = scores["cigarette"]
            sample.smoke = scores["smoke"]
            sample.combined = scores["combined"]
            sample.mouth_cig = scores["mouth_cig"]
            sample.smoking_frame = is_smoking_frame(scores)
            sample.ratio = ef.last_ratio
            sample.confirmed = ef.confirmed
        timeline.setdefault(person.track_id, []).append(sample)


def is_smoking_frame(scores: dict) -> bool:
    """pipeline._check_object와 같은 기준으로 그 프레임이 "흡연" 판정이었는지 계산한다."""
    return (
        scores["combined"] >= pipeline_module.COMBINED_CONFIRM_THRESHOLD
        or scores["mouth_cig"] >= pipeline_module.MOUTH_CIG_CONF_THRESHOLD
    )


def summarize_track(track_id: int, samples: list, events: list) -> TrackSummary:
    """사람 1명의 타임라인을 요약하고, 확정까지 어느 단계에서 막혔는지 설명을 붙인다."""
    checked = [s for s in samples if s.checked]
    confirmed = any(e.kind == EVENT_CONFIRMED and e.track_id == track_id for e in events)
    summary = TrackSummary(
        track_id=track_id,
        first_t=samples[0].t,
        last_t=samples[-1].t,
        max_puffs=max(s.puff_count for s in samples),
        checks=len(checked),
        smoking_frames=sum(1 for s in checked if s.smoking_frame),
        max_cig=max((s.cig for s in checked), default=0.0),
        max_smoke=max((s.smoke for s in checked), default=0.0),
        max_mouth_cig=max((s.mouth_cig for s in checked), default=0.0),
        max_ratio=max((s.ratio for s in checked), default=0.0),
        confirmed=confirmed,
        reason="",
    )
    summary.reason = _explain(summary, measured_dist=any(s.dist is not None for s in samples))
    return summary


def _explain(s: TrackSummary, measured_dist: bool) -> str:
    """확정까지 가는 단계 중 어디서 멈췄는지 한 줄로 설명한다 (화면 표시용)."""
    required = event_filter.EVENT_CONFIRM_RATIO
    if s.confirmed:
        return "이벤트 확정"
    if s.checks == 0:
        return "2차 확인 전에 화면에서 사라짐"
    if s.smoking_frames == 0:
        motion = f"puff {s.max_puffs}회" if measured_dist else "얼굴·어깨를 못 잡아 손-입 거리 측정 불가"
        return (
            f"흡연 증거 없음 — {motion}, 담배 최고 {s.max_cig:.2f} "
            f"(입 근처 {s.max_mouth_cig:.2f}), 연기 최고 {s.max_smoke:.2f}"
        )
    if s.last_t - s.first_t < event_filter.EVENT_WINDOW_SEC:
        return (
            f"관측 시간 부족 — {s.last_t - s.first_t:.0f}초만 보임 "
            f"(확정에는 최소 {event_filter.EVENT_WINDOW_SEC:.0f}초 필요)"
        )
    return (
        f"시간 필터 미달 — 흡연 판정 {s.smoking_frames}/{s.checks}회, "
        f"10초 최고 비율 {s.max_ratio:.0%} (기준 {required:.0%})"
    )
