"""
edge/pipeline.py — 감지 파이프라인 전체를 화면 없이 프레임 단위로 돌리는 모듈

    사람+포즈 감지 → 1차 게이트(손-입 모션) → 2차 확인(담배/연기 객체)
    → 시간 필터 → 음성 안내

SmokingPipeline.process(frame, now)에 프레임을 하나씩 넣으면 사람별 상태와
이번 프레임에 생긴 이벤트(이벤트 확정, 안내 방송)를 돌려준다. 화면 그리기는
하지 않는다 — 실시간 실행(edge/motion_detector.py), 평가 도구(eval/),
웹 데모가 모두 이 모듈을 그대로 재사용한다.

영상·프레임을 저장하거나 전송하는 코드는 없다(프로젝트 원칙). 프레임은
process() 안에서 추론에만 쓰고 보관하지 않는다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from ultralytics import YOLO

from cigarette_detector import CIGARETTE_CLASS, SMOKE_CLASS, CigaretteDetector
from event_filter import EVENT_WINDOW_SEC, EventTimeFilter
from motion_gate import MIN_PUFFS, STATE_SUSPECTED, SmokingMotionDetector, _ts
from voice_announcer import VoiceAnnouncer

# ============================================================
# 튜닝 상수 — 2차 확인(담배/연기 객체 감지)과 점수 합산.
# 1차 게이트 상수는 motion_gate.py, 시간 필터 상수는 event_filter.py에 있다.
# ============================================================
STALE_TRACK_TIMEOUT_SEC = 5.0  # 이 시간 이상 화면에서 안 보이면 해당 track 상태를 폐기

OBJECT_CHECK_INTERVAL_SEC = 1.0  # SUSPECTED가 아닌 사람도 이 주기로 2차 확인을 돌린다(SUSPECTED면
                                  # 매 프레임). 모션 3회를 다 채우지 못해도 담배/연기가 여러 프레임에
                                  # 걸쳐 보이면 잡을 기회를 주되, 상시 실행은 피해서 연산을 아낀다.
COMBINED_CONFIRM_THRESHOLD = 0.5  # combined 점수가 이 값 이상인 프레임을 "이 프레임은 흡연"으로 판정
                                    # (event_filter.EventTimeFilter의 10초 윈도 입력값이 됨)
OBJECT_SIGNAL_HOLD_SEC = 1.5   # cig/smoke 신뢰도가 이번 프레임에 0이어도, 이 시간 이내에 마지막으로
                                # 감지된 값이 있으면 그 값을 페어링에 대신 쓴다. 객체 감지가 프레임마다
                                # 깜빡이는 걸(있다가 없다가) 완화하지 않으면, 연관 점수(_pair_score)가
                                # 매번 정확히 같은 프레임에 두 신호가 겹쳐야만 해서 모션이 최고조여도
                                # 자꾸 0으로 끊겨버린다(실측으로 확인함).
HIGH_CONFIDENCE_THRESHOLD = 0.75  # cig_conf 또는 smoke_conf가 이 값 이상이면 짝(모션/다른 클래스)
                                    # 없이도 combined 후보가 된다 — 단, 아래 두 조건을 같이 만족해야 한다.
                                    # 실측 결과 v2 모델은 배경 물체를 0.88로도 오인하므로 단일 프레임
                                    # 신뢰도만으로는 안전하지 않다.
HIGH_CONF_MIN_PUFFS = 2            # 고신뢰도 바이패스를 쓰려면 최소 이 횟수 이상 puff가 쌓여 있어야 한다
                                    # (모션이 전혀 없는 상태에서 객체 오검출 하나로 확정되는 걸 막는다).
HIGH_CONF_SUSTAIN_WINDOW_SEC = 10.0  # 고신뢰도 검출이 지속되는지 보는 시간 창
HIGH_CONF_SUSTAIN_COUNT = 2        # 이 창 안에 고신뢰도 검출이 이 횟수 이상 있어야 바이패스 인정
                                    # (순간 오검출 한 번으로는 통과 못 함, 진짜 흡연은 몇 초간 계속 보인다).
CIG_SMOKE_COOCCURRENCE_WINDOW_SEC = 5.0  # 모션이 안 잡혀도, 담배와 연기가 "정확히 같은 프레임"이
                                          # 아니라 이 시간 이내에만 각각 감지됐으면 서로 짝으로
                                          # 인정한다. 연기는 담배 연기가 아니어도(김·안개·수증기)
                                          # 오검출될 수 있어 연기 단독보다는 신뢰도가 높지만, 담배를
                                          # 입에 무는 순간과 연기가 보이는 순간은 실제로도 약간의
                                          # 시차가 있을 수 있어 OBJECT_SIGNAL_HOLD_SEC보다 길게 잡는다.

# PipelineEvent.kind 값
EVENT_CONFIRMED = "confirmed"   # 시간 필터가 미확정 → 확정으로 바뀐 순간
EVENT_ANNOUNCED = "announced"   # 안내 방송이 나간 순간 (detail: "1차" / "2차")


def _pair_score(a: float, b: float) -> float:
    """두 신호(0~1)의 기하평균. 하나라도 0이면 0 —
    단일 신호(담배만, 모션만 등)만으로는 점수가 오르지 않고, 두 증거가
    같이 있어야 확정에 가까워지도록 한다."""
    if a <= 0 or b <= 0:
        return 0.0
    return (a * b) ** 0.5


def _held_conf(
    hold_store: dict, track_id: int, cls_name: str, raw_conf: float, now: float, hold_sec: float
) -> float:
    """raw_conf가 0보다 크면 그 값을 최신 기준으로 기록하고 그대로 반환한다.
    raw_conf가 0이면 hold_sec 이내에 마지막으로 감지된 값이 있으면 그 값을
    대신 반환한다(_pair_score 계산용 — 프레임마다 깜빡이는 객체 감지 노이즈
    완화). 그 시간도 지났으면 0.0.

    hold_sec은 호출하는 쪽이 상황에 맞게 넘긴다 — 예를 들어 모션과 짝지을
    때는 짧게(OBJECT_SIGNAL_HOLD_SEC), 담배·연기끼리 짝지을 때는 "동시가
    아니어도 됨"을 반영해 더 길게(CIG_SMOKE_COOCCURRENCE_WINDOW_SEC) 쓴다.
    """
    person_hold = hold_store.setdefault(track_id, {})
    if raw_conf > 0:
        person_hold[cls_name] = (raw_conf, now)
        return raw_conf
    entry = person_hold.get(cls_name)
    if entry is not None:
        held_conf, held_time = entry
        if now - held_time <= hold_sec:
            return held_conf
    return 0.0


def _record_high_conf_hit(hits_store: dict, track_id: int, now: float, is_high: bool) -> int:
    """고신뢰도 검출 시각을 기록하고, HIGH_CONF_SUSTAIN_WINDOW_SEC 창 안의 검출 횟수를 반환한다.
    is_high가 False면 기록 없이 창만 정리해서 횟수를 돌려준다."""
    hits = hits_store.setdefault(track_id, deque())
    if is_high:
        hits.append(now)
    while hits and now - hits[0] > HIGH_CONF_SUSTAIN_WINDOW_SEC:
        hits.popleft()
    return len(hits)


@dataclass
class PersonResult:
    """이번 프레임에 보인 사람 1명의 판정 결과 (화면 그리기·평가에 쓴다)."""
    track_id: int
    keypoints: np.ndarray            # (17, 3) [x, y, conf]
    box: np.ndarray                  # 사람 bbox (x1, y1, x2, y2), 원본 프레임 좌표
    detector: SmokingMotionDetector
    # 이번 프레임에 2차 확인을 돌렸을 때만 채워진다. 각 항목: {"class", "conf", "box"(원본 프레임 좌표)}
    object_boxes: list = field(default_factory=list)


@dataclass
class PipelineEvent:
    time: float      # process()에 넘긴 now 기준 시각(초)
    track_id: int
    kind: str        # EVENT_CONFIRMED / EVENT_ANNOUNCED
    detail: str = ""


@dataclass
class FrameResult:
    persons: list    # PersonResult 목록
    events: list     # 이번 프레임에 생긴 PipelineEvent 목록


class SmokingPipeline:
    """감지 파이프라인 전체. 입력 소스(웹캠/영상 파일)와 화면 출력은 호출하는 쪽이 담당한다.

    사용 예:
        pipeline = SmokingPipeline("yolov8n-pose.pt", "cigarette-v5-best.pt", device="mps")
        result = pipeline.process(frame, now)   # 프레임마다 호출

    now는 초 단위 시각이다. 웹캠은 time.monotonic(), 영상 파일은 재생 위치를 넘겨야
    시간 임계값(MIN_HOLD_SEC 등)이 영상 속 실제 경과 시간과 맞는다.
    """

    def __init__(
        self,
        pose_model_path: str,
        cigarette_model_path: str,
        device: str = "cpu",
        pose_imgsz: int = 640,
        first_mp3: str = "",
        second_mp3: str = "",
        play_audio: bool = True,
        verbose: bool = True,
    ):
        self.pose_model = YOLO(pose_model_path)
        self.cigarette_detector = CigaretteDetector(cigarette_model_path, device=device)
        self.device = device
        self.pose_imgsz = pose_imgsz
        self.first_mp3 = first_mp3
        self.second_mp3 = second_mp3
        self.play_audio = play_audio
        self.verbose = verbose  # False면 2차 확인·이벤트 로그를 출력하지 않는다(평가 일괄 실행용)
        self._init_state()

    def _init_state(self):
        # 모두 track_id를 키로 하는 사람별 상태
        self.detectors: dict = {}
        self.object_scores: dict = {}       # {"cigarette", "smoke", "combined"} (2차 확인 돈 프레임에만 갱신)
        self.event_filters: dict = {}       # EventTimeFilter (2차 확인 돈 프레임에만 갱신)
        self.announcers: dict = {}          # VoiceAnnouncer (매 프레임 갱신 — 쿨다운 타이머는 시간이 지나면 진행돼야 함)
        self.last_object_check: dict = {}   # 마지막으로 2차 확인을 돌린 시각 (주기 실행용)
        self.object_signal_hold: dict = {}  # {클래스명: (conf, 감지시각)} (_held_conf용)
        self.high_conf_hits: dict = {}      # 고신뢰도 검출 시각 deque (지속성 판정용)

    def _log(self, message: str):
        if self.verbose:
            print(f"[{_ts()}] {message}")

    def reset(self):
        """모든 사람의 상태를 지운다 (화면의 r 키)."""
        for detector in self.detectors.values():
            detector.reset()
        self._init_state()

    def process(self, frame: np.ndarray, now: float) -> FrameResult:
        """프레임 하나를 처리하고 사람별 결과와 이번 프레임의 이벤트를 반환한다.

        frame은 읽기만 한다(그리지 않는다) — 화면 표시는 호출하는 쪽에서 결과를 보고 그린다.
        """
        events = []
        persons = []
        result = self._track_people(frame)

        if result.keypoints is not None and result.boxes is not None and result.boxes.id is not None:
            kpts_all = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
            ids = result.boxes.id.cpu().numpy().astype(int)
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4) — kpts_all/ids와 같은 순서

            for kpts, track_id, box in zip(kpts_all, ids, boxes_xyxy):
                track_id = int(track_id)
                detector = self.detectors.get(track_id)
                if detector is None:
                    detector = SmokingMotionDetector(track_id, verbose=self.verbose)
                    self.detectors[track_id] = detector

                detector.update(now, kpts)

                person = PersonResult(track_id=track_id, keypoints=kpts, box=box, detector=detector)
                if self._should_check_object(track_id, detector, now):
                    confirmed_now = self._check_object(frame, person, now)
                    if confirmed_now:
                        ef = self.event_filters[track_id]
                        events.append(PipelineEvent(now, track_id, EVENT_CONFIRMED, f"{ef.last_ratio:.0%}"))
                persons.append(person)

        # 이번 프레임에 안 보인 track에도 update(None)을 줘서 유예시간 로직이 돌게 한다
        seen_ids = {p.track_id for p in persons}
        for track_id, detector in self.detectors.items():
            if track_id not in seen_ids:
                detector.update(now, None)

        # 음성 안내 상태 머신은 화면에 안 보이는 사람에 대해서도 매 프레임 진행시켜야
        # 30초 대기·쿨다운 타이머가 실제 시간에 맞춰 흘러간다
        for track_id in self.detectors:
            announcer = self.announcers.get(track_id)
            if announcer is None:
                announcer = VoiceAnnouncer(track_id, self.first_mp3, self.second_mp3, play_audio=self.play_audio)
                self.announcers[track_id] = announcer
            ef = self.event_filters.get(track_id)
            announced = announcer.update(now, ef.confirmed if ef is not None else False)
            if announced is not None:
                events.append(PipelineEvent(now, track_id, EVENT_ANNOUNCED, announced))

        self._drop_stale_tracks(now)
        return FrameResult(persons=persons, events=events)

    def _track_people(self, frame: np.ndarray):
        """포즈 모델로 사람 박스·keypoints·track id를 얻는다."""
        try:
            # tracker를 명시하지 않으면 ultralytics 버전에 따라 new_track_thresh가
            # 매우 높은 트래커가 기본값이 되어(예: 0.7) 웹캠의 일반적인 포즈 신뢰도로는
            # track id가 영영 배정되지 않을 수 있다. bytetrack은 임계값이 낮고(0.25)
            # 가벼워서 명시적으로 고정한다.
            results = self.pose_model.track(
                frame, persist=True, verbose=False, device=self.device, tracker="bytetrack.yaml",
                imgsz=self.pose_imgsz,
            )
        except Exception as e:  # mps에서 특정 연산이 미지원일 때 cpu로 폴백
            if self.device == "cpu":
                raise
            print(f"[경고] {self.device} 추론 실패({e}) — cpu로 폴백합니다.")
            self.device = "cpu"
            self.cigarette_detector.device = "cpu"
            results = self.pose_model.track(
                frame, persist=True, verbose=False, device=self.device, tracker="bytetrack.yaml",
                imgsz=self.pose_imgsz,
            )
        return results[0]

    def _should_check_object(self, track_id: int, detector: SmokingMotionDetector, now: float) -> bool:
        """2차 확인: 상시 실행은 아니지만 SUSPECTED만 기다리지도 않는다.
        SUSPECTED면 매 프레임, 아니어도 사람이 있는 한 OBJECT_CHECK_INTERVAL_SEC
        주기로 체크한다 — 모션 3회를 다 못 채워도 담배/연기가 여러 프레임에
        걸쳐 보이면 잡을 기회를 준다."""
        last_check = self.last_object_check.get(track_id)
        return (
            detector.state == STATE_SUSPECTED
            or last_check is None
            or now - last_check >= OBJECT_CHECK_INTERVAL_SEC
        )

    def _check_object(self, frame: np.ndarray, person: PersonResult, now: float) -> bool:
        """사람 bbox를 크롭해 담배/연기 모델을 돌리고, 점수 합산과 시간 필터를 갱신한다.
        이번 호출로 이벤트가 새로 확정됐으면 True를 반환한다."""
        track_id = person.track_id
        detector = person.detector
        self.last_object_check[track_id] = now

        x1, y1, x2, y2 = person.box
        crop_x1, crop_y1 = max(0, int(x1)), max(0, int(y1))
        crop = frame[crop_y1:int(y2), crop_x1:int(x2)]
        obj_result = self.cigarette_detector.detect(crop)
        cig_conf = obj_result[CIGARETTE_CLASS]["conf"]
        smoke_conf = obj_result[SMOKE_CLASS]["conf"]
        motion_score = min(detector.puff_count / MIN_PUFFS, 1.0)

        # 연관 기반 점수: 두 증거가 같이 있어야 점수가 오른다.
        # 단일 신호(담배만/연기만/모션만)로는 combined가 0에 가깝다.
        # 화면 표시·로그에는 raw 값(cig_conf/smoke_conf)을 그대로 쓴다.
        #
        # 모션과 짝지을 때는 짧은 유예(OBJECT_SIGNAL_HOLD_SEC)만 써서 프레임 단위
        # 깜빡임만 완화한다. 담배·연기끼리 짝지을 때는 더 긴 유예
        # (CIG_SMOKE_COOCCURRENCE_WINDOW_SEC)를 써서 "정확히 같은 프레임"이
        # 아니어도 인정한다 — 연기는 담배 연기가 아니어도(김·안개·수증기)
        # 오검출될 수 있어 연기 단독보다 신뢰도가 높은 담배와 짝지어서 검증한다.
        hold = self.object_signal_hold
        held_cig_short = _held_conf(hold, track_id, CIGARETTE_CLASS, cig_conf, now, OBJECT_SIGNAL_HOLD_SEC)
        held_smoke_short = _held_conf(hold, track_id, SMOKE_CLASS, smoke_conf, now, OBJECT_SIGNAL_HOLD_SEC)
        held_cig_long = _held_conf(
            hold, track_id, CIGARETTE_CLASS, cig_conf, now, CIG_SMOKE_COOCCURRENCE_WINDOW_SEC
        )
        held_smoke_long = _held_conf(
            hold, track_id, SMOKE_CLASS, smoke_conf, now, CIG_SMOKE_COOCCURRENCE_WINDOW_SEC
        )

        # 아주 확실한 단일 검출(raw 값 기준, held 아님)은 짝 없이도 인정한다 —
        # 단, 지속성(창 안 여러 번)과 최소 모션(puff 누적)을 같이 만족해야 한다.
        # 순간 오검출 한 번이나 모션 없는 상태의 오검출은 이 경로로 확정되지 않는다.
        raw_high = max(cig_conf, smoke_conf) >= HIGH_CONFIDENCE_THRESHOLD
        high_hits = _record_high_conf_hit(self.high_conf_hits, track_id, now, raw_high)
        high_conf_bypass = 0.0
        if (
            raw_high
            and detector.puff_count >= HIGH_CONF_MIN_PUFFS
            and high_hits >= HIGH_CONF_SUSTAIN_COUNT
        ):
            high_conf_bypass = max(
                cig_conf if cig_conf >= HIGH_CONFIDENCE_THRESHOLD else 0.0,
                smoke_conf if smoke_conf >= HIGH_CONFIDENCE_THRESHOLD else 0.0,
            )
        combined = max(
            _pair_score(held_cig_long, held_smoke_long),
            _pair_score(held_cig_short, motion_score),
            _pair_score(held_smoke_short, motion_score),
            high_conf_bypass,
        )

        # 크롭 좌표 -> 원본 프레임 좌표로 변환 (화면에는 이번 프레임에 실제로 검출된 것만 그린다)
        for cls_name, det_info in obj_result.items():
            if det_info["box"] is not None:
                bx1, by1, bx2, by2 = det_info["box"]
                person.object_boxes.append({
                    "class": cls_name,
                    "conf": det_info["conf"],
                    "box": (bx1 + crop_x1, by1 + crop_y1, bx2 + crop_x1, by2 + crop_y1),
                })

        self.object_scores[track_id] = {"cigarette": cig_conf, "smoke": smoke_conf, "combined": combined}
        self._log(
            f"track {track_id}: 2차 확인 cig={cig_conf:.2f} smoke={smoke_conf:.2f} combined={combined:.2f}"
        )

        # 시간 필터: 순간적인 오탐 프레임 하나로 이벤트가 확정되지 않도록
        # 최근 10초 윈도 내 판정 비율(60%)을 본다
        ef = self.event_filters.get(track_id)
        if ef is None:
            ef = EventTimeFilter(track_id)
            self.event_filters[track_id] = ef
        was_confirmed = ef.confirmed
        ef.update(now, combined >= COMBINED_CONFIRM_THRESHOLD)
        confirmed_now = ef.confirmed and not was_confirmed
        if confirmed_now:
            self._log(
                f"track {track_id}: 🔔 이벤트 확정 (최근 {EVENT_WINDOW_SEC:.0f}s 판정 비율 {ef.last_ratio:.0%})"
            )
        return confirmed_now

    def _drop_stale_tracks(self, now: float):
        """오래 안 보인 track은 상태를 폐기한다(메모리 누수 방지)."""
        stale_ids = [tid for tid, d in self.detectors.items() if now - d.last_seen_time > STALE_TRACK_TIMEOUT_SEC]
        for tid in stale_ids:
            del self.detectors[tid]
            self.object_scores.pop(tid, None)
            self.event_filters.pop(tid, None)
            self.announcers.pop(tid, None)
            self.last_object_check.pop(tid, None)
            self.object_signal_hold.pop(tid, None)
            self.high_conf_hits.pop(tid, None)
