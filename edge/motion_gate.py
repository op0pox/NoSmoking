"""
edge/motion_gate.py — 1차 게이트: 손-입 반복 모션(흡연 의심) 판정

담배 객체 감지(2차 확인) 이전에, 포즈 keypoint만으로 "손이 입 근처로
반복해서 올라갔다 내려오는" 동작을 잡아 흡연 의심 후보를 걸러낸다.
학습이 필요 없는 규칙 기반이다.

사람 1명당 SmokingMotionDetector 인스턴스를 1개씩 만들어 쓴다.
파이프라인 전체(2차 확인·시간 필터·음성 안내)와 묶는 것은
edge/pipeline.py의 SmokingPipeline이 담당한다.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import numpy as np

# ============================================================
# 튜닝 상수 — 5주차·9주차 집중 튜닝 대상. 코드 다른 곳에 매직넘버로
# 흩어놓지 말고 항상 여기에 모을 것.
# ============================================================
NEAR_THRESHOLD = 0.6           # 손목-입(추정) 정규화 거리가 이 값 미만이면 "입 근처".
                                # 주먹 쥔 자세는 0.35로도 충분하지만, 손가락 사이에 담배를
                                # 끼우는 자세는 손목-입 사이에 손 길이만큼 실거리가 더 있어
                                # 실측 정규화 거리가 0.44~0.6대로 나온다(2026-09-14 실측).
                                # 1차 게이트(이 모듈)는 recall 우선으로 넉넉히 잡고,
                                # 정밀도는 2차 담배 객체 감지 모델과 시간 필터에서 확보한다.
MOUTH_OFFSET_RATIO = 0.08      # 입 keypoint가 없어 코 좌표를 기준점으로 쓰되, 실제 입은 코보다
                                # 아래에 있으므로 어깨너비 대비 이 비율만큼 아래로 내려서 보정한다.
                                # 값을 올리면 기준점이 더 아래(입에 가깝게)로 내려간다.
MIN_HOLD_SEC = 1.0             # 이 시간 이상 유지되어야 puff(한 모금)로 인정
MAX_HOLD_SEC = 6.0             # 이 시간을 넘기면 무효 처리(통화 등 장시간 접촉)
WINDOW_SEC = 60.0              # puff를 세는 슬라이딩 윈도 길이
MIN_PUFFS = 3                  # 윈도 안에서 이 횟수 이상이면 흡연 의심 확정

KEYPOINT_CONF_THRESHOLD = 0.5  # 코·어깨 등 몸통 keypoint 판정에 쓰는 최소 신뢰도
WRIST_CONF_THRESHOLD = 0.15    # 손목 keypoint 전용 임계값. 실측 결과 웹캠에서는 손을 활발히
                                # 움직이는 중에도 손목 신뢰도가 0.5를 거의 못 넘는다(오른쪽 0.35~0.54,
                                # 왼쪽 0.15~0.25 수준). 0.5를 그대로 쓰면 왼손처럼 신뢰도가 더 낮게
                                # 나오는 쪽은 puff가 사실상 전혀 인정되지 않는다. 가만히 쉬는 손목은
                                # 0.02~0.05대라 이 값으로도 움직임과는 충분히 구분된다.
POSE_LOST_GRACE_SEC = 0.5      # 사람 전체가 잠깐 트래킹을 놓쳐도 진행 중인 puff를 바로 리셋하지 않는 유예시간
WRIST_OCCLUSION_GRACE_SEC = 3.0  # 코/어깨는 보이는데 양쪽 손목만 안 보이는 경우의 유예시간.
                                  # 손이 얼굴을 가리면 포즈 모델이 손목 신뢰도를 낮게 주는 경우가 많아
                                  # (바로 흡연 의심 동작 중일 때) POSE_LOST_GRACE_SEC보다 훨씬 길게 잡는다.
                                  # 최종 유지시간이 MAX_HOLD_SEC을 넘으면 어차피 무효 처리되므로
                                  # 길게 잡아도 오탐 위험은 없다.
NEAR_EXIT_GRACE_SEC = 0.3      # 손이 멀어진 것으로 판정된 프레임이 이 시간 이상 유지되어야 puff 종료로 인정
                                # (포즈 추정 노이즈로 거리값이 한두 프레임 튀는 것 방지)

GRAPH_HISTORY_SEC = 60.0       # 정규화 거리 이력을 보관하는 길이 (화면 하단 그래프·평가 타임라인용)

# COCO 17 keypoints 인덱스 (YOLOv8-pose 순서). 입 keypoint가 없어
# 코(nose) 좌표를 MOUTH_OFFSET_RATIO만큼 아래로 보정해 입 대용으로 쓴다.
NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

STATE_IDLE = "IDLE"
STATE_PUFF = "PUFF"
STATE_SUSPECTED = "SMOKING_SUSPECTED"


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _point(kpts: np.ndarray, idx: int, conf_th: float = KEYPOINT_CONF_THRESHOLD):
    """keypoint 배열에서 신뢰도를 통과한 (x, y) 좌표만 반환, 아니면 None."""
    x, y, c = kpts[idx]
    if c < conf_th:
        return None
    return float(x), float(y)


def _dist(p1, p2) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def _mouth_point(kpts: np.ndarray, conf_th: float = KEYPOINT_CONF_THRESHOLD):
    """코·어깨 keypoint로부터 입 추정 위치(코에서 아래로 보정)를 반환. 실패 시 None."""
    nose = _point(kpts, NOSE, conf_th)
    l_sh = _point(kpts, LEFT_SHOULDER, conf_th)
    r_sh = _point(kpts, RIGHT_SHOULDER, conf_th)
    if nose is None or l_sh is None or r_sh is None:
        return None
    shoulder_width = _dist(l_sh, r_sh)
    return nose[0], nose[1] + MOUTH_OFFSET_RATIO * shoulder_width


class SmokingMotionDetector:
    """사람 1명에 대한 "손-입 반복 모션" 상태 머신.

    프레임마다 update(now, keypoints)를 호출하면 내부적으로 puff(한 모금)
    판정과 SMOKING_SUSPECTED 판정을 갱신한다. 사람 추적(track id)별로
    인스턴스를 하나씩 만들어 쓴다.
    """

    def __init__(self, track_id: int, verbose: bool = True):
        self.track_id = track_id
        self.verbose = verbose  # False면 puff·상태 전환 로그를 출력하지 않는다(평가 일괄 실행용)
        self.state = STATE_IDLE

        self._near_start_time: Optional[float] = None
        self._far_start_time: Optional[float] = None
        self._last_valid_pose_time: Optional[float] = None
        self._puff_timestamps: deque = deque()

        self.last_dist: Optional[float] = None
        self.last_seen_time: float = time.monotonic()
        self.puff_count: int = 0

        # 하단 그래프용 (timestamp, normalized_dist) 이력
        self.history: deque = deque()

    def _log(self, message: str):
        if self.verbose:
            print(f"[{_ts()}] track {self.track_id}: {message}")

    def reset(self):
        was_suspected = self.state == STATE_SUSPECTED
        self._near_start_time = None
        self._far_start_time = None
        self._last_valid_pose_time = None
        self._puff_timestamps.clear()
        self.last_dist = None
        self.puff_count = 0
        self.history.clear()
        self.state = STATE_IDLE
        if was_suspected:
            self._log("SMOKING_SUSPECTED 해제 (수동 리셋)")

    def update(self, now: float, keypoints: Optional[np.ndarray]) -> str:
        """한 프레임 분량의 keypoints로 상태를 갱신하고 현재 상태 문자열을 반환한다.

        keypoints: (17, 3) array [x, y, conf], 탐지 실패 시 None.
        """
        if keypoints is not None:
            self.last_seen_time = now

        normalized, wrists_occluded = (
            self._compute_normalized_dist(keypoints) if keypoints is not None else (None, False)
        )

        if normalized is not None:
            self._last_valid_pose_time = now
            self.last_dist = normalized
            self.history.append((now, normalized))

            if normalized < NEAR_THRESHOLD:
                if self._near_start_time is None:
                    self._near_start_time = now
                self._far_start_time = None  # 다시 가까워졌으니 이탈 타이머 취소
            elif self._near_start_time is not None:
                # 거리값이 한두 프레임 튀는 노이즈일 수 있으니 곧바로 종료하지 않고
                # NEAR_EXIT_GRACE_SEC 동안 실제로 멀어진 상태가 유지되는지 확인한다
                if self._far_start_time is None:
                    self._far_start_time = now
                elif now - self._far_start_time >= NEAR_EXIT_GRACE_SEC:
                    # 유예 만료 시점(now)이 아니라 실제로 멀어지기 시작한 시점을 종료 시각으로 쓴다
                    self._resolve_near_period(self._far_start_time)
                    self._far_start_time = None
        else:
            # 포즈 실패/저신뢰도 프레임: 유예시간 안이면 진행 중이던 puff를 그대로 둔다.
            # 손목만 안 보이는 경우(손이 얼굴을 가려서)는 훨씬 긴 유예를 준다 —
            # 바로 흡연 의심 동작이 진행 중일 가능성이 가장 높은 순간이기 때문이다.
            grace = WRIST_OCCLUSION_GRACE_SEC if wrists_occluded else POSE_LOST_GRACE_SEC
            if (
                self._near_start_time is not None
                and self._last_valid_pose_time is not None
                and now - self._last_valid_pose_time > grace
            ):
                self._near_start_time = None
                self._far_start_time = None

        self._purge_old_puffs(now)
        self._update_state(now)

        while self.history and now - self.history[0][0] > GRAPH_HISTORY_SEC:
            self.history.popleft()

        return self.state

    def _compute_normalized_dist(self, kpts: np.ndarray) -> tuple:
        """정규화 거리를 계산한다. (normalized, wrists_occluded)를 반환한다.

        wrists_occluded는 코·어깨는 보이는데 양쪽 손목만 신뢰도가 낮은 경우 True다.
        손이 얼굴 근처(=우리가 가장 판정하고 싶은 순간)로 갈수록 이 상태가 되기 쉬우므로
        update()에서 별도의 긴 유예시간(WRIST_OCCLUSION_GRACE_SEC)을 적용하기 위해 구분한다.
        """
        l_sh = _point(kpts, LEFT_SHOULDER)
        r_sh = _point(kpts, RIGHT_SHOULDER)
        mouth = _mouth_point(kpts)
        if mouth is None or l_sh is None or r_sh is None:
            return None, False

        shoulder_width = _dist(l_sh, r_sh)
        if shoulder_width < 1e-3:
            return None, False

        candidates = []
        l_wr = _point(kpts, LEFT_WRIST, WRIST_CONF_THRESHOLD)
        r_wr = _point(kpts, RIGHT_WRIST, WRIST_CONF_THRESHOLD)
        if l_wr is not None:
            candidates.append(_dist(l_wr, mouth) / shoulder_width)
        if r_wr is not None:
            candidates.append(_dist(r_wr, mouth) / shoulder_width)

        if not candidates:
            return None, True
        return min(candidates), False

    def _resolve_near_period(self, now: float):
        """손이 입 근처에서 멀어졌을 때, 유지 시간을 보고 puff 인정 여부를 결정한다."""
        if self._near_start_time is None:
            return
        hold = now - self._near_start_time
        self._near_start_time = None
        if MIN_HOLD_SEC <= hold <= MAX_HOLD_SEC:
            self._puff_timestamps.append(now)
            self.puff_count += 1
            self._log(f"puff #{self.puff_count} 인정 (유지 {hold:.2f}s)")

    def _purge_old_puffs(self, now: float):
        while self._puff_timestamps and now - self._puff_timestamps[0] > WINDOW_SEC:
            self._puff_timestamps.popleft()

    def _update_state(self, now: float):
        should_suspect = len(self._puff_timestamps) >= MIN_PUFFS

        if should_suspect and self.state != STATE_SUSPECTED:
            self.state = STATE_SUSPECTED
            self._log(
                f"SMOKING_SUSPECTED 진입 (최근 {WINDOW_SEC:.0f}s 내 puff {len(self._puff_timestamps)}회)"
            )
        elif not should_suspect and self.state == STATE_SUSPECTED:
            self._log("SMOKING_SUSPECTED 해제")
            self.state = STATE_IDLE

        if self.state != STATE_SUSPECTED:
            self.state = STATE_PUFF if self._near_start_time is not None else STATE_IDLE
