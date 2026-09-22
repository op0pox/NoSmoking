"""
edge/motion_detector.py — 손-입 반복 모션(흡연 의심) 실시간 감지 프로토타입

1차 게이트 모듈. 담배 객체 감지(모델 2) 이전에, 포즈만으로
"손이 입 근처로 반복해서 올라갔다 내려오는" 동작을 잡아 흡연 의심
후보를 걸러낸다. 사람 감지는 YOLOv8n-pose 하나로 겸한다(별도 사람
모델 없음). 영상·프레임을 디스크에 저장하는 코드는 없다(프로젝트 원칙).

판정 로직은 SmokingMotionDetector 클래스로 분리되어 있어 나중에
다른 파이프라인(edge 앱 전체, 담배 객체 모델과의 통합 등)에서
`from motion_detector import SmokingMotionDetector` 로 재사용 가능하다.
사람 1명당 인스턴스를 1개씩 만들어 쓴다.

macOS 참고사항
--------------
- 최초 실행 시 "카메라 접근 권한" 팝업이 뜬다. 허용하지 않으면 검은
  화면만 나오거나 카메라를 열지 못한다. 나중에 막았다면
  시스템 설정 > 개인정보 보호 및 보안 > 카메라 에서 터미널(또는 사용
  중인 IDE)에 권한을 켜준다.
- cv2.VideoCapture(0)이 아이폰 연속성 카메라(Continuity Camera)로
  잡히는 경우가 있다. 이 경우 --camera 1 등으로 인덱스를 바꿔서
  실행한다. 실행 시 콘솔에 실제로 열린 카메라의 해상도를 출력하니
  확인할 것.
- cv2.imshow 창이 뒤로 숨어 포커스를 못 받는 경우가 있다. 창을 한 번
  클릭해서 포커스를 준 뒤 q/r 키를 눌러야 한다(waitKey는 표준 패턴 사용).
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from cigarette_detector import CIGARETTE_CLASS, SMOKE_CLASS, CigaretteDetector
from event_filter import EVENT_WINDOW_SEC, EventTimeFilter
from voice_announcer import STATE_IDLE as ANNOUNCER_IDLE
from voice_announcer import VoiceAnnouncer

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
DRAW_CONF_THRESHOLD = 0.2      # 화면 표시 전용(판정 기준 아님). 판정 임계값보다 낮게 잡아
                                # 손목이 얼굴 근처에서 신뢰도가 떨어져도 디버깅용으로 위치를 보여준다.
POSE_LOST_GRACE_SEC = 0.5      # 사람 전체가 잠깐 트래킹을 놓쳐도 진행 중인 puff를 바로 리셋하지 않는 유예시간
WRIST_OCCLUSION_GRACE_SEC = 3.0  # 코/어깨는 보이는데 양쪽 손목만 안 보이는 경우의 유예시간.
                                  # 손이 얼굴을 가리면 포즈 모델이 손목 신뢰도를 낮게 주는 경우가 많아
                                  # (바로 흡연 의심 동작 중일 때) POSE_LOST_GRACE_SEC보다 훨씬 길게 잡는다.
                                  # 최종 유지시간이 MAX_HOLD_SEC을 넘으면 어차피 무효 처리되므로
                                  # 길게 잡아도 오탐 위험은 없다.
NEAR_EXIT_GRACE_SEC = 0.3      # 손이 멀어진 것으로 판정된 프레임이 이 시간 이상 유지되어야 puff 종료로 인정
                                # (포즈 추정 노이즈로 거리값이 한두 프레임 튀는 것 방지)
STALE_TRACK_TIMEOUT_SEC = 5.0  # 이 시간 이상 화면에서 안 보이면 해당 track 상태를 폐기

# 2차 확인(담배/연기 객체 감지) 관련 — SMOKING_SUSPECTED인 사람에 한해서만 실행
MOTION_WEIGHT = 0.5            # 합산 점수에서 모션 점수(puff 진행도) 가중치
OBJECT_WEIGHT = 0.5            # 합산 점수에서 2차 객체 감지 신뢰도 가중치
                                # MOTION_WEIGHT + OBJECT_WEIGHT 합산 점수. v2 모델(cigarette-v2-best.pt)은
                                # mAP50 0.355(cigarette 단독 0.557), precision 0.392로 v1(mAP50 ~0.3, precision
                                # 0.2~0.9 진동) 대비 개선됐지만 여전히 낮은 편이라, 가중치는 5주차 오탐 튜닝
                                # 대상으로 남겨둔다.
COMBINED_CONFIRM_THRESHOLD = 0.5  # combined 점수가 이 값 이상인 프레임을 "이 프레임은 흡연"으로 판정
                                    # (event_filter.EventTimeFilter의 10초 윈도 입력값이 됨)

GRAPH_HISTORY_SEC = 60.0       # 하단 그래프에 표시할 과거 구간 길이
GRAPH_HEIGHT_PX = 140          # 하단 그래프 영역 높이(px)

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

SKELETON_EDGES = [
    (LEFT_ANKLE, LEFT_KNEE), (LEFT_KNEE, LEFT_HIP),
    (RIGHT_ANKLE, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_EYE, RIGHT_EYE), (NOSE, LEFT_EYE), (NOSE, RIGHT_EYE),
    (LEFT_EYE, LEFT_EAR), (RIGHT_EYE, RIGHT_EAR),
    (LEFT_SHOULDER, LEFT_EAR), (RIGHT_SHOULDER, RIGHT_EAR),
]

STATE_IDLE = "IDLE"
STATE_PUFF = "PUFF"
STATE_SUSPECTED = "SMOKING_SUSPECTED"

PALETTE = [
    (66, 135, 245), (66, 245, 173), (245, 66, 230), (245, 173, 66),
    (173, 66, 245), (80, 220, 80), (60, 60, 245), (66, 200, 245),
]

# 2차 확인(담배/연기) 검출 박스 색상 — BGR, 클래스별로 눈에 띄게 다른 색
OBJECT_BOX_COLORS = {
    CIGARETTE_CLASS: (0, 0, 255),    # 빨강
    SMOKE_CLASS: (255, 160, 0),      # 파랑 계열
}


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


# ============================================================
# 판정 로직 (다른 파이프라인에서 import해서 재사용)
# ============================================================
class SmokingMotionDetector:
    """사람 1명에 대한 "손-입 반복 모션" 상태 머신.

    프레임마다 update(now, keypoints)를 호출하면 내부적으로 puff(한 모금)
    판정과 SMOKING_SUSPECTED 판정을 갱신한다. 사람 추적(track id)별로
    인스턴스를 하나씩 만들어 쓴다.
    """

    def __init__(self, track_id: int):
        self.track_id = track_id
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
            print(f"[{_ts()}] track {self.track_id}: SMOKING_SUSPECTED 해제 (수동 리셋)")

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
            print(f"[{_ts()}] track {self.track_id}: puff #{self.puff_count} 인정 (유지 {hold:.2f}s)")

    def _purge_old_puffs(self, now: float):
        while self._puff_timestamps and now - self._puff_timestamps[0] > WINDOW_SEC:
            self._puff_timestamps.popleft()

    def _update_state(self, now: float):
        should_suspect = len(self._puff_timestamps) >= MIN_PUFFS

        if should_suspect and self.state != STATE_SUSPECTED:
            self.state = STATE_SUSPECTED
            print(
                f"[{_ts()}] track {self.track_id}: SMOKING_SUSPECTED 진입 "
                f"(최근 {WINDOW_SEC:.0f}s 내 puff {len(self._puff_timestamps)}회)"
            )
        elif not should_suspect and self.state == STATE_SUSPECTED:
            print(f"[{_ts()}] track {self.track_id}: SMOKING_SUSPECTED 해제")
            self.state = STATE_IDLE

        if self.state != STATE_SUSPECTED:
            self.state = STATE_PUFF if self._near_start_time is not None else STATE_IDLE


# ============================================================
# 디버깅용 화면 표시 (프레임/영상 저장 없음)
# ============================================================
def _color_for_track(track_id: int):
    return PALETTE[track_id % len(PALETTE)]


def draw_skeleton(frame: np.ndarray, kpts: np.ndarray, color):
    for a, b in SKELETON_EDGES:
        pa, pb = _point(kpts, a, DRAW_CONF_THRESHOLD), _point(kpts, b, DRAW_CONF_THRESHOLD)
        if pa and pb:
            cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 2)
    for idx in range(kpts.shape[0]):
        p = _point(kpts, idx, DRAW_CONF_THRESHOLD)
        if p:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, color, -1)


def draw_wrist_nose_lines(frame: np.ndarray, kpts: np.ndarray):
    mouth = _mouth_point(kpts, DRAW_CONF_THRESHOLD)
    if mouth is None:
        return
    mx, my = int(mouth[0]), int(mouth[1])
    cv2.circle(frame, (mx, my), 4, (0, 165, 255), -1)  # 판정 기준점(입 추정 위치) 표시
    for wrist_idx in (LEFT_WRIST, RIGHT_WRIST):
        wr = _point(kpts, wrist_idx, DRAW_CONF_THRESHOLD)
        if wr:
            cv2.line(frame, (int(wr[0]), int(wr[1])), (mx, my), (0, 255, 255), 1)


def draw_status_badge(
    frame: np.ndarray,
    kpts: np.ndarray,
    detector: SmokingMotionDetector,
    object_info: Optional[dict] = None,
    event_filter: Optional[EventTimeFilter] = None,
    announcer: Optional[VoiceAnnouncer] = None,
):
    nose = _point(kpts, NOSE)
    anchor = nose if nose is not None else (float(kpts[0][0]), float(kpts[0][1]))
    x, y = int(anchor[0]), max(20, int(anchor[1]) - 40)

    if event_filter is not None and event_filter.confirmed:
        badge_color, text_color = (0, 0, 180), (0, 255, 255)     # 이벤트 확정 — 더 진한 빨강+노란 글씨로 강조
    elif detector.state == STATE_SUSPECTED:
        badge_color, text_color = (0, 0, 255), (255, 255, 255)   # 빨간 배경
    elif detector.state == STATE_PUFF:
        badge_color, text_color = (0, 200, 255), (0, 0, 0)
    else:
        badge_color, text_color = (60, 60, 60), (255, 255, 255)

    dist_label = f"d={detector.last_dist:.2f}" if detector.last_dist is not None else "d=--"
    text = f"#{detector.track_id} {detector.state} {dist_label} puff={detector.puff_count}"
    if object_info is not None:
        text += (
            f" 2nd(cig={object_info['cigarette']:.2f} "
            f"smoke={object_info['smoke']:.2f} combined={object_info['combined']:.2f})"
        )
    if event_filter is not None:
        marker = "🔔EVENT" if event_filter.confirmed else ""
        text += f" [{event_filter.last_ratio:.0%}/{EVENT_WINDOW_SEC:.0f}s]{marker}"
    if announcer is not None and announcer.state != ANNOUNCER_IDLE:
        text += f" 방송:{announcer.state}"

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + 4), badge_color, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)


def draw_object_boxes(frame: np.ndarray, boxes: list):
    """2차 확인(담배/연기)에서 이번 프레임에 실제로 검출된 것만 박스로 그린다.
    잔상 유지 없음 — boxes가 비어 있으면 아무것도 그리지 않는다."""
    for det in boxes:
        x1, y1, x2, y2 = det["box"]
        color = OBJECT_BOX_COLORS.get(det["class"], (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        label = f"{det['class']} {det['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(th + 6, y1)
        cv2.rectangle(frame, (x1, label_y - th - 6), (x1 + tw + 6, label_y), color, -1)
        cv2.putText(frame, label, (x1 + 3, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def draw_graph_panel(width: int, height: int, detectors: dict, now: float) -> np.ndarray:
    """최근 GRAPH_HISTORY_SEC초 동안의 정규화 거리 파형을 그린다."""
    panel = np.full((height, width, 3), 30, dtype=np.uint8)

    margin = 30
    plot_w = width - margin * 2
    plot_h = height - 30
    y_max = 1.5  # 정규화 거리 표시 상한(그래프용)

    def y_for(v: float) -> int:
        v = min(max(v, 0.0), y_max)
        return int(height - 20 - (v / y_max) * plot_h)

    th_y = y_for(NEAR_THRESHOLD)
    cv2.line(panel, (margin, th_y), (width - margin, th_y), (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"NEAR_THRESHOLD={NEAR_THRESHOLD}", (margin, th_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, "normalized wrist-nose distance (last 60s)", (margin, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    for track_id, det in detectors.items():
        if len(det.history) < 2:
            continue
        color = _color_for_track(track_id)
        pts = []
        for t, d in det.history:
            age = now - t
            if age > GRAPH_HISTORY_SEC:
                continue
            x = int(margin + (1 - age / GRAPH_HISTORY_SEC) * plot_w)
            pts.append((x, y_for(d)))
        for p1, p2 in zip(pts, pts[1:]):
            cv2.line(panel, p1, p2, color, 1, cv2.LINE_AA)

    return panel


def select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="손-입 반복 모션(흡연 의심) 실시간 감지 프로토타입")
    parser.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture 카메라 인덱스 (기본 0)")
    parser.add_argument(
        "--source", type=str, default="webcam",
        help='입력 소스. "webcam"(기본, --camera 인덱스로 라이브 캡처) 또는 영상 파일 경로(예: test.mp4).',
    )
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt", help="YOLOv8-pose 가중치 경로/이름")
    parser.add_argument(
        "--cigarette-model", type=str, default="cigarette-v2-best.pt",
        help="담배/연기 2차 확인 모델 가중치 경로. SMOKING_SUSPECTED인 사람에 대해서만 사용된다.",
    )
    parser.add_argument(
        "--first-mp3", type=str, default="assets/audio/first_announce.mp3",
        help="1차(완곡) 안내 mp3 경로",
    )
    parser.add_argument(
        "--second-mp3", type=str, default="assets/audio/second_announce.mp3",
        help="2차(법적 고지 포함) 안내 mp3 경로",
    )
    parser.add_argument(
        "--pose-imgsz", type=int, default=640,
        help="포즈 모델 추론 해상도. 웹캠 근접 촬영은 기본값(640)으로 충분하지만, "
             "설치 예정 거리처럼 사람이 화면에 작게 나오는 영상은 1280~1920으로 올려야 "
             "손목 등 작은 keypoint 정밀도가 확보된다(속도는 느려짐).",
    )
    args = parser.parse_args()

    device = select_device()
    print(f"[정보] 추론 디바이스: {device}")

    model = YOLO(args.model)
    cigarette_detector = CigaretteDetector(args.cigarette_model, device=device)

    for label, path in (("1차", args.first_mp3), ("2차", args.second_mp3)):
        if not os.path.isfile(path):
            print(f"[경고] {label} 안내 mp3가 없습니다: {path} — 해당 안내는 재생되지 않고 로그만 남습니다.")

    is_webcam = args.source == "webcam"

    if is_webcam:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(
                f"카메라 인덱스 {args.camera}를 열 수 없습니다. macOS에서는 최초 실행 시 카메라 권한 "
                f"팝업이 뜹니다 — 허용했는지, 시스템 설정 > 개인정보 보호 및 보안 > 카메라에서 "
                f"터미널(또는 IDE) 권한이 켜져 있는지 확인하세요."
            )
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[정보] 카메라 {args.camera} 사용 중 — 해상도 {width}x{height}")
        print(
            "[정보] macOS에서는 아이폰(연속성 카메라)이 인덱스 0으로 잡힐 수 있습니다. "
            "다른 카메라를 쓰려면 --camera 1 처럼 인덱스를 바꿔서 실행하세요."
        )
    else:
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            raise RuntimeError(f"영상 파일을 열 수 없습니다: {args.source}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[정보] 영상 파일 사용 중: {args.source} ({width}x{height}, {video_fps:.1f}fps)")

    detectors: dict = {}
    object_scores: dict = {}  # track_id -> {"cigarette", "smoke", "combined"} (SUSPECTED일 때만 갱신)
    event_filters: dict = {}  # track_id -> EventTimeFilter (SUSPECTED일 때만 갱신)
    announcers: dict = {}     # track_id -> VoiceAnnouncer (매 프레임 갱신 — 쿨다운 타임아웃은 시간이 지나면 진행돼야 함)

    window_name = "NoSmoking motion_detector  (q: 종료 / r: 리셋)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.monotonic()
    fps = 0.0
    consecutive_read_failures = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if not is_webcam:
                    # 영상 파일은 읽기 실패 = 끝까지 재생한 것. 웹캠처럼 재시도하지 않고 종료한다.
                    print(f"[정보] 영상 재생 종료: {args.source}")
                    break
                consecutive_read_failures += 1
                if consecutive_read_failures == 1:
                    print("[경고] 프레임을 읽지 못했습니다.")
                elif consecutive_read_failures == 30:
                    print(
                        "[경고] 프레임을 30프레임 연속 못 읽고 있습니다. macOS 카메라 권한이 "
                        "이 터미널/IDE에 허용되어 있는지 확인하세요 "
                        "(시스템 설정 > 개인정보 보호 및 보안 > 카메라)."
                    )
                # 읽기 실패 중에도 waitKey를 계속 호출해야 창이 멈추지 않고 q로 종료할 수 있다
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue
            consecutive_read_failures = 0
            if is_webcam:
                frame = cv2.flip(frame, 1)  # 셀카처럼 좌우 반전(거울 모드) — 영상 파일에는 적용하지 않음

            # 영상 파일은 처리 속도가 실시간보다 빠르거나 느릴 수 있으므로, 시간 임계값
            # (MIN_HOLD_SEC, WINDOW_SEC 등)이 영상에 찍힌 실제 경과 시간 기준으로 맞도록
            # 영상 자체의 재생 위치(ms)를 시간 기준으로 쓴다. 웹캠은 그대로 실시간(wall clock).
            now = time.monotonic() if is_webcam else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            try:
                # tracker를 명시하지 않으면 ultralytics 버전에 따라 new_track_thresh가
                # 매우 높은 트래커가 기본값이 되어(예: 0.7) 웹캠의 일반적인 포즈 신뢰도로는
                # track id가 영영 배정되지 않을 수 있다. bytetrack은 임계값이 낮고(0.25)
                # 가벼워서 명시적으로 고정한다.
                results = model.track(
                    frame, persist=True, verbose=False, device=device, tracker="bytetrack.yaml",
                    imgsz=args.pose_imgsz,
                )
            except Exception as e:  # mps에서 특정 연산이 미지원일 때 cpu로 폴백
                if device != "cpu":
                    print(f"[경고] {device} 추론 실패({e}) — cpu로 폴백합니다.")
                    device = "cpu"
                    cigarette_detector.device = device
                    results = model.track(
                        frame, persist=True, verbose=False, device=device, tracker="bytetrack.yaml",
                        imgsz=args.pose_imgsz,
                    )
                else:
                    raise

            result = results[0]
            seen_ids = set()

            if result.keypoints is not None and result.boxes is not None and result.boxes.id is not None:
                kpts_all = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
                ids = result.boxes.id.cpu().numpy().astype(int)
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4) — kpts_all/ids와 같은 순서

                for kpts, track_id, box in zip(kpts_all, ids, boxes_xyxy):
                    track_id = int(track_id)
                    seen_ids.add(track_id)

                    detector = detectors.get(track_id)
                    if detector is None:
                        detector = SmokingMotionDetector(track_id)
                        detectors[track_id] = detector

                    detector.update(now, kpts)

                    # 2차 확인: SMOKING_SUSPECTED인 사람에 한해서만 돌린다 (상시 실행 금지)
                    if detector.state == STATE_SUSPECTED:
                        x1, y1, x2, y2 = box
                        crop_x1, crop_y1 = max(0, int(x1)), max(0, int(y1))
                        crop = frame[crop_y1:int(y2), crop_x1:int(x2)]
                        obj_result = cigarette_detector.detect(crop)
                        cig_conf = obj_result[CIGARETTE_CLASS]["conf"]
                        smoke_conf = obj_result[SMOKE_CLASS]["conf"]
                        motion_score = min(detector.puff_count / MIN_PUFFS, 1.0)
                        object_score = max(cig_conf, smoke_conf)
                        combined = MOTION_WEIGHT * motion_score + OBJECT_WEIGHT * object_score

                        # 크롭 좌표 -> 원본 프레임 좌표로 변환해서 검출된 프레임에만 박스로 표시
                        # (잔상 유지 없음 — 이번 프레임에 실제로 검출된 것만 그린다)
                        boxes_this_frame = []
                        for cls_name, det_info in obj_result.items():
                            if det_info["box"] is not None:
                                bx1, by1, bx2, by2 = det_info["box"]
                                boxes_this_frame.append({
                                    "class": cls_name,
                                    "conf": det_info["conf"],
                                    "box": (bx1 + crop_x1, by1 + crop_y1, bx2 + crop_x1, by2 + crop_y1),
                                })
                        draw_object_boxes(frame, boxes_this_frame)

                        object_scores[track_id] = {
                            "cigarette": cig_conf, "smoke": smoke_conf, "combined": combined,
                        }
                        print(
                            f"[{_ts()}] track {track_id}: 2차 확인 "
                            f"cig={cig_conf:.2f} smoke={smoke_conf:.2f} "
                            f"combined={combined:.2f}"
                        )

                        # 시간 필터: 순간적인 오탐 프레임 하나로 이벤트가 확정되지 않도록
                        # 최근 10초 윈도 내 판정 비율(60%)을 본다
                        ef = event_filters.get(track_id)
                        if ef is None:
                            ef = EventTimeFilter(track_id)
                            event_filters[track_id] = ef
                        was_confirmed = ef.confirmed
                        is_smoking_frame = combined >= COMBINED_CONFIRM_THRESHOLD
                        ef.update(now, is_smoking_frame)
                        if ef.confirmed and not was_confirmed:
                            print(
                                f"[{_ts()}] track {track_id}: 🔔 이벤트 확정 "
                                f"(최근 {EVENT_WINDOW_SEC:.0f}s 판정 비율 {ef.last_ratio:.0%})"
                            )

                    color = _color_for_track(track_id)
                    draw_skeleton(frame, kpts, color)
                    draw_wrist_nose_lines(frame, kpts)
                    draw_status_badge(
                        frame, kpts, detector, object_scores.get(track_id),
                        event_filters.get(track_id), announcers.get(track_id),
                    )

            # 이번 프레임에 안 보인 track에도 update(None)을 줘서 유예시간 로직이 돌게 한다
            for track_id, detector in detectors.items():
                if track_id not in seen_ids:
                    detector.update(now, None)

            # 음성 안내 상태 머신은 화면에 안 보이는 사람에 대해서도 매 프레임 진행시켜야
            # 30초 대기·쿨다운 타이머가 실제 시간에 맞춰 흘러간다
            for track_id in detectors:
                announcer = announcers.get(track_id)
                if announcer is None:
                    announcer = VoiceAnnouncer(track_id, args.first_mp3, args.second_mp3)
                    announcers[track_id] = announcer
                ef = event_filters.get(track_id)
                announcer.update(now, ef.confirmed if ef is not None else False)

            # 오래 안 보인 track은 상태를 폐기(메모리 누수 방지)
            stale_ids = [tid for tid, d in detectors.items() if now - d.last_seen_time > STALE_TRACK_TIMEOUT_SEC]
            for tid in stale_ids:
                del detectors[tid]
                object_scores.pop(tid, None)
                event_filters.pop(tid, None)
                announcers.pop(tid, None)

            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = fps * 0.9 + (1.0 / dt) * 0.1

            total_puffs = sum(d.puff_count for d in detectors.values())
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}  people: {len(detectors)}  puffs(total): {total_puffs}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
            )

            graph_panel = draw_graph_panel(frame.shape[1], GRAPH_HEIGHT_PX, detectors, now)
            canvas = np.vstack([frame, graph_panel])

            cv2.imshow(window_name, canvas)

            # macOS에서 imshow 창이 포커스를 못 받는 경우가 있어 표준 waitKey 패턴 사용
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                print(f"[{_ts()}] 전체 상태 리셋")
                for d in detectors.values():
                    d.reset()
                detectors.clear()
                object_scores.clear()
                event_filters.clear()
                announcers.clear()

        if not is_webcam:
            print(f"\n[정보] 결과 요약 — {args.source}")
            if not detectors:
                print("  감지된 사람 없음")
            for tid, d in detectors.items():
                ef = event_filters.get(tid)
                event_label = "EVENT 확정" if ef is not None and ef.confirmed else "미확정"
                print(f"  track {tid}: 최종 상태={d.state} puff={d.puff_count} 이벤트={event_label}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
