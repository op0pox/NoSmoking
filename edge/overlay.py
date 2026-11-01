"""
edge/overlay.py — 판정 결과를 프레임 위에 그리는 화면 표시 전용 모듈

터미널 실행(edge/motion_detector.py)과 웹 데모 실시간 페이지(demo/backend)가 같은
화면을 그리도록 공용으로 둔다. 판정에는 관여하지 않는다.
그린 프레임은 화면에 보여주기만 하고 저장하지 않는다(프로젝트 원칙).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from cigarette_detector import CIGARETTE_CLASS, SMOKE_CLASS
from event_filter import EVENT_WINDOW_SEC, EventTimeFilter
from motion_gate import (
    GRAPH_HISTORY_SEC,
    LEFT_ANKLE, LEFT_EAR, LEFT_ELBOW, LEFT_EYE, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST,
    NEAR_THRESHOLD, NOSE,
    RIGHT_ANKLE, RIGHT_EAR, RIGHT_ELBOW, RIGHT_EYE, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST,
    STATE_PUFF, STATE_SUSPECTED,
    SmokingMotionDetector, _mouth_point, _point,
)
from pipeline import FrameResult, SmokingPipeline
from voice_announcer import STATE_IDLE as ANNOUNCER_IDLE
from voice_announcer import VoiceAnnouncer

# ============================================================
# 화면 표시 전용 상수 (판정에는 쓰이지 않음)
# ============================================================
DRAW_CONF_THRESHOLD = 0.2      # 화면 표시 전용(판정 기준 아님). 판정 임계값보다 낮게 잡아
                                # 손목이 얼굴 근처에서 신뢰도가 떨어져도 디버깅용으로 위치를 보여준다.
GRAPH_HEIGHT_PX = 140          # 하단 그래프 영역 높이(px)

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

PALETTE = [
    (66, 135, 245), (66, 245, 173), (245, 66, 230), (245, 173, 66),
    (173, 66, 245), (80, 220, 80), (60, 60, 245), (66, 200, 245),
]

# 2차 확인(담배/연기) 검출 박스 색상 — BGR, 클래스별로 눈에 띄게 다른 색
OBJECT_BOX_COLORS = {
    CIGARETTE_CLASS: (0, 0, 255),    # 빨강
    SMOKE_CLASS: (255, 160, 0),      # 파랑 계열
}


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
            f"smoke={object_info['smoke']:.2f} combined={object_info['combined']:.2f} "
            f"mouth={object_info['mouth_cig']:.2f})"
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


def render_overlay(
    frame: np.ndarray, result: FrameResult, pipeline: SmokingPipeline, now: float, fps: float
) -> np.ndarray:
    """pipeline.process()가 끝난 프레임에 판정 결과를 그리고, 아래에 거리 그래프를 붙인 화면을 반환한다.

    판정을 모두 끝낸 뒤에 그려야 한다 — 그리면서 판정하면 앞사람의 박스·스켈레톤이
    뒷사람의 2차 확인 크롭에 섞여 들어간다. frame은 직접 그려서 바뀐다.
    """
    for person in result.persons:
        track_id = person.track_id
        draw_object_boxes(frame, person.object_boxes)
        draw_skeleton(frame, person.keypoints, _color_for_track(track_id))
        draw_wrist_nose_lines(frame, person.keypoints)
        draw_status_badge(
            frame, person.keypoints, person.detector, pipeline.object_scores.get(track_id),
            pipeline.event_filters.get(track_id), pipeline.announcers.get(track_id),
        )

    total_puffs = sum(d.puff_count for d in pipeline.detectors.values())
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}  people: {len(pipeline.detectors)}  puffs(total): {total_puffs}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
    )
    graph_panel = draw_graph_panel(frame.shape[1], GRAPH_HEIGHT_PX, pipeline.detectors, now)
    return np.vstack([frame, graph_panel])
