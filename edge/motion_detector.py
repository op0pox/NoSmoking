"""
edge/motion_detector.py — 흡연 감지 실시간 실행 + 화면 표시

웹캠 또는 영상 파일을 입력으로 감지 파이프라인(edge/pipeline.py의
SmokingPipeline)을 돌리고, 판정 상태를 OpenCV 창에 그려서 보여준다.
영상·프레임을 디스크에 저장하는 코드는 없다(프로젝트 원칙).

모듈 구성:
- motion_gate.py: 1차 게이트(손-입 반복 모션, SmokingMotionDetector)
- pipeline.py: 사람+포즈 → 1차 게이트 → 2차 확인 → 시간 필터 → 음성 안내
- 이 파일: 입력 소스 선택, 화면 그리기, 키 조작

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
from typing import Optional

import cv2
import numpy as np
import torch

from cigarette_detector import CIGARETTE_CLASS, SMOKE_CLASS
from event_filter import EVENT_WINDOW_SEC, EventTimeFilter
from motion_gate import (
    GRAPH_HISTORY_SEC,
    LEFT_ANKLE, LEFT_EAR, LEFT_ELBOW, LEFT_EYE, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST,
    NEAR_THRESHOLD, NOSE,
    RIGHT_ANKLE, RIGHT_EAR, RIGHT_ELBOW, RIGHT_EYE, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST,
    STATE_PUFF, STATE_SUSPECTED,
    SmokingMotionDetector, _mouth_point, _point, _ts,
)
from pipeline import SmokingPipeline
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
        "--cigarette-model", type=str, default="cigarette-v5-best.pt",
        help="담배/연기 2차 확인 모델 가중치 경로. SMOKING_SUSPECTED인 사람은 매 프레임, "
             "그 외는 1초 주기로 사용된다.",
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

    pipeline = SmokingPipeline(
        args.model, args.cigarette_model, device=device, pose_imgsz=args.pose_imgsz,
        first_mp3=args.first_mp3, second_mp3=args.second_mp3,
    )

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

            # 판정을 모든 사람에 대해 먼저 끝낸 뒤에 그린다 — 그리면서 판정하면
            # 앞사람의 박스·스켈레톤이 뒷사람의 2차 확인 크롭에 섞여 들어간다.
            result = pipeline.process(frame, now)

            for person in result.persons:
                track_id = person.track_id
                draw_object_boxes(frame, person.object_boxes)
                color = _color_for_track(track_id)
                draw_skeleton(frame, person.keypoints, color)
                draw_wrist_nose_lines(frame, person.keypoints)
                draw_status_badge(
                    frame, person.keypoints, person.detector, pipeline.object_scores.get(track_id),
                    pipeline.event_filters.get(track_id), pipeline.announcers.get(track_id),
                )

            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = fps * 0.9 + (1.0 / dt) * 0.1

            total_puffs = sum(d.puff_count for d in pipeline.detectors.values())
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}  people: {len(pipeline.detectors)}  puffs(total): {total_puffs}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
            )

            graph_panel = draw_graph_panel(frame.shape[1], GRAPH_HEIGHT_PX, pipeline.detectors, now)
            canvas = np.vstack([frame, graph_panel])

            cv2.imshow(window_name, canvas)

            # macOS에서 imshow 창이 포커스를 못 받는 경우가 있어 표준 waitKey 패턴 사용
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                print(f"[{_ts()}] 전체 상태 리셋")
                pipeline.reset()

        if not is_webcam:
            print(f"\n[정보] 결과 요약 — {args.source}")
            if not pipeline.detectors:
                print("  감지된 사람 없음")
            for tid, d in pipeline.detectors.items():
                ef = pipeline.event_filters.get(tid)
                event_label = "EVENT 확정" if ef is not None and ef.confirmed else "미확정"
                print(f"  track {tid}: 최종 상태={d.state} puff={d.puff_count} 이벤트={event_label}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
