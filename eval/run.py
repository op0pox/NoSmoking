"""
eval/run.py — 이벤트 단위 정량 평가

eval/labels.csv에 적힌 영상마다 감지 파이프라인(edge/pipeline.py)을 화면 없이
끝까지 돌리고, 정답과 비교해 정밀도·재현율·오탐·지연·처리 속도를 표로 낸다.
영상·프레임은 저장하지 않는다. 결과는 숫자만 출력한다(--out이면 마크다운 파일로도 저장).

사용 (저장소 루트에서):
    python -m eval.run                                                  # 기본: v5, 포즈 640
    python -m eval.run --weights cigarette-v4-best.pt cigarette-v5-best.pt   # 모델 비교
    python -m eval.run --pose-imgsz 640 1280                            # 포즈 해상도 비교
    python -m eval.run --set pipeline.OBJECT_CHECK_INTERVAL_SEC=0.2     # 상수 바꿔서 실행
    python -m eval.run --include-reference --out eval/results/v5.md     # 참고 영상 포함, 파일 저장

--weights와 --pose-imgsz에 여러 값을 주면 모든 조합을 돌려 한 표로 비교한다.

채점 기준
---------
- 장면 단위: 흡연 영상은 흡연 구간 안에서 이벤트가 하나라도 확정되면 정탐(TP), 아니면 미탐(FN).
  비흡연(함정) 영상은 이벤트가 하나라도 확정되면 오탐(FP), 아니면 정상(TN).
  방송은 구역 전체에 나가므로 계도 목적에서는 이 지표가 핵심이다.
- 이벤트 정밀도: 확정된 이벤트 중 흡연 구간 안에서 나온 비율.
- 흡연 구간: labels.csv의 smoke_start_sec~smoke_end_sec. 비어 있으면 영상 전체.
  시간 필터가 최대 EVENT_WINDOW_SEC만큼 늦게 확정되므로 끝 시각에는 그만큼 여유를 준다.
- 지연: 흡연 시작 시각부터 첫 1차 안내 방송까지(초). smoke_start_sec가 있을 때만 계산한다.
- split=reference인 영상(동작 조건 밖·모자이크 등)은 메인 수치에서 빼고 참고용으로만 보여준다.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import torch

# edge/ 모듈은 서로를 `from pipeline import ...`처럼 폴더 안 기준으로 import하므로,
# 저장소 루트에서 실행해도 찾을 수 있게 edge/를 import 경로에 추가한다.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "edge"))

import event_filter  # noqa: E402
import motion_gate  # noqa: E402
import pipeline  # noqa: E402
import voice_announcer  # noqa: E402
from pipeline import EVENT_ANNOUNCED, EVENT_CONFIRMED, SmokingPipeline  # noqa: E402

# --set으로 상수를 바꿀 수 있는 모듈
TUNABLE_MODULES = {
    "motion_gate": motion_gate,
    "pipeline": pipeline,
    "event_filter": event_filter,
    "voice_announcer": voice_announcer,
}

SMOKING = "smoking"
NON_SMOKING = "non_smoking"


@dataclass
class VideoLabel:
    video: str
    label: str                       # SMOKING / NON_SMOKING
    category: str                    # 함정 종류(drink, coffee, phone ...) 또는 smoking
    smoke_start_sec: Optional[float]
    smoke_end_sec: Optional[float]
    split: str                       # main / reference
    note: str


@dataclass
class VideoResult:
    label: VideoLabel
    config: str
    duration_sec: float = 0.0
    frames: int = 0
    wall_sec: float = 0.0
    confirmed_times: list = field(default_factory=list)   # 이벤트 확정 시각들
    announced_times: list = field(default_factory=list)   # 1차 안내 시각들
    max_ratio: float = 0.0           # 영상 전체에서 시간 필터 판정 비율의 최댓값 (아깝게 놓친 정도 확인용)

    def _interval(self):
        start = self.label.smoke_start_sec or 0.0
        end = self.label.smoke_end_sec if self.label.smoke_end_sec is not None else self.duration_sec
        return start, end + event_filter.EVENT_WINDOW_SEC

    def true_events(self) -> int:
        if self.label.label != SMOKING:
            return 0
        start, end = self._interval()
        return sum(1 for t in self.confirmed_times if start <= t <= end)

    def false_events(self) -> int:
        return len(self.confirmed_times) - self.true_events()

    def outcome(self) -> str:
        if self.label.label == SMOKING:
            return "TP" if self.true_events() > 0 else "FN"
        return "FP" if self.confirmed_times else "TN"

    def latency_sec(self) -> Optional[float]:
        if self.label.label != SMOKING or self.label.smoke_start_sec is None:
            return None
        start, end = self._interval()
        in_interval = [t for t in self.announced_times if start <= t <= end]
        return (in_interval[0] - start) if in_interval else None


def _optional_float(text: str) -> Optional[float]:
    text = text.strip()
    return float(text) if text else None


def load_labels(path: Path) -> list:
    labels = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] not in (SMOKING, NON_SMOKING):
                raise ValueError(f"{path}: label은 {SMOKING} 또는 {NON_SMOKING}이어야 합니다 — {row}")
            labels.append(VideoLabel(
                video=row["video"].strip(),
                label=row["label"].strip(),
                category=row["category"].strip(),
                smoke_start_sec=_optional_float(row["smoke_start_sec"]),
                smoke_end_sec=_optional_float(row["smoke_end_sec"]),
                split=row["split"].strip() or "main",
                note=row["note"].strip(),
            ))
    return labels


def apply_overrides(overrides: list) -> list:
    """--set 모듈.상수=값 을 적용하고, 적용한 내용을 "모듈.상수=값" 문자열 목록으로 돌려준다.

    pipeline.py처럼 다른 모듈의 상수를 `from motion_gate import MIN_PUFFS` 식으로 가져다 쓰는
    곳이 있어서, 같은 이름·같은 값을 들고 있는 다른 edge 모듈에도 함께 반영한다.
    단, 함수 기본 인자로 박힌 값(예: _point의 KEYPOINT_CONF_THRESHOLD)은 바뀌지 않는다.
    """
    applied = []
    for item in overrides:
        key, _, value_text = item.partition("=")
        module_name, _, const_name = key.partition(".")
        module = TUNABLE_MODULES.get(module_name)
        if module is None or not value_text or not hasattr(module, const_name):
            raise SystemExit(f"--set 형식 오류: {item} (예: pipeline.OBJECT_CHECK_INTERVAL_SEC=0.2)")
        old_value = getattr(module, const_name)
        new_value = type(old_value)(value_text)
        for other in TUNABLE_MODULES.values():
            if getattr(other, const_name, None) == old_value:
                setattr(other, const_name, new_value)
        applied.append(f"{module_name}.{const_name}={new_value}")
    return applied


def evaluate_video(
    label: VideoLabel, video_path: Path, weights: str, pose_model: str, pose_imgsz: int, device: str, config: str
) -> VideoResult:
    result = VideoResult(label=label, config=config)
    pipe = SmokingPipeline(
        pose_model, weights, device=device, pose_imgsz=pose_imgsz,
        play_audio=False, verbose=False,
    )
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    result.duration_sec = frame_count / fps if fps else 0.0

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

            for event in frame_result.events:
                if event.kind == EVENT_CONFIRMED:
                    result.confirmed_times.append(event.time)
                elif event.kind == EVENT_ANNOUNCED and event.detail == "1차":
                    result.announced_times.append(event.time)
            for ef in pipe.event_filters.values():
                result.max_ratio = max(result.max_ratio, ef.last_ratio)
    finally:
        cap.release()
    result.wall_sec = time.monotonic() - started
    return result


def _pct(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.0%} ({numerator}/{denominator})" if denominator else "—"


def _sec(value: Optional[float]) -> str:
    return f"{value:.1f}" if value is not None else "—"


def summary_rows(results: list) -> list:
    """설정별 요약 행. results는 한 split(main 또는 reference)의 결과만 넘긴다."""
    rows = []
    for config in dict.fromkeys(r.config for r in results):
        rs = [r for r in results if r.config == config]
        count = {k: sum(1 for r in rs if r.outcome() == k) for k in ("TP", "FN", "FP", "TN")}
        true_ev = sum(r.true_events() for r in rs)
        false_ev = sum(r.false_events() for r in rs)
        latencies = [r.latency_sec() for r in rs if r.latency_sec() is not None]
        frames = sum(r.frames for r in rs)
        wall = sum(r.wall_sec for r in rs)
        rows.append([
            config,
            _pct(count["TP"], count["TP"] + count["FP"]),
            _pct(count["TP"], count["TP"] + count["FN"]),
            _pct(true_ev, true_ev + false_ev),
            f"{count['FP']}/{count['FP'] + count['TN']}",
            _sec(sum(latencies) / len(latencies)) if latencies else "—",
            f"{frames / wall:.1f}" if wall else "—",
        ])
    return rows


def trap_rows(results: list) -> tuple:
    """함정 종류별 오탐 (오탐 영상 수/영상 수, 오탐 이벤트 수)."""
    categories = list(dict.fromkeys(r.label.category for r in results if r.label.label == NON_SMOKING))
    rows = []
    for config in dict.fromkeys(r.config for r in results):
        row = [config]
        for cat in categories:
            rs = [r for r in results if r.config == config and r.label.category == cat]
            fp_videos = sum(1 for r in rs if r.outcome() == "FP")
            row.append(f"{fp_videos}/{len(rs)} (이벤트 {sum(r.false_events() for r in rs)})")
        rows.append(row)
    return categories, rows


def video_rows(results: list) -> list:
    rows = []
    for r in results:
        rows.append([
            r.label.video, r.label.label, r.config, r.outcome(),
            str(len(r.confirmed_times)),
            _sec(r.confirmed_times[0] if r.confirmed_times else None),
            _sec(r.announced_times[0] if r.announced_times else None),
            f"{r.max_ratio:.0%}",
            _sec(r.latency_sec()),
            r.label.note,
        ])
    return rows


def markdown_table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def build_report(results: list, overrides: list) -> str:
    main = [r for r in results if r.label.split == "main"]
    reference = [r for r in results if r.label.split != "main"]
    first_config = results[0].config if results else ""
    main_labels = [r.label for r in main if r.config == first_config]

    parts = [f"# 평가 결과 ({time.strftime('%Y-%m-%d %H:%M')})", ""]
    if overrides:
        parts += [f"상수 변경: `{', '.join(overrides)}`", ""]
    parts += [
        f"메인 평가 영상 {len(main_labels)}개 "
        f"(흡연 {sum(1 for l in main_labels if l.label == SMOKING)}, "
        f"비흡연 {sum(1 for l in main_labels if l.label == NON_SMOKING)}). "
        "영상 수가 적으면 수치가 크게 흔들리니 참고용으로 볼 것.",
        "",
        "## 요약 (메인)",
        "",
        markdown_table(
            ["설정", "장면 정밀도", "장면 재현율", "이벤트 정밀도", "오탐 영상", "평균 지연(s)", "처리 FPS"],
            summary_rows(main),
        ),
        "",
    ]
    categories, rows = trap_rows(main)
    if categories:
        parts += ["## 함정 종류별 오탐 (메인)", "", markdown_table(["설정"] + categories, rows), ""]

    video_header = ["영상", "정답", "설정", "판정", "확정 수", "첫 확정(s)", "첫 방송(s)", "최고 판정비율", "지연(s)", "메모"]
    parts += ["## 영상별 결과 (메인)", "", markdown_table(video_header, video_rows(main)), ""]
    if reference:
        parts += [
            "## 참고 영상 (메인 수치에서 제외)", "",
            markdown_table(["설정", "장면 정밀도", "장면 재현율", "이벤트 정밀도", "오탐 영상", "평균 지연(s)", "처리 FPS"],
                           summary_rows(reference)),
            "",
            markdown_table(video_header, video_rows(reference)),
            "",
        ]
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="감지 파이프라인 이벤트 단위 정량 평가")
    parser.add_argument("--weights", nargs="+", default=["cigarette-v5-best.pt"], help="2차 확인 모델 가중치 (여러 개면 비교)")
    parser.add_argument("--pose-imgsz", nargs="+", type=int, default=[640], help="포즈 추론 해상도 (여러 개면 비교)")
    parser.add_argument("--pose-model", default="yolov8n-pose.pt", help="포즈 모델 가중치")
    parser.add_argument("--labels", default="eval/labels.csv", help="정답 라벨 CSV")
    parser.add_argument("--video-dir", default="test_videos", help="평가 영상 폴더")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="상수 변경 (예: pipeline.OBJECT_CHECK_INTERVAL_SEC=0.2). 여러 번 지정 가능")
    parser.add_argument("--include-reference", action="store_true", help="split=reference 영상도 돌린다(메인 수치와 분리 표시)")
    parser.add_argument("--out", help="결과를 마크다운 파일로도 저장할 경로")
    args = parser.parse_args()

    overrides = apply_overrides(args.overrides)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    labels = load_labels(Path(args.labels))
    if not args.include_reference:
        labels = [l for l in labels if l.split == "main"]

    results = []
    for weights in args.weights:
        if not Path(weights).is_file():
            raise SystemExit(f"가중치 파일이 없습니다: {weights}")
        for imgsz in args.pose_imgsz:
            config = f"{Path(weights).stem} / pose {imgsz}"
            for label in labels:
                video_path = Path(args.video_dir) / label.video
                if not video_path.is_file():
                    print(f"[경고] 영상이 없어 건너뜁니다: {video_path}")
                    continue
                print(f"[진행] {config} — {label.video}", flush=True)
                r = evaluate_video(label, video_path, weights, args.pose_model, imgsz, device, config)
                print(f"        → {r.outcome()} (확정 {len(r.confirmed_times)}회, 최고 판정비율 {r.max_ratio:.0%}, "
                      f"{r.frames / r.wall_sec:.1f} FPS)", flush=True)
                results.append(r)

    report = build_report(results, overrides)
    print("\n" + report)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report + "\n", encoding="utf-8")
        print(f"\n[정보] 결과 저장: {out_path}")


if __name__ == "__main__":
    main()
