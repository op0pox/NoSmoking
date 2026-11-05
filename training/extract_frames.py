"""
training/extract_frames.py — 학습용 영상에서 라벨링용 사진 뽑기

흡연 영상(CCTV 데이터셋 등)에서 사진을 일정 간격으로 뽑아 Roboflow에 올릴 수 있게 저장한다.
뽑은 사진에는 사람이 나오므로 개인정보다. 저장 위치는 깃에 안 올라가는 폴더만 허용한다.

※ 이 스크립트는 학습 데이터 준비 전용이다. 현장 장치(edge/)는 '영상 무저장' 원칙에 따라
  프레임을 디스크에 쓰지 않는다. edge/ 코드에서 이 파일을 가져다 쓰지 말 것.

사용 (저장소 루트에서):
    python training/extract_frames.py E04_012.mp4 E04_043.mp4
    python training/extract_frames.py test_videos/E04_0*.mp4 --fps 2
    python training/extract_frames.py E04_012.mp4 --labels-dir test_videos/labels --event-only
    python training/extract_frames.py E04_012.mp4 --out datasets/frames_day --max-frames 100

결과: <out>/<영상이름>/<영상이름>_f<프레임번호>.jpg
      파일 이름에 영상 이름이 들어 있어, Roboflow에서 학습/검증을 영상 단위로 나눌 수 있다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── 튜닝 상수 ────────────────────────────────────────────
DEFAULT_FPS = 1.0              # 1초에 몇 장 뽑을지. 연속 사진은 거의 같아서 많이 뽑아도 도움이 적다
DEFAULT_OUT_DIR = "datasets/frames"   # datasets/는 .gitignore에 들어 있다
# 중복 판정: 직전에 저장한 사진과 비교해 '바뀐 픽셀' 비율이 아주 작으면 같은 장면으로 보고 건너뜀.
# 밝기 차이 평균은 압축 노이즈와 손동작을 구분하지 못해서(E04 영상에서 실측) 바뀐 픽셀 수로 본다.
# 멀리 있는 사람의 손동작도 남기도록 기준을 아주 낮게 잡았다 — 사실상 '완전히 멈춘 장면'만 뺀다.
DUPLICATE_CHANGED_RATIO = 0.0002  # 바뀐 픽셀이 전체의 0.02% 미만이면 중복
PIXEL_CHANGE_THRESHOLD = 25       # 픽셀 하나가 '바뀌었다'고 볼 밝기 차이 (0~255)
DIFF_RESIZE_WIDTH = 320           # 중복 비교는 작게 줄여서 한다 (속도·노이즈 감소)
LABEL_END_TOLERANCE = 30          # 라벨 구간 끝이 영상 길이보다 이만큼(프레임) 넘게 길 때만 경고
JPEG_QUALITY = 95              # 라벨링용이라 화질을 높게 둔다

REPO_ROOT = Path(__file__).resolve().parent.parent


def is_git_ignored(path: Path) -> bool:
    """저장소 안의 경로라면 .gitignore에 걸리는지 확인한다. 저장소 밖이면 True."""
    try:
        path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return True  # 저장소 밖은 커밋될 일이 없다
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(path)],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return result.returncode == 0


def load_event_range(labels_dir: Path, video_path: Path, frame_count: int) -> Optional[tuple[int, int]]:
    """라벨 json의 event_frame(군집 구간)을 읽는다. 파일이 없으면 None."""
    label_path = labels_dir / f"{video_path.stem}.json"
    if not label_path.exists():
        return None
    with open(label_path, encoding="utf-8") as f:
        data = json.load(f)
    start, end = data["annotations"]["event_frame"][0]
    if end > frame_count + LABEL_END_TOLERANCE:
        # E04_036처럼 구간 끝이 영상 길이보다 훨씬 긴 라벨이 있다 → 라벨 오류로 보고 알린다
        print(f"  ! 라벨 구간 끝({end})이 영상 길이({frame_count})보다 깁니다. 영상 끝까지로 자릅니다.")
    end = min(end, frame_count)  # 1~2프레임 차이는 흔해서 조용히 맞춘다
    return start, end


def small_gray(frame: np.ndarray) -> np.ndarray:
    """중복 비교용으로 작게 줄이고 살짝 흐리게 한 흑백 사진 (흐리게 해서 압축 노이즈를 줄인다)."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (DIFF_RESIZE_WIDTH, int(h * DIFF_RESIZE_WIDTH / w)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0).astype(np.int16)


def is_same_scene(a: np.ndarray, b: np.ndarray) -> bool:
    """바뀐 픽셀 비율이 기준보다 작으면 같은 장면."""
    changed_ratio = float(np.mean(np.abs(a - b) > PIXEL_CHANGE_THRESHOLD))
    return changed_ratio < DUPLICATE_CHANGED_RATIO


def extract_video(
    video_path: Path,
    out_dir: Path,
    fps: float,
    event_range: Optional[tuple[int, int]],
    max_frames: Optional[int],
    skip_duplicates: bool,
) -> int:
    """영상 하나에서 사진을 뽑아 저장하고, 저장한 장수를 돌려준다."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ! 영상을 열 수 없습니다: {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(video_fps / fps))  # 몇 프레임마다 한 장씩 뽑을지

    start, end = 0, frame_count
    if event_range is not None:
        start, end = event_range
        print(f"  군집 구간만: {start / video_fps:.1f}초 ~ {end / video_fps:.1f}초")

    video_out = out_dir / video_path.stem
    video_out.mkdir(parents=True, exist_ok=True)

    saved = skipped = 0
    last_small: Optional[np.ndarray] = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_idx = start

    while frame_idx < end:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx - start) % step == 0:
            current_small = small_gray(frame)
            is_duplicate = (
                skip_duplicates
                and last_small is not None
                and is_same_scene(current_small, last_small)
            )
            if is_duplicate:
                skipped += 1
            else:
                name = f"{video_path.stem}_f{frame_idx:06d}.jpg"
                cv2.imwrite(str(video_out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                last_small = current_small
                saved += 1
                if max_frames is not None and saved >= max_frames:
                    break
        frame_idx += 1

    cap.release()
    print(f"  저장 {saved}장, 중복이라 건너뜀 {skipped}장 → {video_out}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="학습용 영상에서 라벨링용 사진을 뽑는다.")
    parser.add_argument("videos", nargs="+", type=Path, help="영상 파일 경로 (여러 개 가능)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / DEFAULT_OUT_DIR,
                        help=f"저장 폴더 (기본: {DEFAULT_OUT_DIR}). 깃에 안 올라가는 곳만 허용")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help=f"1초에 몇 장 (기본: {DEFAULT_FPS})")
    parser.add_argument("--labels-dir", type=Path, help="라벨 json 폴더 (예: test_videos/labels)")
    parser.add_argument("--event-only", action="store_true", help="라벨의 군집 구간(event_frame)만 뽑기")
    parser.add_argument("--max-frames", type=int, help="영상 하나당 최대 몇 장")
    parser.add_argument("--keep-duplicates", action="store_true", help="거의 같은 연속 사진도 모두 저장")
    args = parser.parse_args()

    if args.event_only and args.labels_dir is None:
        parser.error("--event-only를 쓰려면 --labels-dir도 지정해야 합니다.")
    if args.fps <= 0:
        parser.error("--fps는 0보다 커야 합니다.")

    # 사람이 나오는 사진이라, 실수로 커밋될 수 있는 폴더에는 저장하지 않는다
    if not is_git_ignored(args.out):
        sys.exit(f"저장 폴더 {args.out}가 .gitignore에 없습니다. 사진이 커밋될 수 있어 중단합니다.\n"
                 f"기본 폴더({DEFAULT_OUT_DIR})를 쓰거나 저장소 밖 폴더를 지정하세요.")
    args.out.mkdir(parents=True, exist_ok=True)

    total = 0
    for video_path in args.videos:
        print(f"[{video_path.name}]")
        event_range = None
        if args.event_only:
            cap = cv2.VideoCapture(str(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            event_range = load_event_range(args.labels_dir, video_path, frame_count)
            if event_range is None:
                print("  ! 라벨 파일이 없어 건너뜁니다.")
                continue
        total += extract_video(video_path, args.out, args.fps, event_range,
                               args.max_frames, not args.keep_duplicates)

    print(f"\n총 {total}장 저장. 사람이 나오는 사진이니 Roboflow에만 올리고 저장소에는 커밋하지 마세요.")


if __name__ == "__main__":
    main()
