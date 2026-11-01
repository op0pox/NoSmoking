"""
demo/backend/app.py — 웹 데모 백엔드 (FastAPI, localhost 전용)

영상 분석 페이지가 업로드한 영상을 감지 파이프라인으로 분석하고, 사람별 타임라인과
이벤트를 JSON으로 돌려준다. 분석 로직은 평가 도구와 같은 edge/video_analysis.py를 쓴다.

실행 (저장소 루트에서):
    NS-venv/bin/uvicorn demo.backend.app:app --host 127.0.0.1 --port 8000

개인정보 원칙 (프로젝트 "영상 무저장" 원칙):
- 127.0.0.1에만 바인딩한다. 같은 PC의 브라우저만 접속할 수 있다.
- 업로드한 영상은 분석하는 동안만 OS 임시 폴더에 두고, 분석이 끝나면(실패해도) 즉시 삭제한다.
- 프레임을 저장하거나 외부로 보내지 않는다. 메모리에 남는 결과는 시각·점수 같은 숫자뿐이다.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

# edge/ 모듈은 서로를 `from pipeline import ...`처럼 폴더 안 기준으로 import하므로 경로에 추가한다
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "edge"))

import event_filter  # noqa: E402
import motion_gate  # noqa: E402
import pipeline  # noqa: E402
from pipeline import SmokingPipeline  # noqa: E402
from video_analysis import analyze_video, summarize_track  # noqa: E402

POSE_MODEL = os.environ.get("NOSMOKING_POSE_MODEL", str(REPO_ROOT / "yolov8n-pose.pt"))
CIGARETTE_MODEL = os.environ.get("NOSMOKING_CIGARETTE_MODEL", str(REPO_ROOT / "cigarette-v5-best.pt"))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_KEPT_JOBS = 10   # 메모리에 결과를 들고 있을 최근 분석 수 (숫자 결과만, 영상 없음)

# 화면에서 바꿔 볼 수 있는 상수. (화면용 이름) -> (모듈, 상수 이름)
TUNABLE_PARAMS = {
    "near_threshold": (motion_gate, "NEAR_THRESHOLD"),
    "mouth_cig_conf_threshold": (pipeline, "MOUTH_CIG_CONF_THRESHOLD"),
    "combined_confirm_threshold": (pipeline, "COMBINED_CONFIRM_THRESHOLD"),
    "event_confirm_ratio": (event_filter, "EVENT_CONFIRM_RATIO"),
}

app = FastAPI(title="금연존 지킴이 데모 백엔드")

# 분석은 한 번에 하나씩만 돌린다 — 상수 변경(TUNABLE_PARAMS)이 모듈 전역 값이라 동시에 돌리면 섞인다
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict = {}
_jobs_lock = threading.Lock()


# 서버 시작 시점의 값 = 코드에 적힌 기본값
_DEFAULT_PARAMS = {name: getattr(module, const) for name, (module, const) in TUNABLE_PARAMS.items()}


@app.middleware("http")
async def only_local_clients(request: Request, call_next):
    """127.0.0.1로 띄우는 게 원칙이지만, 실수로 0.0.0.0에 띄워도 다른 PC의 요청은 거절한다."""
    if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
        return JSONResponse({"detail": "이 데모는 같은 PC에서만 접속할 수 있습니다."}, status_code=403)
    return await call_next(request)


@app.get("/api/status")
def status():
    """홈 화면의 상태 표시용: 백엔드·모델·디바이스·기본 파라미터."""
    return {
        "ok": True,
        "device": DEVICE,
        "pose_model": Path(POSE_MODEL).name,
        "cigarette_model": Path(CIGARETTE_MODEL).name,
        "models_found": Path(POSE_MODEL).is_file() and Path(CIGARETTE_MODEL).is_file(),
        "default_params": _DEFAULT_PARAMS,
        "event_window_sec": event_filter.EVENT_WINDOW_SEC,
    }


@app.post("/api/analyze")
async def start_analysis(
    file: UploadFile = File(...),
    pose_imgsz: int = Form(640),
    near_threshold: Optional[float] = Form(None),
    mouth_cig_conf_threshold: Optional[float] = Form(None),
    combined_confirm_threshold: Optional[float] = Form(None),
    event_confirm_ratio: Optional[float] = Form(None),
):
    """영상을 받아 분석 작업을 등록하고 job_id를 돌려준다. 결과는 GET /api/analyze/{job_id}로 확인."""
    if pose_imgsz not in (640, 960, 1280, 1920):
        raise HTTPException(400, "pose_imgsz는 640, 960, 1280, 1920 중 하나여야 합니다.")
    overrides = {
        name: value for name, value in {
            "near_threshold": near_threshold,
            "mouth_cig_conf_threshold": mouth_cig_conf_threshold,
            "combined_confirm_threshold": combined_confirm_threshold,
            "event_confirm_ratio": event_confirm_ratio,
        }.items() if value is not None
    }

    # 분석하는 동안만 쓰는 임시 파일. 분석 작업이 끝나면 _run_job의 finally에서 지운다.
    suffix = Path(file.filename or "").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="nosmoking_") as tmp:
        while chunk := await file.read(1024 * 1024):
            tmp.write(chunk)
        temp_path = tmp.name

    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued", "progress": 0.0, "error": None, "result": None,
            "file_name": file.filename, "pose_imgsz": pose_imgsz, "params": {**_DEFAULT_PARAMS, **overrides},
        }
        _forget_old_jobs()
    _executor.submit(_run_job, job_id, temp_path, pose_imgsz, overrides)
    return {"job_id": job_id}


@app.get("/api/analyze/{job_id}")
def get_analysis(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "해당 분석 작업이 없습니다 (오래돼서 지워졌을 수 있습니다).")
        return dict(job)


def _forget_old_jobs():
    """오래된 결과부터 지워 최근 MAX_KEPT_JOBS개만 남긴다 (_jobs_lock 안에서 호출)."""
    while len(_jobs) > MAX_KEPT_JOBS:
        del _jobs[next(iter(_jobs))]


def _update_job(job_id: str, **fields):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def _run_job(job_id: str, temp_path: str, pose_imgsz: int, overrides: dict):
    try:
        _update_job(job_id, status="running")
        _apply_params(overrides)
        pipe = SmokingPipeline(
            POSE_MODEL, CIGARETTE_MODEL, device=DEVICE, pose_imgsz=pose_imgsz,
            play_audio=False, verbose=False,
        )
        analysis = analyze_video(
            temp_path, pipe, record_timeline=True,
            on_progress=lambda done, total: _update_job(job_id, progress=done / total if total else 0.0),
        )
        _update_job(job_id, status="done", progress=1.0, result=_to_response(analysis))
    except Exception as e:  # 실패 내용을 화면에 보여준다
        _update_job(job_id, status="error", error=str(e))
    finally:
        _apply_params({})  # 기본값으로 되돌린다
        os.remove(temp_path)  # 업로드 영상은 분석이 끝나면 반드시 지운다


def _apply_params(overrides: dict):
    """화면에서 지정한 상수를 적용한다. overrides에 없는 상수는 기본값으로 되돌린다."""
    for name, (module, const) in TUNABLE_PARAMS.items():
        setattr(module, const, overrides.get(name, _DEFAULT_PARAMS[name]))


def _rounded(values: dict) -> dict:
    """응답 크기를 줄이려고 소수는 셋째 자리까지만 보낸다 (영상이 길면 샘플이 수만 개다)."""
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in values.items()}


def _to_response(analysis) -> dict:
    tracks = []
    for track_id, samples in analysis.timeline.items():
        summary = summarize_track(track_id, samples, analysis.events)
        tracks.append({"summary": asdict(summary), "samples": [_rounded(asdict(s)) for s in samples]})
    # 확정된 사람 → 2차 확인을 많이 받은 사람 순으로 (화면에서 중요한 사람이 위로 오게)
    tracks.sort(key=lambda t: (not t["summary"]["confirmed"], -t["summary"]["checks"]))
    return {
        "duration_sec": analysis.duration_sec,
        "fps": analysis.fps,
        "frames": analysis.frames,
        "wall_sec": analysis.wall_sec,
        "max_ratio": analysis.max_ratio,
        "events": [asdict(e) for e in analysis.events],
        "tracks": tracks,
    }
