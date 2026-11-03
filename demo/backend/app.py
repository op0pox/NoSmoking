"""
demo/backend/app.py — 웹 데모 백엔드 (FastAPI, localhost 전용)

- 영상 분석: 업로드한 영상을 감지 파이프라인으로 분석하고, 사람별 타임라인과 이벤트를
  JSON으로 돌려준다. 분석 로직은 평가 도구와 같은 edge/video_analysis.py를 쓴다.
- 실시간: 백엔드가 웹캠을 직접 읽어 판정하고, 그린 화면을 MJPEG로 보여준다(demo/backend/live.py).
  안내 음성은 이 PC의 스피커로 재생한다.

실행 (저장소 루트에서):
    NS-venv/bin/uvicorn demo.backend.app:app --host 127.0.0.1 --port 8000

개인정보 원칙 (프로젝트 "영상 무저장" 원칙):
- 127.0.0.1에만 바인딩한다. 같은 PC의 브라우저만 접속할 수 있다.
- 업로드한 영상은 분석하는 동안만 OS 임시 폴더에 두고, 분석이 끝나면(실패해도) 즉시 삭제한다.
- 프레임을 저장하거나 외부로 보내지 않는다. 메모리에 남는 결과는 시각·점수 같은 숫자뿐이다.
  실시간 화면용으로는 가장 최근 프레임 한 장만 메모리에 두고 계속 덮어쓴다.
- 실시간 카메라는 [시작] 요청이 있어야 켜지고, 화면을 보는 곳이 없으면 스스로 꺼진다.
"""

from __future__ import annotations

import json
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
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# edge/ 모듈은 서로를 `from pipeline import ...`처럼 폴더 안 기준으로 import하므로 경로에 추가한다
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "edge"))

import event_filter  # noqa: E402
import motion_gate  # noqa: E402
import pipeline  # noqa: E402
from pipeline import SmokingPipeline  # noqa: E402
from video_analysis import analyze_video, summarize_track  # noqa: E402

from .event_log import EventLog  # noqa: E402
from .camera_source import NetworkCamera, SourceError, mask_credentials, open_local_camera, validate_network_url  # noqa: E402
from .live import LiveSession  # noqa: E402

POSE_MODEL = os.environ.get("NOSMOKING_POSE_MODEL", str(REPO_ROOT / "yolov8n-pose.pt"))
CIGARETTE_MODEL = os.environ.get("NOSMOKING_CIGARETTE_MODEL", str(REPO_ROOT / "cigarette-v5-best.pt"))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FIRST_MP3 = str(REPO_ROOT / "assets" / "audio" / "first_announce.mp3")
SECOND_MP3 = str(REPO_ROOT / "assets" / "audio" / "second_announce.mp3")
DATA_DIR = Path(__file__).resolve().parent / "data"
# 등록한 카메라(설정만)와 판정 기록(숫자만). 둘 다 커밋하지 않는다. 시험용으로 따로 쓰려면 환경 변수로 바꾼다.
CAMERA_CONFIG_PATH = Path(os.environ.get("NOSMOKING_CAMERA_CONFIG", DATA_DIR / "cameras.json"))
EVENT_DB_PATH = Path(os.environ.get("NOSMOKING_EVENT_DB", DATA_DIR / "events.db"))
CAMERA_INDEXES = (0, 1, 2, 3)  # macOS는 아이폰 연속성 카메라가 0번으로 잡힐 수 있어 고를 수 있게 한다
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
        "audio_found": Path(FIRST_MP3).is_file() and Path(SECOND_MP3).is_file(),
        "live_running": _live is not None and _live.running,
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
    if _live is not None and _live.running:
        raise HTTPException(409, "실시간 감지가 켜져 있습니다. 실시간을 정지한 뒤 분석해 주세요.")
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
        "width": analysis.width,
        "height": analysis.height,
        "frames": analysis.frames,
        "wall_sec": analysis.wall_sec,
        "max_ratio": analysis.max_ratio,
        "events": [asdict(e) for e in analysis.events],
        "tracks": tracks,
    }


# ======================================================================
# 실시간 (웹캠)
# ======================================================================
_live: Optional[LiveSession] = None


class LiveStartRequest(BaseModel):
    camera: int = 0                    # 이 PC 카메라 번호 (camera_id가 없을 때)
    camera_id: Optional[str] = None    # 등록한 네트워크 카메라
    pose_imgsz: int = 640
    play_audio: bool = True


# ----------------------------------------------------------------------
# 네트워크 카메라 등록 — 설치 위치 이름과 주소(설정)만 저장한다. 영상은 저장하지 않는다.
# 주소에 카메라 비밀번호가 들어갈 수 있어 파일은 커밋하지 않고(.gitignore), 화면에는 가려서 보낸다.
# ----------------------------------------------------------------------
class CameraCreateRequest(BaseModel):
    name: str
    url: str


def _load_cameras() -> list:
    if not CAMERA_CONFIG_PATH.is_file():
        return []
    return json.loads(CAMERA_CONFIG_PATH.read_text(encoding="utf-8"))


def _save_cameras(cameras: list):
    CAMERA_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAMERA_CONFIG_PATH.write_text(json.dumps(cameras, ensure_ascii=False, indent=2), encoding="utf-8")


def _public_camera(camera: dict) -> dict:
    return {"id": camera["id"], "name": camera["name"], "url": mask_credentials(camera["url"])}


@app.get("/api/cameras")
def list_cameras():
    return [_public_camera(c) for c in _load_cameras()]


@app.post("/api/cameras")
def add_camera(req: CameraCreateRequest):
    name, url = req.name.strip(), req.url.strip()
    if not name:
        raise HTTPException(400, "설치 위치 이름을 입력해 주세요.")
    try:
        validate_network_url(url)
    except SourceError as e:
        raise HTTPException(400, str(e)) from None
    cameras = _load_cameras()
    camera = {"id": uuid.uuid4().hex[:8], "name": name, "url": url}
    cameras.append(camera)
    _save_cameras(cameras)
    return _public_camera(camera)


@app.delete("/api/cameras/{camera_id}")
def delete_camera(camera_id: str):
    cameras = _load_cameras()
    remaining = [c for c in cameras if c["id"] != camera_id]
    if len(remaining) == len(cameras):
        raise HTTPException(404, "해당 카메라가 없습니다.")
    _save_cameras(remaining)
    return {"ok": True}


def _analysis_busy() -> bool:
    with _jobs_lock:
        return any(job["status"] in ("queued", "running") for job in _jobs.values())


@app.post("/api/live/start")
async def live_start(req: LiveStartRequest):
    """카메라를 켜고 실시간 감지를 시작한다.

    - 이 PC 카메라: async 함수라 이벤트 루프(메인 스레드)에서 연다 — macOS는 카메라 권한 요청을
      메인 스레드에서 해야 팝업이 제대로 뜬다.
    - 네트워크 카메라: 연결에 최대 몇 초 걸려서, 다른 요청이 멈추지 않게 작업 스레드에서 연다.
    프레임 읽기·판정은 LiveSession의 작업 스레드가 한다.
    """
    global _live
    if _live is not None and _live.running:
        raise HTTPException(409, "이미 실시간 감지가 켜져 있습니다.")
    if _analysis_busy():
        raise HTTPException(409, "영상 분석이 진행 중입니다. 끝난 뒤 시작해 주세요.")
    if req.pose_imgsz not in (640, 960, 1280):
        raise HTTPException(400, "pose_imgsz는 640, 960, 1280 중 하나여야 합니다.")

    try:
        if req.camera_id is not None:
            camera = next((c for c in _load_cameras() if c["id"] == req.camera_id), None)
            if camera is None:
                raise HTTPException(404, "등록한 카메라를 찾을 수 없습니다.")
            cap = await run_in_threadpool(NetworkCamera, camera["url"])
            mirror, label = False, camera["name"]
        else:
            if req.camera not in CAMERA_INDEXES:
                raise HTTPException(400, f"카메라 번호는 {CAMERA_INDEXES} 중 하나여야 합니다.")
            cap = open_local_camera(req.camera)
            mirror, label = True, f"이 PC 카메라 {req.camera}번"   # 셀카처럼 좌우 반전
    except SourceError as e:
        raise HTTPException(400, str(e)) from None

    _apply_params({})  # 영상 분석에서 바꿔 본 기준이 남지 않게 기본값으로
    pipe = SmokingPipeline(
        POSE_MODEL, CIGARETTE_MODEL, device=DEVICE, pose_imgsz=req.pose_imgsz,
        first_mp3=FIRST_MP3, second_mp3=SECOND_MP3, play_audio=req.play_audio, verbose=False,
    )
    settings = {
        "cigarette_model": Path(CIGARETTE_MODEL).name,
        "pose_imgsz": req.pose_imgsz,
        **_DEFAULT_PARAMS,
    }
    _live = LiveSession(cap, pipe, mirror=mirror, source_label=label, event_log=_event_log, settings=settings)
    _live.start()
    return _live.state()


@app.post("/api/live/stop")
def live_stop():
    if _live is not None:
        _live.stop()
        return _live.state()
    return {"running": False}


@app.get("/api/live/state")
def live_state():
    if _live is None:
        return {"running": False, "people": [], "events": [], "fps": 0, "error": None, "stop_reason": None}
    if _live.running:
        _live.touch()
    return _live.state()


@app.get("/api/live/stream")
def live_stream():
    """판정 결과를 그린 화면을 MJPEG로 보낸다. 브라우저에서 <img src="/api/live/stream">로 본다."""
    session = _live
    if session is None or not session.running:
        raise HTTPException(409, "실시간 감지가 꺼져 있습니다.")

    def frames():
        last_id = 0
        while session.running:
            session.touch()
            last_id, jpeg = session.wait_for_frame(last_id)
            if jpeg is None:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


# ======================================================================
# 판정 기록 (실시간 감지의 이벤트 확정 1건당 1줄, 숫자만)
# ======================================================================
_event_log = EventLog(EVENT_DB_PATH)


class FeedbackRequest(BaseModel):
    value: Optional[str] = None   # "correct"(맞음) / "wrong"(틀림) / null(취소)


@app.get("/api/events")
def list_events(date_from: Optional[str] = None, date_to: Optional[str] = None,
                source: Optional[str] = None, feedback: Optional[str] = None):
    return {
        "events": _event_log.list(date_from, date_to, source, feedback),
        "sources": _event_log.sources(),
        "summary": _event_log.summary(),
    }


@app.post("/api/events/{event_id}/feedback")
def set_event_feedback(event_id: int, req: FeedbackRequest):
    try:
        found = _event_log.set_feedback(event_id, req.value)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    if not found:
        raise HTTPException(404, "해당 기록이 없습니다.")
    return {"ok": True, "summary": _event_log.summary()}
