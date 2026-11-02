// 백엔드(demo/backend/app.py) 응답 형식과 호출 함수.
// 필드 이름은 파이썬 쪽 dataclass(edge/video_analysis.py, edge/pipeline.py)와 똑같이 맞춘다.

export interface AnalysisParams {
  near_threshold: number;             // 손-입 정규화 거리 기준 (motion_gate.NEAR_THRESHOLD)
  mouth_cig_conf_threshold: number;   // 입 근처 담배 단독 인정 기준 (pipeline.MOUTH_CIG_CONF_THRESHOLD)
  combined_confirm_threshold: number; // 짝 점수 기준 (pipeline.COMBINED_CONFIRM_THRESHOLD)
  event_confirm_ratio: number;        // 시간 필터 확정 비율 (event_filter.EVENT_CONFIRM_RATIO)
}

export interface StatusResponse {
  ok: boolean;
  device: string;
  pose_model: string;
  cigarette_model: string;
  models_found: boolean;
  default_params: AnalysisParams;
  event_window_sec: number;
  audio_found: boolean;     // 안내 음성 mp3가 있나
  live_running: boolean;
}

export interface PipelineEvent {
  time: number;          // 영상 속 시각(초)
  track_id: number;
  kind: "confirmed" | "announced";
  detail: string;        // confirmed: 판정 비율, announced: "1차" / "2차"
}

export interface TrackSample {
  t: number;
  state: string;         // IDLE / PUFF / SMOKING_SUSPECTED
  puff_count: number;
  dist: number | null;
  box: [number, number, number, number] | null;  // 사람 박스 (원본 영상 픽셀 좌표)
  checked: boolean;      // 이 프레임에 2차 확인을 돌렸나 (아래 값들은 이때만 채워짐)
  cig: number | null;
  smoke: number | null;
  combined: number | null;
  mouth_cig: number | null;
  smoking_frame: boolean | null;
  ratio: number | null;
  ready: boolean | null;   // 관측 10초 이상 — 확정이 가능한 구간
  confirmed: boolean | null;
  objects: DetectedObject[] | null;  // 2차 확인에서 잡힌 담배·연기 박스
}

export interface DetectedObject {
  class: "cigarette" | "smoke";
  conf: number;
  box: [number, number, number, number];  // 원본 영상 픽셀 좌표
}

export interface TrackSummary {
  track_id: number;
  first_t: number;
  last_t: number;
  max_puffs: number;
  checks: number;
  smoking_frames: number;
  max_cig: number;
  max_smoke: number;
  max_mouth_cig: number;
  max_ratio: number;        // 확정이 가능한 구간의 최고 비율
  max_ratio_early: number;  // 관측 10초 전의 최고 비율 (확정에는 안 쓰임)
  confirmed: boolean;
  reason: string;
}

export interface TrackResult {
  summary: TrackSummary;
  samples: TrackSample[];
}

export interface AnalysisResult {
  duration_sec: number;
  fps: number;
  width: number;         // 영상 가로·세로(픽셀)
  height: number;
  frames: number;
  wall_sec: number;
  max_ratio: number;
  events: PipelineEvent[];
  tracks: TrackResult[];
}

export interface AnalysisJob {
  status: "queued" | "running" | "done" | "error";
  progress: number;      // 0~1
  error: string | null;
  result: AnalysisResult | null;
  file_name: string;
  pose_imgsz: number;
  params: AnalysisParams;
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `요청 실패 (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export async function fetchStatus(): Promise<StatusResponse> {
  return parseJson(await fetch("/api/status"));
}

export async function startAnalysis(file: File, poseImgsz: number, params: AnalysisParams): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  form.append("pose_imgsz", String(poseImgsz));
  for (const [name, value] of Object.entries(params)) {
    form.append(name, String(value));
  }
  const body = await parseJson<{ job_id: string }>(await fetch("/api/analyze", { method: "POST", body: form }));
  return body.job_id;
}

export async function fetchAnalysis(jobId: string): Promise<AnalysisJob> {
  return parseJson(await fetch(`/api/analyze/${jobId}`));
}

// ---------------------------------------------------------------------------
// 실시간 (demo/backend/live.py)

export interface LivePerson {
  track_id: number;
  state: string;                 // IDLE / PUFF / SMOKING_SUSPECTED
  puff_count: number;
  dist: number | null;
  cig: number | null;            // 아래 4개는 2차 확인을 한 번이라도 돈 뒤부터 채워진다
  smoke: number | null;
  mouth_cig: number | null;
  ratio: number | null;
  confirmed: boolean;
  announcer_state: string | null; // IDLE / FIRST_ANNOUNCED / COOLDOWN
}

export interface LiveEvent {
  clock: string;                 // 발생 시각 (HH:MM:SS)
  track_id: number;
  kind: "confirmed" | "announced";
  detail: string;
}

export interface LiveState {
  running: boolean;
  fps: number;
  error: string | null;
  stop_reason: string | null;
  source_label: string;          // 카메라 이름 (예: "본관 뒤 금연구역", "이 PC 카메라 0번")
  note: string | null;           // 일시적인 안내 (예: 네트워크 카메라 재연결 중)
  people: LivePerson[];
  events: LiveEvent[];           // 최신이 앞
}

// 실시간 입력: 이 PC 카메라 번호 또는 등록한 네트워크 카메라 id
export type LiveSource = { kind: "local"; camera: number } | { kind: "network"; cameraId: string };

export async function startLive(source: LiveSource, poseImgsz: number, playAudio: boolean): Promise<LiveState> {
  const sourceBody = source.kind === "local" ? { camera: source.camera } : { camera_id: source.cameraId };
  return parseJson(await fetch("/api/live/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...sourceBody, pose_imgsz: poseImgsz, play_audio: playAudio }),
  }));
}

// ---------------------------------------------------------------------------
// 네트워크 카메라 등록 (설치 위치 이름 + 주소. 비밀번호는 서버가 가려서 보낸다)

export interface NetworkCameraInfo {
  id: string;
  name: string;
  url: string;   // rtsp://아이디:****@192.168.0.30:554/... 처럼 비밀번호가 가려진 주소
}

export async function listCameras(): Promise<NetworkCameraInfo[]> {
  return parseJson(await fetch("/api/cameras"));
}

export async function addCamera(name: string, url: string): Promise<NetworkCameraInfo> {
  return parseJson(await fetch("/api/cameras", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, url }),
  }));
}

export async function deleteCamera(id: string): Promise<void> {
  await parseJson(await fetch(`/api/cameras/${id}`, { method: "DELETE" }));
}

// keepalive: 페이지를 떠나는 중에도 정지 요청이 끝까지 가도록 한다
export async function stopLive(): Promise<void> {
  await fetch("/api/live/stop", { method: "POST", keepalive: true });
}

export async function fetchLiveState(): Promise<LiveState> {
  return parseJson(await fetch("/api/live/state"));
}
