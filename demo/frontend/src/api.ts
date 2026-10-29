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
  checked: boolean;      // 이 프레임에 2차 확인을 돌렸나 (아래 값들은 이때만 채워짐)
  cig: number | null;
  smoke: number | null;
  combined: number | null;
  mouth_cig: number | null;
  smoking_frame: boolean | null;
  ratio: number | null;
  confirmed: boolean | null;
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
  max_ratio: number;
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
