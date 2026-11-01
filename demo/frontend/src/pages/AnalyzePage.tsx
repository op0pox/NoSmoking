import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  fetchAnalysis,
  fetchStatus,
  startAnalysis,
  type AnalysisJob,
  type AnalysisParams,
  type AnalysisResult,
  type TrackResult,
} from "../api";
import Timeline from "../components/Timeline";

const POLL_INTERVAL_MS = 1000;
const POSE_IMGSZ_OPTIONS = [640, 1280];

// 화면에서 바꿔 볼 수 있는 파라미터 (이름은 api.ts의 AnalysisParams와 같다)
const PARAM_FIELDS: { key: keyof AnalysisParams; label: string; step: number; min: number; max: number }[] = [
  { key: "near_threshold", label: "손-입 거리 기준 (NEAR_THRESHOLD)", step: 0.05, min: 0.1, max: 1.5 },
  { key: "mouth_cig_conf_threshold", label: "입 근처 담배 신뢰도 기준", step: 0.05, min: 0.05, max: 1 },
  { key: "combined_confirm_threshold", label: "짝 점수 기준 (combined)", step: 0.05, min: 0.05, max: 1 },
  { key: "event_confirm_ratio", label: "시간 필터 확정 비율", step: 0.05, min: 0.05, max: 1 },
];

export default function AnalyzePage() {
  const [defaultParams, setDefaultParams] = useState<AnalysisParams | null>(null);
  const [params, setParams] = useState<AnalysisParams | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [poseImgsz, setPoseImgsz] = useState(640);
  // 작업 번호를 주소(?job=…)에 남겨 새로고침해도 결과를 다시 볼 수 있게 한다 (결과는 백엔드 메모리에 최근 10개까지)
  const [searchParams, setSearchParams] = useSearchParams();
  const jobId = searchParams.get("job");
  const [job, setJob] = useState<AnalysisJob | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchStatus()
      .then((s) => {
        setDefaultParams(s.default_params);
        setParams(s.default_params);
      })
      .catch((e: Error) => setError(`백엔드에 연결할 수 없습니다 — ${e.message}`));
  }, []);

  // 분석이 끝날 때까지 1초마다 진행 상황을 묻는다
  useEffect(() => {
    if (!jobId) return;
    let stopped = false;
    const poll = async () => {
      try {
        const next = await fetchAnalysis(jobId);
        if (stopped) return;
        setJob(next);
        if (next.status === "done" || next.status === "error") return;
      } catch (e) {
        if (!stopped) setError((e as Error).message);
        return;
      }
      setTimeout(poll, POLL_INTERVAL_MS);
    };
    poll();
    return () => {
      stopped = true;
    };
  }, [jobId]);

  const running = job !== null && (job.status === "queued" || job.status === "running");

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!file || !params) return;
    setError(null);
    setJob(null);
    try {
      setSearchParams({ job: await startAnalysis(file, poseImgsz, params) });
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <>
      <h1>영상 분석</h1>
      <p className="secondary">영상을 올리면 판정 과정을 사람별·시간별로 보여줍니다.</p>
      <span className="privacy-note">🔒 영상은 이 PC에서만 분석되고, 분석이 끝나면 바로 삭제됩니다</span>

      <form className="card" style={{ marginTop: 16 }} onSubmit={onSubmit}>
        <div className="form-grid">
          <label>
            영상 파일
            <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </label>
          <label>
            포즈 추론 해상도
            <select value={poseImgsz} onChange={(e) => setPoseImgsz(Number(e.target.value))}>
              {POSE_IMGSZ_OPTIONS.map((v) => (
                <option key={v} value={v}>{v}{v > 640 ? " (사람이 작게 나오는 영상, 느림)" : " (기본)"}</option>
              ))}
            </select>
          </label>
        </div>

        {params && defaultParams && (
          <details>
            <summary>판정 기준 바꿔 보기</summary>
            <div className="form-grid">
              {PARAM_FIELDS.map((f) => (
                <label key={f.key}>
                  {f.label}
                  <input
                    type="number" step={f.step} min={f.min} max={f.max} value={params[f.key]}
                    onChange={(e) => setParams({ ...params, [f.key]: Number(e.target.value) })}
                  />
                  <span className="muted">기본값 {defaultParams[f.key]}</span>
                </label>
              ))}
            </div>
            <button type="button" className="ghost" style={{ marginTop: 12 }} onClick={() => setParams(defaultParams)}>
              기본값으로
            </button>
          </details>
        )}

        <div style={{ marginTop: 16 }}>
          <button type="submit" disabled={!file || !params || running}>
            {running ? "분석 중…" : "분석 시작"}
          </button>
        </div>
        {running && job && (
          <>
            <div className="progress"><div style={{ width: `${Math.round(job.progress * 100)}%` }} /></div>
            <p className="muted">
              {job.status === "queued" ? "대기 중" : `${Math.round(job.progress * 100)}% 처리`} — 영상 길이와 해상도에 따라 수십 초~몇 분 걸립니다.
            </p>
          </>
        )}
        {error && <p className="error">⚠️ {error}</p>}
        {job?.status === "error" && <p className="error">⚠️ 분석 실패 — {job.error}</p>}
      </form>

      {job?.status === "done" && job.result && <AnalysisView key={jobId} job={job} result={job.result} />}
    </>
  );
}

// ---------------------------------------------------------------------------

function AnalysisView({ job, result }: { job: AnalysisJob; result: AnalysisResult }) {
  const [onlyChecked, setOnlyChecked] = useState(true);
  const [selectedId, setSelectedId] = useState<number | null>(result.tracks[0]?.summary.track_id ?? null);
  const [showTable, setShowTable] = useState(false);

  const confirms = result.events.filter((e) => e.kind === "confirmed");
  const announces = result.events.filter((e) => e.kind === "announced");
  const visibleTracks = onlyChecked ? result.tracks.filter((t) => t.summary.checks > 0) : result.tracks;
  const selected = useMemo<TrackResult | undefined>(
    () => result.tracks.find((t) => t.summary.track_id === selectedId),
    [result, selectedId],
  );

  return (
    <>
      <h2>결과 — {job.file_name}</h2>
      <div className="card">
        <div className="verdict">
          {confirms.length > 0 ? (
            <>
              <span className="badge confirmed">🔔 이벤트 확정</span>
              {confirms.length}회 · 첫 확정 {confirms[0].time.toFixed(1)}초
              {announces.length > 0 && ` · 첫 방송 ${announces[0].time.toFixed(1)}초`}
            </>
          ) : (
            <>
              <span className="badge none">✓ 이벤트 없음</span>
              시간 필터 최고 판정 비율 {Math.round(result.max_ratio * 100)}% (기준 {Math.round(job.params.event_confirm_ratio * 100)}%)
            </>
          )}
        </div>
        <div className="meta">
          <span>영상 {result.duration_sec.toFixed(1)}초 · {result.fps.toFixed(0)}fps · {result.frames}프레임</span>
          <span>처리 {result.wall_sec.toFixed(1)}초</span>
          <span>포즈 {job.pose_imgsz}</span>
          <span>
            기준: 손-입 {job.params.near_threshold} · 입 근처 담배 {job.params.mouth_cig_conf_threshold} · 짝 점수{" "}
            {job.params.combined_confirm_threshold} · 확정 비율 {Math.round(job.params.event_confirm_ratio * 100)}%
          </span>
        </div>
      </div>

      {result.events.length > 0 && (
        <>
          <h2>이벤트</h2>
          <div className="card table-wrap">
            <table>
              <thead>
                <tr><th className="num">시각</th><th>사람</th><th>종류</th></tr>
              </thead>
              <tbody>
                {result.events.map((e, i) => (
                  <tr key={i}>
                    <td className="num">{e.time.toFixed(1)}초</td>
                    <td>#{e.track_id}</td>
                    <td>{e.kind === "confirmed" ? `🔔 이벤트 확정 (판정 비율 ${e.detail})` : `📢 ${e.detail} 안내 방송`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      <h2>사람별 판정</h2>
      <div className="card">
        <div className="toolbar">
          <span className="secondary">행을 누르면 아래에 그 사람의 타임라인이 나옵니다.</span>
          <label style={{ flexDirection: "row", alignItems: "center" }}>
            <input type="checkbox" checked={onlyChecked} onChange={(e) => setOnlyChecked(e.target.checked)} />
            2차 확인을 받은 사람만 ({result.tracks.filter((t) => t.summary.checks > 0).length}/{result.tracks.length})
          </label>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>사람</th><th className="num">보인 구간</th><th className="num">puff</th><th className="num">2차 확인</th>
                <th className="num">흡연 판정</th><th className="num">담배 / 입 근처 / 연기 최고</th><th className="num">최고 비율</th>
                <th>판정 근거</th>
              </tr>
            </thead>
            <tbody>
              {visibleTracks.map(({ summary: s }) => (
                <tr
                  key={s.track_id}
                  className={`clickable${s.track_id === selectedId ? " selected" : ""}`}
                  onClick={() => setSelectedId(s.track_id)}
                >
                  <td>{s.confirmed ? "🔔 " : ""}#{s.track_id}</td>
                  <td className="num">{s.first_t.toFixed(0)}~{s.last_t.toFixed(0)}초</td>
                  <td className="num">{s.max_puffs}</td>
                  <td className="num">{s.checks}</td>
                  <td className="num">{s.smoking_frames}</td>
                  <td className="num">{s.max_cig.toFixed(2)} / {s.max_mouth_cig.toFixed(2)} / {s.max_smoke.toFixed(2)}</td>
                  <td className="num">{Math.round(s.max_ratio * 100)}%</td>
                  <td>{s.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {selected && (
        <>
          <h2>사람 #{selected.summary.track_id} 타임라인</h2>
          <div className="card">
            <p className="secondary" style={{ marginTop: 0 }}>{selected.summary.reason}</p>
            <Timeline
              samples={selected.samples}
              events={result.events.filter((e) => e.track_id === selected.summary.track_id)}
              params={job.params}
              durationSec={result.duration_sec}
            />
            <button type="button" className="ghost" style={{ marginTop: 16 }} onClick={() => setShowTable(!showTable)}>
              {showTable ? "표 닫기" : "2차 확인 값 표로 보기"}
            </button>
            {showTable && <CheckTable track={selected} />}
          </div>
        </>
      )}
    </>
  );
}

// 차트 값을 표로도 볼 수 있게 한다 (색만으로 읽기 어려운 경우 대비)
function CheckTable({ track }: { track: TrackResult }) {
  const rows = track.samples.filter((s) => s.checked);
  return (
    <div className="table-wrap" style={{ marginTop: 12, maxHeight: 360, overflowY: "auto" }}>
      <table>
        <thead>
          <tr>
            <th className="num">시각</th><th className="num">담배</th><th className="num">입 근처 담배</th><th className="num">연기</th>
            <th className="num">짝 점수</th><th>판정</th><th className="num">10초 비율</th><th>1차 게이트</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((s, i) => (
            <tr key={i}>
              <td className="num">{s.t.toFixed(1)}초</td>
              <td className="num">{s.cig?.toFixed(2)}</td>
              <td className="num">{s.mouth_cig?.toFixed(2)}</td>
              <td className="num">{s.smoke?.toFixed(2)}</td>
              <td className="num">{s.combined?.toFixed(2)}</td>
              <td>{s.smoking_frame ? "흡연" : "—"}</td>
              <td className="num">{s.ratio !== null ? `${Math.round(s.ratio * 100)}%` : "—"}</td>
              <td>{s.state} (puff {s.puff_count})</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
