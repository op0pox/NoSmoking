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
import VideoReview from "../components/VideoReview";

const POLL_INTERVAL_MS = 1000;
const POSE_IMGSZ_OPTIONS = [640, 1280];
const PEOPLE_ROWS = 6;   // 사람별 판정 표는 처음에 이만큼만 보여주고 "모두 보기"로 펼친다

// 화면에서 바꿔 볼 수 있는 파라미터 (이름은 api.ts의 AnalysisParams와 같다)
const PARAM_FIELDS: { key: keyof AnalysisParams; label: string; step: number; min: number; max: number }[] = [
  { key: "near_threshold", label: "손-입 거리 기준", step: 0.05, min: 0.1, max: 1.5 },
  { key: "mouth_cig_conf_threshold", label: "입 근처 담배 신뢰도 기준", step: 0.05, min: 0.05, max: 1 },
  { key: "combined_confirm_threshold", label: "짝 점수 기준", step: 0.05, min: 0.05, max: 1 },
  { key: "event_confirm_ratio", label: "확정 비율 (10초 중)", step: 0.05, min: 0.05, max: 1 },
];

// 영상 분석(검증 도구): 녹화한 영상으로 판정 과정을 사람별·시간별로 확인한다.
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
      .catch((e: Error) => setError(`백엔드에 연결할 수 없습니다. ${e.message}`));
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
      <div className="title-row">
        <div>
          <h1>영상 분석</h1>
          <p className="sub">
            녹화한 영상으로 판정 과정을 사람별·시간별로 검증합니다. 영상은 이 PC에서만 분석하고, 분석이 끝나면 바로 지웁니다.
          </p>
        </div>
      </div>

      <form className="panel form" onSubmit={onSubmit} aria-label="분석할 영상">
        <div className="grid">
          <label className="field span-6">
            <span>영상 파일</span>
            <input className="input" type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </label>
          <label className="field span-3">
            <span>포즈 추론 해상도</span>
            <select className="input" value={poseImgsz} onChange={(e) => setPoseImgsz(Number(e.target.value))}>
              {POSE_IMGSZ_OPTIONS.map((v) => (
                <option key={v} value={v}>{v}{v > 640 ? " (작게 찍힌 사람용, 느림)" : " (기본)"}</option>
              ))}
            </select>
          </label>
          <div className="span-3" style={{ alignSelf: "end" }}>
            <button type="submit" className="btn primary" style={{ width: "100%", height: 38 }} disabled={!file || !params || running}>
              {running ? "분석 중…" : "분석 시작"}
            </button>
          </div>
        </div>

        {params && defaultParams && (
          <details style={{ marginTop: 16 }}>
            <summary>판정 기준 바꿔 보기</summary>
            <div className="grid" style={{ marginTop: 12 }}>
              {PARAM_FIELDS.map((f) => (
                <label key={f.key} className="field span-3">
                  <span>{f.label}</span>
                  <input
                    className="input" type="number" step={f.step} min={f.min} max={f.max} value={params[f.key]}
                    onChange={(e) => setParams({ ...params, [f.key]: Number(e.target.value) })}
                  />
                  <span className="hint">기본값 {defaultParams[f.key]}</span>
                </label>
              ))}
            </div>
            <button type="button" className="btn small" style={{ marginTop: 12 }} onClick={() => setParams(defaultParams)}>
              기본값으로
            </button>
          </details>
        )}

        {running && job && (
          <div style={{ marginTop: 16 }}>
            <div className="progress"><div style={{ width: `${Math.round(job.progress * 100)}%` }} /></div>
            <p className="small muted" style={{ marginTop: 8 }}>
              {job.status === "queued" ? "대기 중" : `${Math.round(job.progress * 100)}% 처리`}. 영상 길이와 해상도에 따라 수십 초~몇 분 걸립니다.
            </p>
          </div>
        )}
        {error && <p className="error" style={{ marginTop: 12 }}>{error}</p>}
        {job?.status === "error" && <p className="error" style={{ marginTop: 12 }}>분석하지 못했습니다. {job.error}</p>}
      </form>

      {job?.status === "done" && job.result && (
        <AnalysisView key={jobId} job={job} result={job.result} file={file} onPickFile={setFile} />
      )}
    </>
  );
}

// ---------------------------------------------------------------------------

function AnalysisView({ job, result, file, onPickFile }: {
  job: AnalysisJob;
  result: AnalysisResult;
  file: File | null;
  onPickFile: (file: File) => void;
}) {
  const [onlyChecked, setOnlyChecked] = useState(true);
  const [showAll, setShowAll] = useState(false);
  const [selectedId, setSelectedId] = useState<number | null>(result.tracks[0]?.summary.track_id ?? null);
  const [showTable, setShowTable] = useState(false);

  const confirms = result.events.filter((e) => e.kind === "confirmed");
  const announces = result.events.filter((e) => e.kind === "announced");
  const filtered = onlyChecked ? result.tracks.filter((t) => t.summary.checks > 0) : result.tracks;
  const visibleTracks = showAll ? filtered : filtered.slice(0, PEOPLE_ROWS);
  const selected = useMemo<TrackResult | undefined>(
    () => result.tracks.find((t) => t.summary.track_id === selectedId),
    [result, selectedId],
  );
  const required = job.params.event_confirm_ratio;

  return (
    <>
      <div className="title-row section" style={{ marginTop: 32, marginBottom: 16 }}>
        <div>
          <h2 style={{ fontSize: 18, lineHeight: "24px", fontWeight: 600 }}>{job.file_name}</h2>
          <p className="sub small">
            영상 {result.duration_sec.toFixed(1)}초 · {result.fps.toFixed(0)}fps · {result.frames}프레임 · 처리 {result.wall_sec.toFixed(1)}초 · 포즈 {job.pose_imgsz} ·
            기준 손-입 {job.params.near_threshold} · 입 근처 담배 {job.params.mouth_cig_conf_threshold} · 짝 점수 {job.params.combined_confirm_threshold} · 확정 {Math.round(required * 100)}%
          </p>
        </div>
      </div>

      <div className="kpis" aria-label="분석 결과">
        <div className="kpi">
          <div className="kpi-label">판정</div>
          <div className="kpi-value">{confirms.length > 0 ? <>확정 {confirms.length}<small>회</small></> : "이벤트 없음"}</div>
          <div className="kpi-note">{confirms.length > 0 ? `사람 ${new Set(confirms.map((c) => c.track_id)).size}명` : "안내 방송이 나가지 않는 영상"}</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">첫 확정</div>
          <div className="kpi-value">{confirms.length > 0 ? <>{confirms[0].time.toFixed(1)}<small>초</small></> : <span className="faint">—</span>}</div>
          <div className="kpi-note">영상 시작부터</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">첫 안내 방송</div>
          <div className="kpi-value">{announces.length > 0 ? <>{announces[0].time.toFixed(1)}<small>초</small></> : <span className="faint">—</span>}</div>
          <div className="kpi-note">1차 안내 기준</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">10초 판정 비율 최고</div>
          <div className="kpi-value">{Math.round(result.max_ratio * 100)}<small>%</small></div>
          <div className="kpi-note">확정 기준 {Math.round(required * 100)}% · 관측 10초 이후만</div>
        </div>
      </div>

      <section className="panel section" aria-labelledby="review-title">
        <header>
          <h2 id="review-title">영상으로 보기</h2>
          <span className="meta">브라우저 안에서만 재생 · 업로드하지 않음</span>
        </header>
        <VideoReview
          file={file}
          expectedFileName={job.file_name}
          onPickFile={onPickFile}
          result={result}
          selectedId={selectedId}
          onSelect={setSelectedId}
        />
      </section>

      <section className="panel section" aria-labelledby="people-title">
        <header>
          <h2 id="people-title">사람별 판정</h2>
          <label className="inline-field small">
            <input type="checkbox" className="switch" checked={onlyChecked} onChange={(e) => { setOnlyChecked(e.target.checked); setShowAll(false); }} />
            2차 확인을 받은 사람만 ({result.tracks.filter((t) => t.summary.checks > 0).length}/{result.tracks.length}명)
          </label>
        </header>
        <div className="table-wrap">
          <table className="wide-table">
            <thead>
              <tr>
                <th>사람</th><th className="n">보인 구간</th><th className="n">puff</th><th className="n">흡연 판정</th>
                <th className="n">담배 · 입 근처 · 연기 최고</th><th className="n">판정 비율 최고</th><th>판정 근거</th>
              </tr>
            </thead>
            <tbody>
              {visibleTracks.map(({ summary: s }) => (
                <tr
                  key={s.track_id}
                  className="clickable"
                  aria-selected={s.track_id === selectedId}
                  tabIndex={0}
                  onClick={() => setSelectedId(s.track_id)}
                  onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setSelectedId(s.track_id); } }}
                >
                  <td>사람 #{s.track_id} {s.confirmed && <span className="pill neg" style={{ marginLeft: 6 }}>확정</span>}</td>
                  <td className="n">{s.first_t.toFixed(0)}–{s.last_t.toFixed(0)}초</td>
                  <td className="n">{s.max_puffs}</td>
                  <td className="n">{s.smoking_frames}<span className="faint">/{s.checks}</span></td>
                  <td className="n">{s.max_cig.toFixed(2)} · {s.max_mouth_cig.toFixed(2)} · {s.max_smoke.toFixed(2)}</td>
                  <td className="n">{Math.round(s.max_ratio * 100)}%</td>
                  <td className="muted">{s.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {filtered.length > PEOPLE_ROWS && (
          <button type="button" className="card-link" style={{ marginTop: 12 }} onClick={() => setShowAll(!showAll)}>
            {showAll ? "접기" : `모두 보기 (${filtered.length}명)`}
          </button>
        )}
      </section>

      {selected && (
        <section className="panel section" aria-labelledby="timeline-title">
          <header>
            <h2 id="timeline-title">사람 #{selected.summary.track_id} 타임라인</h2>
            <button type="button" className="btn small" onClick={() => setShowTable(!showTable)}>
              {showTable ? "표 닫기" : "값을 표로 보기"}
            </button>
          </header>
          <p className="muted" style={{ marginBottom: 16 }}>{selected.summary.reason}</p>
          <Timeline
            samples={selected.samples}
            events={result.events.filter((e) => e.track_id === selected.summary.track_id)}
            params={job.params}
            durationSec={result.duration_sec}
          />
          {showTable && <CheckTable track={selected} />}
        </section>
      )}
    </>
  );
}

// 차트 값을 표로도 볼 수 있게 한다 (색만으로 읽기 어려운 경우 대비)
function CheckTable({ track }: { track: TrackResult }) {
  const rows = track.samples.filter((s) => s.checked);
  return (
    <div className="table-wrap" style={{ marginTop: 16, maxHeight: 360, overflowY: "auto" }}>
      <table>
        <thead>
          <tr>
            <th className="n">시각</th><th className="n">담배</th><th className="n">입 근처 담배</th><th className="n">연기</th>
            <th className="n">짝 점수</th><th>판정</th><th className="n">10초 비율</th><th>1차 게이트</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((s, i) => (
            <tr key={i}>
              <td className="n">{s.t.toFixed(1)}초</td>
              <td className="n">{s.cig?.toFixed(2)}</td>
              <td className="n">{s.mouth_cig?.toFixed(2)}</td>
              <td className="n">{s.smoke?.toFixed(2)}</td>
              <td className="n">{s.combined?.toFixed(2)}</td>
              <td>{s.smoking_frame ? "흡연" : <span className="faint">—</span>}</td>
              <td className="n">{s.ratio !== null ? `${Math.round(s.ratio * 100)}%` : "—"}</td>
              <td>{s.state} · puff {s.puff_count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
