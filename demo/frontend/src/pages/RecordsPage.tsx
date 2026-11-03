import { useCallback, useEffect, useState } from "react";
import { fetchEvents, setEventFeedback, type EventFilters, type EventRecord, type EventSummary } from "../api";
import WindowStrip from "../components/WindowStrip";
import { formatClock, todayLocal } from "../format";

// 판정 기록: 실시간 감지에서 이벤트가 확정될 때마다 남은 기록을 근거와 함께 보여준다.
// 현장에서 본 사람이 맞음/틀림을 누르면 현장 정밀도가 계산된다. 영상·사진은 남기지 않는다.

type Period = "today" | "7d" | "30d" | "all";
const PERIODS: { key: Period; label: string }[] = [
  { key: "today", label: "오늘" },
  { key: "7d", label: "최근 7일" },
  { key: "30d", label: "최근 30일" },
  { key: "all", label: "전체" },
];

function periodFilter(period: Period): Pick<EventFilters, "dateFrom" | "dateTo"> {
  if (period === "all") return {};
  const days = period === "today" ? 0 : period === "7d" ? 6 : 29;
  return { dateFrom: todayLocal(days), dateTo: todayLocal() };
}

export default function RecordsPage() {
  const [period, setPeriod] = useState<Period>("7d");
  const [source, setSource] = useState("");
  const [feedback, setFeedback] = useState<"" | "correct" | "wrong" | "none">("");
  const [events, setEvents] = useState<EventRecord[] | null>(null);
  const [sources, setSources] = useState<string[]>([]);
  const [summary, setSummary] = useState<EventSummary | null>(null);
  const [loadedAt, setLoadedAt] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const result = await fetchEvents({
        ...periodFilter(period),
        source: source || undefined,
        feedback: feedback || undefined,
      });
      setEvents(result.events);
      setSources(result.sources);
      setSummary(result.summary);
      setLoadedAt(new Date().toTimeString().slice(0, 5));
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, [period, source, feedback]);

  useEffect(() => { load(); }, [load]);

  async function onFeedback(record: EventRecord, value: "correct" | "wrong") {
    const next = record.feedback === value ? null : value;   // 같은 버튼을 다시 누르면 취소
    try {
      setSummary(await setEventFeedback(record.id, next));
      setEvents((list) => list?.map((e) => (e.id === record.id ? { ...e, feedback: next } : e)) ?? null);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const reviewed = summary ? summary.correct + summary.wrong : 0;

  return (
    <>
      <div className="title-row">
        <div>
          <h1>판정 기록</h1>
          <p className="sub">
            실시간 감지에서 이벤트가 확정될 때마다 판정 근거와 함께 남습니다. 영상·사진은 남기지 않고, 사람 번호는 그 세션 안의 임시 번호입니다.
          </p>
        </div>
        <div className="actions">
          {loadedAt && <span className="stamp">{loadedAt} 기준</span>}
          <button type="button" className="btn" onClick={load} disabled={loading}>{loading ? "불러오는 중…" : "새로고침"}</button>
        </div>
      </div>

      <div className="kpis" aria-label="기록 요약">
        <div className="kpi">
          <div className="kpi-label">이벤트 확정</div>
          <div className="kpi-value">{summary ? <>{summary.total}<small>건</small></> : <span className="faint">—</span>}</div>
          <div className="kpi-note">{summary ? `그중 안내 방송 ${summary.announced}건 · 전체 기간` : " "}</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">현장 정밀도</div>
          <div className="kpi-value">
            {summary?.field_precision == null ? <span className="faint">—</span> : `${Math.round(summary.field_precision * 100)}%`}
          </div>
          <div className="kpi-note">
            {reviewed > 0 ? `확인한 ${reviewed}건 중 맞음 ${summary?.correct}건 · 틀림 ${summary?.wrong}건` : "맞음·틀림을 표시하면 계산됩니다"}
          </div>
        </div>
        <div className="kpi">
          <div className="kpi-label">현장 확인 전</div>
          <div className="kpi-value">{summary ? <>{summary.unreviewed}<small>건</small></> : <span className="faint">—</span>}</div>
          <div className="kpi-note">맞음·틀림 표시가 필요한 기록</div>
        </div>
      </div>

      <div className="filters section">
        <div className="segmented" role="group" aria-label="기간">
          {PERIODS.map((p) => (
            <button key={p.key} type="button" aria-pressed={period === p.key} onClick={() => setPeriod(p.key)}>{p.label}</button>
          ))}
        </div>
        <label className="inline-field">
          카메라
          <select className="input" value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="">전체</option>
            {sources.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label className="inline-field">
          현장 확인
          <select className="input" value={feedback} onChange={(e) => setFeedback(e.target.value as typeof feedback)}>
            <option value="">전체</option>
            <option value="none">확인 전</option>
            <option value="correct">맞음</option>
            <option value="wrong">틀림</option>
          </select>
        </label>
        <span className="spacer" />
        {events && <span className="stamp">{events.length}건 표시</span>}
      </div>
      {error && <p className="error">기록을 불러오지 못했습니다. {error}</p>}

      <section className="panel" aria-label="기록 목록">
        {events === null ? (
          <p className="muted">불러오는 중…</p>
        ) : events.length === 0 ? (
          <div className="empty">
            {summary?.total ? "이 조건에 맞는 기록이 없습니다." : "아직 기록이 없습니다. 실시간 감지에서 이벤트가 확정되면 여기에 남습니다."}
          </div>
        ) : (
          <div className="table-wrap">
            <table className="wide-table">
              <thead>
                <tr>
                  <th>시각</th><th>카메라</th><th className="n">사람</th><th>안내 방송</th>
                  <th>10초 판정 창</th><th>판정 근거</th><th>현장 확인</th>
                </tr>
              </thead>
              <tbody>
                {events.map((e) => <RecordRow key={e.id} record={e} onFeedback={onFeedback} />)}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}

function RecordRow({ record: e, onFeedback }: { record: EventRecord; onFeedback: (r: EventRecord, v: "correct" | "wrong") => void }) {
  const ev = e.evidence;
  const required = Number(e.settings.event_confirm_ratio ?? 0.6);
  // 기록에는 흡연 판정 "횟수"만 있어 칸의 순서는 실제 시간 순서가 아니다 (판정된 칸을 앞에 모은다)
  const cells = Array.from({ length: ev.checks }, (_, i) => i < ev.smoking_frames);
  const paths = Object.entries(ev.path_counts ?? {}).sort((a, b) => b[1] - a[1]);
  return (
    <tr>
      <td className="num">{formatClock(e.created_at)}</td>
      <td>{e.source_label}</td>
      <td className="n">#{e.track_id}</td>
      <td>{e.second_announced ? "1차 · 2차" : e.first_announced ? "1차" : <span className="faint">—</span>}</td>
      <td style={{ minWidth: 170 }}>
        <WindowStrip cells={cells} requiredRatio={required} caption={`${ev.checks}회 중 ${ev.smoking_frames}회 · ${Math.round(ev.ratio * 100)}%`} />
      </td>
      <td>
        {paths.length > 0 ? paths.map(([name, n]) => `${name} ${n}회`).join(" · ") : <span className="faint">—</span>}
        <span className="sub">담배 {ev.max_cig.toFixed(2)} · 입 근처 담배 {ev.max_mouth_cig.toFixed(2)} · 연기 {ev.max_smoke.toFixed(2)} · puff {ev.puff_count}회</span>
      </td>
      <td>
        <div style={{ display: "flex", gap: 6 }}>
          <button type="button" className="btn small fb-correct" aria-pressed={e.feedback === "correct"} onClick={() => onFeedback(e, "correct")}>맞음</button>
          <button type="button" className="btn small fb-wrong" aria-pressed={e.feedback === "wrong"} onClick={() => onFeedback(e, "wrong")}>틀림</button>
        </div>
      </td>
    </tr>
  );
}
