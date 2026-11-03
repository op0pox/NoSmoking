import { useCallback, useEffect, useState } from "react";
import { fetchEvents, setEventFeedback, type EventFilters, type EventRecord, type EventSummary } from "../api";

// 판정 기록: 실시간 감지에서 이벤트가 확정될 때마다 남은 기록을 근거와 함께 보여준다.
// 현장에서 본 사람이 맞음/틀림을 누르면 현장 정밀도가 계산된다. 영상·사진은 남기지 않는다.

type Period = "today" | "7d" | "30d" | "all";
const PERIODS: { key: Period; label: string }[] = [
  { key: "today", label: "오늘" },
  { key: "7d", label: "최근 7일" },
  { key: "30d", label: "최근 30일" },
  { key: "all", label: "전체" },
];

function localDate(daysAgo: number): string {
  const d = new Date();
  d.setDate(d.getDate() - daysAgo);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

function periodFilter(period: Period): Pick<EventFilters, "dateFrom" | "dateTo"> {
  if (period === "all") return {};
  const days = period === "today" ? 0 : period === "7d" ? 6 : 29;
  return { dateFrom: localDate(days), dateTo: localDate(0) };
}

export default function RecordsPage() {
  const [period, setPeriod] = useState<Period>("7d");
  const [source, setSource] = useState("");
  const [feedback, setFeedback] = useState<"" | "correct" | "wrong" | "none">("");
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [sources, setSources] = useState<string[]>([]);
  const [summary, setSummary] = useState<EventSummary | null>(null);
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
      setEvents((list) => list.map((e) => (e.id === record.id ? { ...e, feedback: next } : e)));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <>
      <h1>판정 기록</h1>
      <p className="secondary">
        실시간 감지에서 이벤트가 확정될 때마다 판정 근거와 함께 기록됩니다. 현장에서 확인한 결과를 맞음/틀림으로 표시해 주세요.
      </p>
      <span className="privacy-note">🔒 영상·사진은 남기지 않습니다 · 사람 번호는 그 실시간 세션 안의 임시 번호입니다</span>

      {summary && <SummaryTiles summary={summary} />}

      <div className="card filter-row">
        <div className="segmented" role="group" aria-label="기간">
          {PERIODS.map((p) => (
            <button key={p.key} type="button" className={period === p.key ? "active" : ""} onClick={() => setPeriod(p.key)}>
              {p.label}
            </button>
          ))}
        </div>
        <label className="inline">
          카메라
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="">전체</option>
            {sources.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label className="inline">
          확인
          <select value={feedback} onChange={(e) => setFeedback(e.target.value as typeof feedback)}>
            <option value="">전체</option>
            <option value="none">확인 안 함</option>
            <option value="correct">맞음</option>
            <option value="wrong">틀림</option>
          </select>
        </label>
        <button type="button" className="ghost" onClick={load} disabled={loading}>{loading ? "불러오는 중…" : "새로고침"}</button>
      </div>
      {error && <p className="error">⚠️ {error}</p>}

      <div className="card table-wrap" style={{ marginTop: 16 }}>
        {events.length === 0 ? (
          <p className="muted" style={{ margin: 0 }}>
            {summary?.total ? "조건에 맞는 기록이 없습니다." : "아직 기록이 없습니다. 실시간 감지에서 이벤트가 확정되면 여기에 남습니다."}
          </p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>시각</th><th>카메라</th><th className="num">사람</th><th>방송</th><th>판정 근거</th><th>현장 확인</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e) => (
                <tr key={e.id}>
                  <td className="num">{e.created_at.slice(5, 10).replace("-", "/")} {e.created_at.slice(11)}</td>
                  <td>{e.source_label}</td>
                  <td className="num">#{e.track_id}</td>
                  <td>
                    {e.first_announced ? "📢 1차" : "—"}
                    {e.second_announced && " · 2차"}
                  </td>
                  <td><Evidence record={e} /></td>
                  <td>
                    <div className="feedback-buttons">
                      <button type="button" className={e.feedback === "correct" ? "fb correct active" : "fb correct"} onClick={() => onFeedback(e, "correct")}>
                        ✓ 맞음
                      </button>
                      <button type="button" className={e.feedback === "wrong" ? "fb wrong active" : "fb wrong"} onClick={() => onFeedback(e, "wrong")}>
                        ✗ 틀림
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}

function SummaryTiles({ summary }: { summary: EventSummary }) {
  const reviewed = summary.correct + summary.wrong;
  return (
    <div className="stat-tiles">
      <div className="card stat">
        <div className="stat-label">이벤트 확정</div>
        <div className="stat-value">{summary.total}</div>
        <div className="muted small">그중 안내 방송 {summary.announced}건</div>
      </div>
      <div className="card stat">
        <div className="stat-label">현장 정밀도</div>
        <div className="stat-value">{summary.field_precision === null ? "—" : `${Math.round(summary.field_precision * 100)}%`}</div>
        <div className="muted small">
          {reviewed > 0 ? `확인한 ${reviewed}건 중 맞음 ${summary.correct}건 · 틀림 ${summary.wrong}건` : "맞음/틀림을 표시하면 계산됩니다"}
        </div>
      </div>
      <div className="card stat">
        <div className="stat-label">확인 안 함</div>
        <div className="stat-value">{summary.unreviewed}</div>
        <div className="muted small">현장 확인이 필요한 기록</div>
      </div>
    </div>
  );
}

// 판정 근거를 한눈에: 10초 판정 비율 → 어떤 경로로 판정됐는지 → 최고 신뢰도
function Evidence({ record }: { record: EventRecord }) {
  const ev = record.evidence;
  const paths = Object.entries(ev.path_counts ?? {}).sort((a, b) => b[1] - a[1]);
  return (
    <div className="evidence">
      <div>
        <b>10초 중 {Math.round(ev.ratio * 100)}%</b>{" "}
        <span className="muted">(확인 {ev.checks}회 중 흡연 판정 {ev.smoking_frames}회)</span>
      </div>
      {paths.length > 0 && (
        <div className="secondary small">경로: {paths.map(([name, n]) => `${name} ${n}회`).join(" · ")}</div>
      )}
      <div className="muted small">
        담배 {ev.max_cig.toFixed(2)} · 입 근처 담배 {ev.max_mouth_cig.toFixed(2)} · 연기 {ev.max_smoke.toFixed(2)} · puff {ev.puff_count}회
      </div>
    </div>
  );
}
