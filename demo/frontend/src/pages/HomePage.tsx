import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  fetchEvents,
  fetchStatus,
  listCameras,
  type EventRecord,
  type EventSummary,
  type NetworkCameraInfo,
  type StatusResponse,
} from "../api";
import bannerUrl from "../assets/home-banner.webp";
import { formatClock, todayLocal } from "../format";

// 홈: 오늘 현장에서 무슨 일이 있었는지(확정·방송·현장 정밀도)와 바로 할 일(실시간 시작, 기록 확인)
export default function HomePage() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [today, setToday] = useState<EventRecord[] | null>(null);
  const [recent, setRecent] = useState<EventRecord[] | null>(null);
  const [summary, setSummary] = useState<EventSummary | null>(null);
  const [cameras, setCameras] = useState<NetworkCameraInfo[] | null>(null);

  useEffect(() => {
    fetchStatus().then(setStatus).catch((e: Error) => setStatusError(e.message));
    const day = todayLocal();
    fetchEvents({ dateFrom: day, dateTo: day }).then((r) => setToday(r.events)).catch(() => setToday([]));
    fetchEvents({}).then((r) => { setRecent(r.events.slice(0, 5)); setSummary(r.summary); }).catch(() => setRecent([]));
    listCameras().then(setCameras).catch(() => setCameras([]));
  }, []);

  const todayAnnounced = today?.filter((e) => e.first_announced).length;
  const reviewed = summary ? summary.correct + summary.wrong : 0;

  return (
    <div className="column">
      {/* 배너(AI 생성 이미지)는 브랜드 자산이라 유지. 그림 속 문구는 화면 낭독기를 위해 alt에 그대로 둔다 */}
      <img
        className="home-banner"
        src={bannerUrl}
        alt="실시간 감지 · 자동 음성 안내. AI 흡연 감지 시스템. 실시간 영상 분석으로 흡연을 자동 감지하고, 음성으로 즉시 안내합니다."
        width={2172}
        height={724}
      />

      <div className="title-row">
        <div>
          <h1>금연존 지킴이</h1>
          <p className="sub">
            금연구역의 흡연 동작을 감지하면 안내 음성으로 계도합니다. 사람을 식별하지 않고, 영상은 저장하지 않습니다.
          </p>
        </div>
        <div className="actions">
          <Link to="/live" className="btn primary">실시간 감지 시작</Link>
        </div>
      </div>

      <div className="kpis" aria-label="오늘의 현장">
        <Kpi label="오늘 이벤트 확정" value={today?.length} unit="건" note="실시간 감지 기준" />
        <Kpi label="오늘 안내 방송" value={todayAnnounced} unit="회" note="1차 안내 기준" />
        <div className="kpi">
          <div className="kpi-label">현장 정밀도</div>
          <div className="kpi-value">
            {summary?.field_precision == null ? <span className="faint">—</span> : `${Math.round(summary.field_precision * 100)}%`}
          </div>
          <div className="kpi-note">
            {reviewed > 0 ? `확인한 ${reviewed}건 중 맞음 ${summary?.correct}건` : "맞음·틀림을 표시하면 계산됩니다"}
          </div>
        </div>
        <Kpi label="등록한 네트워크 카메라" value={cameras?.length} unit="대" note="이 PC 카메라는 별도" />
      </div>

      <div className="grid top section">
        <div className="span-8 stack">
        <section className="panel" aria-labelledby="recent-title">
          <header>
            <h2 id="recent-title">최근 판정 기록</h2>
            <Link to="/records" className="card-link">기록 전체 보기</Link>
          </header>
          {recent === null ? (
            <p className="muted">불러오는 중…</p>
          ) : recent.length === 0 ? (
            <div className="empty">아직 기록이 없습니다. 실시간 감지에서 이벤트가 확정되면 여기에 남습니다.</div>
          ) : (
            <div className="rows">
              {recent.map((e) => (
                <div className="row" key={e.id}>
                  <span className="num muted" style={{ width: 92, flex: "none" }}>{formatClock(e.created_at)}</span>
                  <span className="grow">
                    {e.source_label}
                    <span className="sub">사람 #{e.track_id} · 10초 중 {Math.round(e.evidence.ratio * 100)}% 흡연 판정{e.first_announced ? " · 1차 안내" : ""}</span>
                  </span>
                  <FeedbackPill value={e.feedback} />
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="panel" aria-labelledby="verify-title">
          <header>
            <h2 id="verify-title">녹화 영상으로 검증</h2>
            <Link to="/analyze" className="card-link">영상 분석 열기</Link>
          </header>
          <p className="muted">
            촬영한 영상을 넣으면 사람별로 언제, 어느 단계에서 흡연으로 판정했는지 타임라인으로 보여줍니다. 영상은 이 PC에서만 분석하고 끝나면 지웁니다.
          </p>
        </section>
        </div>

        <section className="panel span-4" aria-labelledby="camera-title">
          <header>
            <h2 id="camera-title">카메라</h2>
            <Link to="/live" className="card-link">연결 관리</Link>
          </header>
          <div className="rows">
            <div className="row">
              <span className="grow">이 PC 카메라<span className="sub">웹캠·연속성 카메라</span></span>
            </div>
            {cameras?.map((c) => (
              <div className="row" key={c.id}>
                <span className="grow">{c.name}<span className="sub">{c.url}</span></span>
              </div>
            ))}
          </div>
          <p className="small faint" style={{ marginTop: 12 }}>
            {statusError
              ? `백엔드에 연결할 수 없습니다. ${statusError}`
              : status
                ? `백엔드 연결됨 · ${status.cigarette_model} · ${status.device}${status.models_found ? "" : " · 모델 파일 없음"}`
                : "백엔드 상태 확인 중…"}
          </p>
        </section>
      </div>
    </div>
  );
}

function Kpi({ label, value, unit, note }: { label: string; value: number | undefined; unit: string; note: string }) {
  return (
    <div className="kpi">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">
        {value === undefined ? <span className="faint">—</span> : <>{value}<small>{unit}</small></>}
      </div>
      <div className="kpi-note">{note}</div>
    </div>
  );
}

export function FeedbackPill({ value }: { value: EventRecord["feedback"] }) {
  if (value === "correct") return <span className="pill pos">맞음</span>;
  if (value === "wrong") return <span className="pill neg">틀림</span>;
  return <span className="pill">확인 전</span>;
}
