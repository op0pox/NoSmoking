import { useEffect, useRef, useState } from "react";
import {
  addCamera,
  deleteCamera,
  fetchLiveState,
  fetchStatus,
  listCameras,
  startLive,
  stopLive,
  type LivePerson,
  type LiveSource,
  type LiveState,
  type NetworkCameraInfo,
  type StatusResponse,
} from "../api";
import WindowStrip from "../components/WindowStrip";

const POLL_INTERVAL_MS = 500;
const CAMERA_OPTIONS = [0, 1, 2, 3];
const POSE_IMGSZ_OPTIONS = [640, 1280];

const STATE_LABELS: Record<string, string> = {
  IDLE: "보통",
  PUFF: "손이 입 근처",
  SMOKING_SUSPECTED: "흡연 의심",
};
const ANNOUNCER_LABELS: Record<string, string> = {
  FIRST_ANNOUNCED: "1차 안내 후 30초 대기",
  COOLDOWN: "안내 쿨다운",
};

// 카메라 선택 값: "local:0" (이 PC 카메라 0번) / "net:ab12cd34" (등록한 네트워크 카메라)
function toSource(value: string): LiveSource {
  return value.startsWith("net:")
    ? { kind: "network", cameraId: value.slice(4) }
    : { kind: "local", camera: Number(value.slice(6)) };
}

// 실시간: 백엔드가 카메라(이 PC 카메라 또는 IP 카메라)를 읽고 판정한 화면(MJPEG)과 사람별 상태를 보여준다.
// 카메라는 [시작]을 눌러야 켜지고, 이 페이지에서 켠 카메라는 페이지를 떠나면 꺼진다.
export default function LivePage() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [sourceValue, setSourceValue] = useState("local:0");
  const [cameras, setCameras] = useState<NetworkCameraInfo[]>([]);
  const [poseImgsz, setPoseImgsz] = useState(640);
  const [playAudio, setPlayAudio] = useState(true);
  const [live, setLive] = useState<LiveState | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [streamKey, setStreamKey] = useState(0);   // 다시 시작할 때 스트림 주소를 바꿔 새로 연결
  const startedHere = useRef(false);               // 이 페이지에서 켠 카메라만 떠날 때 끈다

  const running = live?.running ?? false;

  useEffect(() => {
    fetchStatus()
      .then((s) => {
        setStatus(s);
        if (!s.audio_found) setPlayAudio(false);
      })
      .catch((e: Error) => setError(`백엔드에 연결할 수 없습니다. ${e.message}`));
    fetchLiveState().then(setLive).catch(() => undefined);
    listCameras().then(setCameras).catch(() => undefined);
  }, []);

  // 켜져 있는 동안 상태를 주기적으로 받는다 (요청 자체가 "화면을 보고 있다"는 신호가 된다)
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => {
      fetchLiveState().then(setLive).catch((e: Error) => setError(e.message));
    }, POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [running]);

  // 페이지를 떠나거나 탭을 닫으면 카메라를 끈다
  useEffect(() => {
    const stopIfMine = () => {
      if (startedHere.current) {
        startedHere.current = false;
        stopLive().catch(() => undefined);
      }
    };
    window.addEventListener("pagehide", stopIfMine);
    return () => {
      window.removeEventListener("pagehide", stopIfMine);
      stopIfMine();
    };
  }, []);

  async function onStart() {
    setError(null);
    setStarting(true);
    try {
      const state = await startLive(toSource(sourceValue), poseImgsz, playAudio);
      startedHere.current = true;
      setStreamKey((k) => k + 1);
      setLive(state);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setStarting(false);
    }
  }

  async function onStop() {
    startedHere.current = false;
    await stopLive();
    setLive(await fetchLiveState());
  }

  const requiredRatio = status?.default_params.event_confirm_ratio ?? 0.5;
  const people = live?.people ?? [];
  const events = live?.events ?? [];

  return (
    <>
      <div className="title-row">
        <div>
          <h1>실시간</h1>
          <p className="sub">
            {running
              ? `${live?.source_label}에서 감지하는 중입니다. 화면 표시용 최신 프레임 한 장만 메모리에 두고 저장하지 않습니다.`
              : "카메라를 골라 감지를 시작합니다. 이 PC 카메라나 같은 와이파이의 IP 카메라(CCTV)를 쓸 수 있습니다."}
          </p>
        </div>
        <div className="actions">
          {running ? (
            <button type="button" className="btn primary" onClick={onStop}>감지 정지</button>
          ) : (
            <button type="button" className="btn primary" onClick={onStart} disabled={starting || !status}>
              {starting ? "카메라 연결 중…" : "감지 시작"}
            </button>
          )}
        </div>
      </div>

      <div className="filters">
        {running ? (
          <span className="inline-field">카메라 <b style={{ color: "var(--text)", fontWeight: 500 }}>{live?.source_label}</b></span>
        ) : (
        <label className="inline-field">
          카메라
          <select className="input" value={sourceValue} onChange={(e) => setSourceValue(e.target.value)}>
            <optgroup label="이 PC 카메라">
              {CAMERA_OPTIONS.map((c) => <option key={c} value={`local:${c}`}>{c}번</option>)}
            </optgroup>
            {cameras.length > 0 && (
              <optgroup label="네트워크 카메라">
                {cameras.map((c) => <option key={c.id} value={`net:${c.id}`}>{c.name}</option>)}
              </optgroup>
            )}
          </select>
        </label>
        )}
        <label className="inline-field">
          포즈 해상도
          <select className="input" value={poseImgsz} disabled={running} onChange={(e) => setPoseImgsz(Number(e.target.value))}>
            {POSE_IMGSZ_OPTIONS.map((v) => <option key={v} value={v}>{v}{v > 640 ? " (느림)" : ""}</option>)}
          </select>
        </label>
        <label className="inline-field">
          <input
            type="checkbox" className="switch" checked={playAudio} disabled={running || !status?.audio_found}
            onChange={(e) => setPlayAudio(e.target.checked)}
          />
          안내 음성 재생
        </label>
        <span className="spacer" />
        <span className="stamp">아이폰 연속성 카메라가 0번으로 잡히면 1번을 고르세요</span>
      </div>
      {status && !status.audio_found && <p className="small faint">안내 음성 파일(assets/audio/*.mp3)이 없어 소리 없이 진행됩니다.</p>}
      {error && <p className="error">{error}</p>}
      {running && live?.note && <p className="muted">{live.note}</p>}
      {!running && live?.error && <p className="error">{live.error}</p>}
      {!running && !live?.error && live?.stop_reason && <p className="muted">정지됨: {live.stop_reason}</p>}

      <div className="grid top">
        <div className="span-8">
          <div className="panel video-panel">
            {running ? (
              <img src={`/api/live/stream?k=${streamKey}`} alt="실시간 판정 화면 (사람 박스·관절·판정 상태 표시)" />
            ) : (
              <div className="video-placeholder">카메라가 꺼져 있습니다. 감지 시작을 누르면 켜집니다.</div>
            )}
            <div className="video-meta">
              <span>{running ? live?.source_label : "꺼짐"}</span>
              <span className="num">{running ? `${live?.fps.toFixed(1)} FPS · 화면에 ${people.length}명` : "—"}</span>
            </div>
          </div>
        </div>

        <div className="span-4 stack">
          <section className="panel" aria-labelledby="people-title">
            <header>
              <h2 id="people-title">사람별 판정</h2>
              <span className="meta">확정 기준 {Math.round(requiredRatio * 100)}%</span>
            </header>
            {people.length === 0 ? (
              <p className="muted">{running ? "화면에 사람이 없습니다." : "감지를 시작하면 사람별 판정이 나타납니다."}</p>
            ) : (
              people.map((p) => <PersonRow key={p.track_id} person={p} requiredRatio={requiredRatio} />)
            )}
          </section>
          <section className="panel" aria-labelledby="events-title">
            <header>
              <h2 id="events-title">이번 세션의 이벤트</h2>
            </header>
            {events.length === 0 ? (
              <p className="muted">아직 없습니다.</p>
            ) : (
              <div className="rows">
                {events.slice(0, 6).map((e, i) => (
                  <div className="row" key={i}>
                    <span className="num muted" style={{ width: 64, flex: "none" }}>{e.clock}</span>
                    <span className="grow">
                      {e.kind === "confirmed" ? `사람 #${e.track_id} 이벤트 확정` : `사람 #${e.track_id} ${e.detail} 안내 방송`}
                    </span>
                    {e.kind === "confirmed" && <span className="pill neg">확정</span>}
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>
      </div>

      <CameraManager
        cameras={cameras}
        disabled={running}
        onChange={(next, added) => {
          setCameras(next);
          if (added) setSourceValue(`net:${added.id}`);
          else if (sourceValue.startsWith("net:") && !next.some((c) => `net:${c.id}` === sourceValue)) setSourceValue("local:0");
        }}
      />
    </>
  );
}

function PersonRow({ person: p, requiredRatio }: { person: LivePerson; requiredRatio: number }) {
  const smoking = p.recent.filter(Boolean).length;
  return (
    <div className="person">
      <div className="person-head">
        <b>사람 #{p.track_id}</b>
        {p.confirmed ? <span className="pill neg">이벤트 확정</span> : <span className="muted">{STATE_LABELS[p.state] ?? p.state}</span>}
      </div>
      <WindowStrip
        cells={p.recent}
        requiredRatio={requiredRatio}
        caption={p.recent.length > 0
          ? `최근 10초 확인 ${p.recent.length}회 중 흡연 판정 ${smoking}회 (${Math.round((p.ratio ?? 0) * 100)}%)`
          : "아직 2차 확인 전"}
      />
      <div className="person-stats">
        <span>puff {p.puff_count}회</span>
        <span>손-입 {p.dist?.toFixed(2) ?? "—"}</span>
        <span>입 근처 담배 {p.mouth_cig?.toFixed(2) ?? "—"}</span>
        <span>연기 {p.smoke?.toFixed(2) ?? "—"}</span>
        {p.announcer_state && ANNOUNCER_LABELS[p.announcer_state] && <span>{ANNOUNCER_LABELS[p.announcer_state]}</span>}
      </div>
    </div>
  );
}

// 네트워크 카메라(IP 카메라·CCTV·폰 IP 카메라 앱) 등록·삭제
function CameraManager({ cameras, disabled, onChange }: {
  cameras: NetworkCameraInfo[];
  disabled: boolean;
  onChange: (cameras: NetworkCameraInfo[], added?: NetworkCameraInfo) => void;
}) {
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onAdd(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setSaving(true);
    try {
      const added = await addCamera(name, url);
      onChange([...cameras, added], added);
      setName("");
      setUrl("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  async function onDelete(id: string) {
    await deleteCamera(id);
    onChange(cameras.filter((c) => c.id !== id));
  }

  return (
    <section className="panel section" aria-labelledby="cameras-title">
      <header>
        <h2 id="cameras-title">네트워크 카메라</h2>
        <span className="meta">{cameras.length}대 등록</span>
      </header>
      <p className="muted" style={{ marginBottom: 16 }}>
        같은 와이파이의 IP 카메라·CCTV(rtsp://)나 폰 IP 카메라 앱(http://)을 연결합니다. 영상이 외부 서버를 거치지 않도록
        같은 공유기 안의 주소(192.168.x.x 등)만 허용합니다. 연결이 안 되면 VLC 같은 플레이어에서 주소가 열리는지 먼저 확인해 주세요.
      </p>
      {cameras.length > 0 && (
        <div className="rows" style={{ marginBottom: 16 }}>
          {cameras.map((c) => (
            <div className="row" key={c.id}>
              <span className="grow">{c.name}<span className="sub">{c.url}</span></span>
              <button type="button" className="btn small" disabled={disabled} onClick={() => onDelete(c.id)}>삭제</button>
            </div>
          ))}
        </div>
      )}
      <form className="grid form" onSubmit={onAdd}>
        <label className="field span-4">
          <span>설치 위치 이름</span>
          <input className="input" type="text" value={name} placeholder="예: 본관 뒤 금연구역" onChange={(e) => setName(e.target.value)} />
        </label>
        <label className="field span-6">
          <span>카메라 주소</span>
          <input className="input" type="text" value={url} placeholder="rtsp://아이디:비밀번호@192.168.0.30:554/stream1" onChange={(e) => setUrl(e.target.value)} />
        </label>
        <div className="span-2" style={{ alignSelf: "end" }}>
          <button type="submit" className="btn" style={{ width: "100%", height: 38 }} disabled={saving || disabled || !name.trim() || !url.trim()}>
            {saving ? "확인 중…" : "등록"}
          </button>
        </div>
      </form>
      {error && <p className="error" style={{ marginTop: 12 }}>{error}</p>}
    </section>
  );
}
