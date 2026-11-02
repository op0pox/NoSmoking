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

const POLL_INTERVAL_MS = 500;
const CAMERA_OPTIONS = [0, 1, 2, 3];
const POSE_IMGSZ_OPTIONS = [640, 1280];

const STATE_LABELS: Record<string, string> = {
  IDLE: "정상",
  PUFF: "손이 입 근처",
  SMOKING_SUSPECTED: "흡연 의심 (puff 3회 이상)",
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
      .catch((e: Error) => setError(`백엔드에 연결할 수 없습니다 — ${e.message}`));
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

  const requiredRatio = status?.default_params.event_confirm_ratio ?? 0.6;

  return (
    <>
      <h1>실시간</h1>
      <p className="secondary">이 PC 카메라나 같은 와이파이의 IP 카메라(CCTV)를 연결해, 사람별 판정 과정을 실시간으로 보여줍니다.</p>
      <span className="privacy-note">🔒 영상은 저장되지 않습니다 · 화면 표시용 최신 프레임 한 장만 메모리에 둡니다</span>

      <div className="card toolbar" style={{ marginTop: 16 }}>
        <div className="controls">
          <label>
            카메라
            <select value={sourceValue} disabled={running} onChange={(e) => setSourceValue(e.target.value)}>
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
          <label>
            포즈 해상도
            <select value={poseImgsz} disabled={running} onChange={(e) => setPoseImgsz(Number(e.target.value))}>
              {POSE_IMGSZ_OPTIONS.map((v) => <option key={v} value={v}>{v}{v > 640 ? " (느림)" : ""}</option>)}
            </select>
          </label>
          <label className="inline">
            <input
              type="checkbox" checked={playAudio} disabled={running || !status?.audio_found}
              onChange={(e) => setPlayAudio(e.target.checked)}
            />
            🔊 안내 음성 재생
          </label>
        </div>
        {running ? (
          <button type="button" onClick={onStop}>■ 정지</button>
        ) : (
          <button type="button" onClick={onStart} disabled={starting || !status}>
            {starting ? "카메라 연결 중…" : "▶ 시작"}
          </button>
        )}
      </div>
      {status && !status.audio_found && (
        <p className="muted">안내 음성 파일(assets/audio/*.mp3)이 없어 소리 없이 진행됩니다.</p>
      )}
      <p className="muted">
        macOS에서 아이폰 연속성 카메라가 0번으로 잡히면 1번을 선택하세요. 카메라 권한은 백엔드를 실행한 터미널에 필요합니다.
      </p>
      <CameraManager
        cameras={cameras}
        disabled={running}
        onChange={(next, added) => {
          setCameras(next);
          if (added) setSourceValue(`net:${added.id}`);
          else if (sourceValue.startsWith("net:") && !next.some((c) => `net:${c.id}` === sourceValue)) setSourceValue("local:0");
        }}
      />
      {error && <p className="error">⚠️ {error}</p>}
      {running && live?.note && <p className="muted">⏳ {live.note}</p>}
      {!running && live?.error && <p className="error">⚠️ {live.error}</p>}
      {!running && !live?.error && live?.stop_reason && <p className="muted">정지됨 — {live.stop_reason}</p>}

      <div className="live-layout">
        <div className="card video-box">
          {running ? (
            <img src={`/api/live/stream?k=${streamKey}`} alt="실시간 감지 화면 (판정 결과 표시)" />
          ) : (
            <div className="video-placeholder muted">카메라가 꺼져 있습니다. [▶ 시작]을 누르면 켜집니다.</div>
          )}
          {running && (
            <div className="meta">{live?.source_label} · FPS {live?.fps.toFixed(1)} · 화면에 {live?.people.length ?? 0}명</div>
          )}
        </div>

        <div className="side">
          <div className="card">
            <h3>사람별 상태</h3>
            {(live?.people.length ?? 0) === 0 && <p className="muted">화면에 사람이 없습니다.</p>}
            {live?.people.map((p) => <PersonCard key={p.track_id} person={p} requiredRatio={requiredRatio} />)}
          </div>
          <div className="card" style={{ marginTop: 16 }}>
            <h3>이벤트</h3>
            {(live?.events.length ?? 0) === 0 && <p className="muted">아직 없습니다.</p>}
            <ul className="event-list">
              {live?.events.map((e, i) => (
                <li key={i}>
                  <span className="muted">{e.clock}</span>{" "}
                  {e.kind === "confirmed" ? `🔔 #${e.track_id} 이벤트 확정` : `📢 #${e.track_id} ${e.detail} 안내 방송`}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </>
  );
}

function PersonCard({ person: p, requiredRatio }: { person: LivePerson; requiredRatio: number }) {
  const ratio = p.ratio ?? 0;
  return (
    <div className="person">
      <div className="person-head">
        <b>#{p.track_id}</b>
        {p.confirmed ? (
          <span className="badge confirmed">🔔 이벤트 확정</span>
        ) : (
          <span className="secondary">{STATE_LABELS[p.state] ?? p.state}</span>
        )}
      </div>
      <div className="person-stats">
        <span>puff {p.puff_count}</span>
        <span>손-입 {p.dist?.toFixed(2) ?? "—"}</span>
        <span>입 근처 담배 {p.mouth_cig?.toFixed(2) ?? "—"}</span>
        <span>연기 {p.smoke?.toFixed(2) ?? "—"}</span>
      </div>
      <div className="meter" title={`최근 10초 흡연 판정 비율 ${Math.round(ratio * 100)}% (확정 기준 ${Math.round(requiredRatio * 100)}%)`}>
        <div className="fill" style={{ width: `${Math.round(ratio * 100)}%` }} />
        <div className="threshold" style={{ left: `${Math.round(requiredRatio * 100)}%` }} />
      </div>
      <div className="muted small">
        판정 비율 {Math.round(ratio * 100)}% / 기준 {Math.round(requiredRatio * 100)}%
        {p.announcer_state && ANNOUNCER_LABELS[p.announcer_state] && ` · ${ANNOUNCER_LABELS[p.announcer_state]}`}
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
    <details className="card camera-manager">
      <summary>📡 네트워크 카메라 등록 ({cameras.length}대)</summary>
      <p className="muted">
        같은 와이파이에 있는 IP 카메라·CCTV(rtsp://…)나 폰 IP 카메라 앱(http://…)을 연결할 수 있어요.
        영상이 외부 서버를 거치지 않도록 같은 공유기 안의 주소(192.168.x.x 등)만 허용합니다.
        연결이 안 되면 VLC 같은 플레이어에서 그 주소가 열리는지 먼저 확인해 주세요.
      </p>
      {cameras.length > 0 && (
        <ul className="camera-list">
          {cameras.map((c) => (
            <li key={c.id}>
              <span><b>{c.name}</b> <span className="muted">{c.url}</span></span>
              <button type="button" className="ghost" disabled={disabled} onClick={() => onDelete(c.id)}>삭제</button>
            </li>
          ))}
        </ul>
      )}
      <form className="form-grid" onSubmit={onAdd}>
        <label>
          설치 위치 이름
          <input type="text" value={name} placeholder="예: 본관 뒤 금연구역" onChange={(e) => setName(e.target.value)} />
        </label>
        <label>
          카메라 주소
          <input type="text" value={url} placeholder="rtsp://아이디:비밀번호@192.168.0.30:554/stream1" onChange={(e) => setUrl(e.target.value)} />
        </label>
        <div style={{ alignSelf: "end" }}>
          <button type="submit" disabled={saving || disabled || !name.trim() || !url.trim()}>
            {saving ? "확인 중…" : "등록"}
          </button>
        </div>
      </form>
      {error && <p className="error">⚠️ {error}</p>}
    </details>
  );
}
