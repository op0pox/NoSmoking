import { useEffect, useMemo, useRef, useState } from "react";
import type { AnalysisResult, TrackResult, TrackSample } from "../api";

// 분석한 영상을 그 자리에서 재생하며, 판정 결과를 영상 위에 겹쳐 보여준다.
// - 영상은 사용자가 고른 파일을 브라우저 안에서만 재생한다(URL.createObjectURL). 다시 올리거나 저장하지 않는다.
// - 백엔드에서 받는 것은 박스 좌표·점수 같은 숫자뿐이다.

interface Props {
  file: File | null;                 // 분석에 쓴 파일 (새로고침 뒤에는 없어서 다시 골라야 함)
  expectedFileName: string;
  onPickFile: (file: File) => void;
  result: AnalysisResult;
  selectedId: number | null;
  onSelect: (trackId: number) => void;
}

interface Segment { start: number; end: number; trackId: number }

const SEGMENT_MERGE_GAP_SEC = 1.5;   // 확정 상태가 이 간격 안에서 이어지면 한 구간으로 본다
const PERSON_VISIBLE_SEC = 0.25;     // 이 시간 안의 샘플만 "지금 보이는 사람"으로 그린다
const OBJECT_HOLD_SEC = 1.2;         // 2차 확인은 1초 주기라, 마지막 확인 결과를 이만큼 유지해서 그린다

export default function VideoReview({ file, expectedFileName, onPickFile, result, selectedId, onSelect }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [currentTime, setCurrentTime] = useState(0);

  // 고른 파일을 이 브라우저 안에서만 재생할 주소. 화면을 떠나면 해제한다.
  // 주소를 만들고 해제하는 일을 같은 effect 안에서 해야 한다 — 개발 모드(StrictMode)는 effect를
  // "실행 → 정리 → 다시 실행"하는데, 만들기를 useMemo에 두면 정리 때 해제된 주소가 그대로 남아 재생이 안 된다.
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!file) {
      setVideoUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const segments = useMemo(() => confirmedSegments(result.tracks), [result]);
  const smokingTicks = useMemo(
    () => result.tracks.flatMap((tr) => tr.samples.filter((s) => s.checked && s.smoking_frame).map((s) => s.t)),
    [result],
  );
  const duration = Math.max(result.duration_sec, 0.001);

  function seek(t: number) {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = Math.max(0, Math.min(t, duration));
    setCurrentTime(video.currentTime);
  }

  // 재생 중에는 매 화면 갱신마다, 멈춰 있으면 시각이 바뀔 때마다 다시 그린다
  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    let frame = 0;
    const draw = () => {
      drawOverlay(canvas, video, result, selectedId);
      setCurrentTime(video.currentTime);
      if (!video.paused && !video.ended) frame = requestAnimationFrame(draw);
    };
    const onPlay = () => { frame = requestAnimationFrame(draw); };
    video.addEventListener("play", onPlay);
    video.addEventListener("seeked", draw);
    video.addEventListener("loadeddata", draw);
    window.addEventListener("resize", draw);
    draw();
    return () => {
      cancelAnimationFrame(frame);
      video.removeEventListener("play", onPlay);
      video.removeEventListener("seeked", draw);
      video.removeEventListener("loadeddata", draw);
      window.removeEventListener("resize", draw);
    };
  }, [result, selectedId, videoUrl]);

  if (!videoUrl) {
    return (
      <div className="video-review">
        <label>
          분석한 영상 파일을 다시 고르면 결과를 영상 위에서 볼 수 있어요 (업로드되지 않고 이 브라우저에서만 재생)
          <input type="file" accept="video/*" onChange={(e) => e.target.files?.[0] && onPickFile(e.target.files[0])} />
        </label>
      </div>
    );
  }

  return (
    <div className="video-review">
      {file && file.name !== expectedFileName && (
        <p className="error">⚠️ 고른 파일({file.name})이 분석한 파일({expectedFileName})과 이름이 다릅니다. 표시가 맞지 않을 수 있어요.</p>
      )}
      <div className="video-stage">
        <video ref={videoRef} src={videoUrl} controls playsInline />
        {/* 판정 표시용 그림판. 클릭은 아래 영상(재생 버튼·진행 막대)으로 통과시킨다 */}
        <canvas ref={canvasRef} />
      </div>

      <div className="segment-legend">
        <span className="key"><span className="swatch confirmed" /> 이벤트 확정 구간</span>
        <span className="key"><span className="swatch tick" /> 흡연 판정 순간</span>
        <span className="muted">막대를 누르면 그 시각으로 이동합니다</span>
      </div>
      <div
        className="segment-bar"
        role="slider"
        aria-label="영상 시각"
        aria-valuemin={0}
        aria-valuemax={Math.round(duration)}
        aria-valuenow={Math.round(currentTime)}
        onClick={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          seek(((e.clientX - rect.left) / rect.width) * duration);
        }}
      >
        {smokingTicks.map((t, i) => (
          <span key={i} className="tick" style={{ left: `${(t / duration) * 100}%` }} />
        ))}
        {segments.map((seg, i) => (
          <span
            key={i}
            className="segment"
            title={`#${seg.trackId} 이벤트 확정 ${seg.start.toFixed(1)}~${seg.end.toFixed(1)}초`}
            style={{ left: `${(seg.start / duration) * 100}%`, width: `${Math.max(((seg.end - seg.start) / duration) * 100, 0.6)}%` }}
            onClick={(e) => { e.stopPropagation(); seek(seg.start); onSelect(seg.trackId); }}
          />
        ))}
        <span className="playhead" style={{ left: `${(currentTime / duration) * 100}%` }} />
      </div>
      {segments.length > 0 && (
        <div className="segment-chips">
          {segments.map((seg, i) => (
            <button key={i} type="button" className="ghost" onClick={() => { seek(seg.start); onSelect(seg.trackId); }}>
              🔔 {seg.start.toFixed(1)}초 · #{seg.trackId}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------

/** 사람별로 "이벤트 확정" 상태가 이어진 구간을 모은다. */
function confirmedSegments(tracks: TrackResult[]): Segment[] {
  const segments: Segment[] = [];
  for (const track of tracks) {
    let current: Segment | null = null;
    for (const s of track.samples) {
      if (!s.checked || !s.confirmed) continue;
      if (current && s.t - current.end <= SEGMENT_MERGE_GAP_SEC) {
        current.end = s.t;
      } else {
        if (current) segments.push(current);
        current = { start: s.t, end: s.t, trackId: track.summary.track_id };
      }
    }
    if (current) segments.push(current);
  }
  return segments.sort((a, b) => a.start - b.start);
}

/** t 이하에서 가장 가까운 샘플의 위치 (samples는 시각 순). */
function lastIndexAtOrBefore(samples: TrackSample[], t: number): number {
  let lo = 0;
  let hi = samples.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (samples[mid].t <= t) { found = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  return found;
}

function visibleSample(samples: TrackSample[], t: number): TrackSample | null {
  const i = lastIndexAtOrBefore(samples, t + 0.05);
  if (i < 0 || t - samples[i].t > PERSON_VISIBLE_SEC) return null;
  return samples[i];
}

function lastCheckedSample(samples: TrackSample[], t: number): TrackSample | null {
  for (let i = lastIndexAtOrBefore(samples, t + 0.05); i >= 0; i--) {
    if (t - samples[i].t > OBJECT_HOLD_SEC) return null;
    if (samples[i].checked) return samples[i];
  }
  return null;
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

/** 지금 영상 시각에 보이는 사람·담배·연기 박스를 캔버스에 그린다. */
function drawOverlay(canvas: HTMLCanvasElement, video: HTMLVideoElement, result: AnalysisResult, selectedId: number | null) {
  const ratio = window.devicePixelRatio || 1;
  const width = video.clientWidth;
  const height = video.clientHeight;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  if (!ctx || !result.width) return;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  // 영상이 비율을 유지한 채 가운데 놓이므로(object-fit: contain) 그만큼 맞춘다
  const scale = Math.min(width / result.width, height / result.height);
  const offsetX = (width - result.width * scale) / 2;
  const offsetY = (height - result.height * scale) / 2;
  const toScreen = (b: number[]) => [offsetX + b[0] * scale, offsetY + b[1] * scale, (b[2] - b[0]) * scale, (b[3] - b[1]) * scale];

  const colors = {
    confirmed: cssVar("--status-critical"),
    suspected: cssVar("--status-warning"),
    normal: "#ffffff",
    cigarette: cssVar("--series-1"),
    smoke: cssVar("--series-2"),
  };
  const t = video.currentTime;
  ctx.font = "600 13px system-ui, -apple-system, sans-serif";
  ctx.textBaseline = "bottom";

  for (const track of result.tracks) {
    const sample = visibleSample(track.samples, t);
    if (!sample?.box) continue;
    const check = lastCheckedSample(track.samples, t);
    const confirmed = Boolean(check?.confirmed);
    const color = confirmed ? colors.confirmed : sample.state === "SMOKING_SUSPECTED" ? colors.suspected : colors.normal;
    const selected = track.summary.track_id === selectedId;

    const [x, y, w, h] = toScreen(sample.box);
    ctx.lineWidth = selected ? 4 : confirmed ? 3 : 1.5;
    ctx.strokeStyle = "rgba(0,0,0,0.6)";
    ctx.strokeRect(x - 1, y - 1, w + 2, h + 2);  // 밝은 배경에서도 보이도록 어두운 테두리를 한 겹 더
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);

    const ratioText = check?.ratio !== null && check?.ratio !== undefined ? ` · 판정 ${Math.round(check.ratio * 100)}%` : "";
    drawLabel(ctx, `${confirmed ? "🔔 " : ""}#${track.summary.track_id}${ratioText}`, x, y, color);

    for (const obj of check?.objects ?? []) {
      const [ox, oy, ow, oh] = toScreen(obj.box);
      const objColor = obj.class === "cigarette" ? colors.cigarette : colors.smoke;
      ctx.lineWidth = 2;
      ctx.strokeStyle = objColor;
      ctx.strokeRect(ox, oy, ow, oh);
      drawLabel(ctx, `${obj.class === "cigarette" ? "담배" : "연기"} ${Math.round(obj.conf * 100)}%`, ox, oy, objColor);
    }
  }
}

function drawLabel(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, background: string) {
  const padding = 4;
  const textWidth = ctx.measureText(text).width;
  const top = Math.max(18, y);
  ctx.fillStyle = background;
  ctx.fillRect(x, top - 18, textWidth + padding * 2, 18);
  // 밝은 배경색(흰색·노랑) 위에는 검정, 그 외에는 흰 글자
  ctx.fillStyle = background === "#ffffff" || background.toLowerCase() === "#fab219" ? "#0b0b0b" : "#ffffff";
  ctx.fillText(text, x + padding, top - 3);
}
