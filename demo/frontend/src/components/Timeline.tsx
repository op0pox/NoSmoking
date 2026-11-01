import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipProps,
} from "recharts";
import type { AnalysisParams, PipelineEvent, TrackSample } from "../api";

// 사람 1명의 판정 과정을 시간 축을 공유하는 3단 차트로 보여준다.
//   1) 손-입 거리 (1차 게이트)   2) 담배·연기 신뢰도 (2차 확인)   3) 시간 필터 판정 비율
// 단위가 다른 값을 한 차트에 겹치지 않는다(축 하나 원칙). 마우스를 올리면 세 패널이 같은 시각을 함께 짚는다.

interface Props {
  samples: TrackSample[];
  events: PipelineEvent[];      // 이 사람의 이벤트만
  params: AnalysisParams;
  durationSec: number;
}

const SYNC_ID = "track-timeline";
const DIST_AXIS_MAX = 3; // 손-입 거리 축 상한. 손을 내리고 있으면 2~4가 흔해서, 그 이상은 위쪽 끝에 붙여 그린다
const CHART_MARGIN = { top: 8, right: 16, bottom: 4, left: 0 };

type DotShape = "circle" | "triangle" | "square";

const CONFIDENCE_SERIES: { key: keyof TrackSample; label: string; color: string; shape: DotShape }[] = [
  { key: "cig", label: "담배 (최고)", color: "var(--series-1)", shape: "circle" },
  { key: "smoke", label: "연기 (최고)", color: "var(--series-2)", shape: "triangle" },
  { key: "mouth_cig", label: "입 근처 담배", color: "var(--series-3)", shape: "square" },
];

export default function Timeline({ samples, events, params, durationSec }: Props) {
  const xDomain: [number, number] = [0, Math.max(durationSec, 1)];
  // ②·③은 2차 확인을 돌린 순간에만 값이 있다. 그 순간만 모아야 툴팁이 빈 프레임을 짚지 않는다.
  const checked = samples.filter((s) => s.checked);
  const confirmTimes = events.filter((e) => e.kind === "confirmed").map((e) => e.time);
  const announceEvents = events.filter((e) => e.kind === "announced");

  // 세 패널에 공통으로 그리는 이벤트 표시선
  const eventLines = [
    ...confirmTimes.map((t) => (
      <ReferenceLine key={`c${t}`} x={t} stroke="var(--status-critical)" strokeWidth={2}
        label={{ value: "🔔 확정", position: "insideTopRight", fill: "var(--text-secondary)", fontSize: 12 }} />
    )),
    ...announceEvents.map((e) => (
      <ReferenceLine key={`a${e.time}`} x={e.time} stroke="var(--text-muted)" strokeDasharray="2 3"
        label={{ value: `📢 ${e.detail}`, position: "insideBottomRight", fill: "var(--text-secondary)", fontSize: 12 }} />
    )),
  ];

  const xAxis = (
    <XAxis
      type="number" dataKey="t" domain={xDomain} tickFormatter={(v: number) => `${v.toFixed(0)}s`}
      stroke="var(--axis)" tick={{ fill: "var(--text-muted)", fontSize: 12 }}
    />
  );
  const grid = <CartesianGrid stroke="var(--grid)" vertical={false} />;

  return (
    <div>
      <div className="chart-panel">
        <h3>① 손-입 거리 <span className="muted">어깨너비로 정규화 · 기준선 아래로 1~6초 머물렀다 벗어나면 puff 1회</span></h3>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={samples} syncId={SYNC_ID} syncMethod="value" margin={CHART_MARGIN}>
            {grid}
            {xAxis}
            <YAxis domain={[0, DIST_AXIS_MAX]} allowDataOverflow stroke="var(--axis)" tick={{ fill: "var(--text-muted)", fontSize: 12 }} width={44} />
            <ReferenceLine y={params.near_threshold} stroke="var(--text-muted)" strokeDasharray="4 4"
              label={{ value: `기준 ${params.near_threshold}`, position: "insideTopLeft", fill: "var(--text-muted)", fontSize: 12 }} />
            {eventLines}
            <Line dataKey="dist" stroke="var(--series-1)" strokeWidth={2} dot={false} isAnimationActive={false} />
            <Tooltip content={<SampleTooltip fields={["dist", "puff_count", "state"]} />} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-panel">
        <h3>② 2차 확인 신뢰도 <span className="muted">확인한 순간({checked.length}회) 중 검출된 것만 점으로 표시</span></h3>
        <div className="legend">
          {CONFIDENCE_SERIES.map((s) => (
            <span key={s.key} className="key">
              <svg width="12" height="12"><Marker shape={s.shape} cx={6} cy={6} color={s.color} /></svg>
              {s.label}
            </span>
          ))}
        </div>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={checked} syncId={SYNC_ID} syncMethod="value" margin={CHART_MARGIN}>
            {grid}
            {xAxis}
            <YAxis domain={[0, 1]} stroke="var(--axis)" tick={{ fill: "var(--text-muted)", fontSize: 12 }} width={44} />
            <ReferenceLine y={params.mouth_cig_conf_threshold} stroke="var(--text-muted)" strokeDasharray="4 4"
              label={{ value: `입 근처 담배 기준 ${params.mouth_cig_conf_threshold}`, position: "insideTopLeft", fill: "var(--text-muted)", fontSize: 12 }} />
            {eventLines}
            {CONFIDENCE_SERIES.map((s) => (
              <Line key={s.key} dataKey={s.key} stroke="none" isAnimationActive={false} activeDot={false}
                dot={(props: DotProps) => renderDot(props, s.shape, s.color)} />
            ))}
            <Tooltip content={<SampleTooltip fields={["cig", "smoke", "mouth_cig", "combined"]} />} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-panel">
        <h3>③ 시간 필터 판정 비율 <span className="muted">최근 10초 중 "흡연" 판정 비율 · 기준 이상이면 이벤트 확정</span></h3>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={checked} syncId={SYNC_ID} syncMethod="value" margin={CHART_MARGIN}>
            {grid}
            {xAxis}
            <YAxis domain={[0, 1]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} stroke="var(--axis)"
              tick={{ fill: "var(--text-muted)", fontSize: 12 }} width={44} />
            <ReferenceLine y={params.event_confirm_ratio} stroke="var(--text-muted)" strokeDasharray="4 4"
              label={{ value: `확정 기준 ${Math.round(params.event_confirm_ratio * 100)}%`, position: "insideTopLeft", fill: "var(--text-muted)", fontSize: 12 }} />
            {eventLines}
            <Line dataKey="ratio" type="stepAfter" stroke="var(--series-1)" strokeWidth={2} dot={false} isAnimationActive={false} />
            <Tooltip content={<SampleTooltip fields={["ratio", "smoking_frame"]} />} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 점 모양: 색만으로 구분하지 않도록 시리즈마다 모양도 다르게 한다 (색약·흑백 인쇄 대비)

interface DotProps {
  cx?: number;
  cy?: number;
  value?: number | null;
  index?: number;
}

function renderDot(props: DotProps, shape: DotShape, color: string) {
  const { cx, cy, value, index } = props;
  // 0은 "검출 안 됨"이라 점을 찍지 않는다 (바닥에 점이 깔려 실제 검출이 묻히는 것 방지)
  if (value === null || value === undefined || value <= 0 || cx === undefined || cy === undefined) {
    return <g key={index} />;
  }
  return <Marker key={index} shape={shape} cx={cx} cy={cy} color={color} />;
}

function Marker({ shape, cx, cy, color }: { shape: DotShape; cx: number; cy: number; color: string }) {
  const r = 4;
  // 겹친 점이 구분되도록 배경색 테두리(링)를 두른다
  const common = { fill: color, stroke: "var(--surface-1)", strokeWidth: 1.5 };
  if (shape === "triangle") {
    return <polygon points={`${cx},${cy - r - 1} ${cx - r - 1},${cy + r} ${cx + r + 1},${cy + r}`} {...common} />;
  }
  if (shape === "square") {
    return <rect x={cx - r} y={cy - r} width={r * 2} height={r * 2} rx={1} {...common} />;
  }
  return <circle cx={cx} cy={cy} r={r} {...common} />;
}

// ---------------------------------------------------------------------------
// 툴팁: 값이 먼저(굵게), 이름이 뒤에

const FIELD_LABELS: Partial<Record<keyof TrackSample, string>> = {
  dist: "손-입 거리",
  puff_count: "puff 누적",
  state: "1차 게이트 상태",
  cig: "담배",
  smoke: "연기",
  mouth_cig: "입 근처 담배",
  combined: "짝 점수(combined)",
  ratio: "10초 판정 비율",
  smoking_frame: "이 프레임 판정",
};

function formatField(key: keyof TrackSample, value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (key === "ratio") return `${Math.round((value as number) * 100)}%`;
  if (key === "smoking_frame") return value ? "흡연" : "아님";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(2);
  return String(value);
}

function SampleTooltip({ active, payload, fields }: TooltipProps<number, string> & { fields: (keyof TrackSample)[] }) {
  if (!active || !payload || payload.length === 0) return null;
  const sample = payload[0].payload as TrackSample;
  return (
    <div className="chart-tooltip">
      <div className="label">{sample.t.toFixed(1)}초</div>
      {fields.map((key) => (
        <div key={key} className="row">
          <span className="value">{formatField(key, sample[key])}</span>
          <span className="label">{FIELD_LABELS[key]}</span>
        </div>
      ))}
    </div>
  );
}
