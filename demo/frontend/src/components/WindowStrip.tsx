// 10초 판정 창: 이 시스템이 실제로 판단하는 방식을 그대로 그린다.
// 최근 2차 확인 결과를 한 칸씩(흡연 판정 = 브랜드 색 칸) 놓고, 확정 기준(기본 50%) 위치에 세로선을 긋는다.
// 칸 순서가 시간 순서인지(실시간) 개수만 아는지(기록)는 호출하는 쪽이 caption으로 밝힌다.

interface Props {
  cells: boolean[];        // 각 칸: 그 확인에서 "흡연"으로 판정했나
  requiredRatio: number;   // 확정 기준 비율 (0~1)
  caption?: string;
}

export default function WindowStrip({ cells, requiredRatio, caption }: Props) {
  const count = Math.max(cells.length, 1);
  const smoking = cells.filter(Boolean).length;
  const label = `최근 확인 ${cells.length}회 중 흡연 판정 ${smoking}회, 확정 기준 ${Math.round(requiredRatio * 100)}%`;
  return (
    <div>
      <div className="window-strip" role="img" aria-label={label} style={{ gridTemplateColumns: `repeat(${count}, 1fr)` }}>
        {cells.map((on, i) => <span key={i} className={on ? "on" : ""} />)}
        <i style={{ left: `${requiredRatio * 100}%` }} />
      </div>
      {caption && <div className="window-caption">{caption}</div>}
    </div>
  );
}
