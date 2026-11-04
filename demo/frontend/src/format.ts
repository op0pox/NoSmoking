// 화면에 숫자·시각을 보여줄 때 쓰는 공통 형식

/** 오늘 날짜 YYYY-MM-DD (이 PC의 현지 시각 기준) */
export function todayLocal(daysAgo = 0): string {
  const d = new Date();
  d.setDate(d.getDate() - daysAgo);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

/** 기록 시각(ISO) → 오늘이면 "14:03:15", 아니면 "11/02 14:03" */
export function formatClock(iso: string): string {
  const date = iso.slice(0, 10);
  const time = iso.slice(11, 19);
  if (date === todayLocal()) return time;
  return `${date.slice(5).replace("-", "/")} ${time.slice(0, 5)}`;
}

/** 0~1 비율 → "60%" */
export function pct(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}
