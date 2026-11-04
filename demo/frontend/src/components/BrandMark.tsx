// 제품 마크: 금연 표시(원과 사선) 안의 담배. 브랜드 색으로 그린다.
export default function BrandMark({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 28 28" aria-hidden="true" focusable="false">
      <rect x="6" y="12.5" width="13" height="3" rx="1" fill="var(--text)" />
      <rect x="19" y="12.5" width="3" height="3" rx="0.5" fill="var(--accent)" />
      <circle cx="14" cy="14" r="11.5" fill="none" stroke="var(--accent)" strokeWidth="2.5" />
      <line x1="5.9" y1="5.9" x2="22.1" y2="22.1" stroke="var(--accent)" strokeWidth="2.5" />
    </svg>
  );
}
