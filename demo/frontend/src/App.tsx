import { useEffect, useState } from "react";
import { NavLink, Route, Routes, useLocation } from "react-router-dom";
import BrandMark from "./components/BrandMark";
import AnalyzePage from "./pages/AnalyzePage";
import HomePage from "./pages/HomePage";
import LivePage from "./pages/LivePage";
import RecordsPage from "./pages/RecordsPage";
import { applyTheme, currentTheme, type Theme } from "./theme";

// 화면 이름 (휴대폰 상단 바에 표시)
const SCREEN_NAMES: Record<string, string> = {
  "/": "홈",
  "/live": "실시간",
  "/records": "판정 기록",
  "/analyze": "영상 분석",
};

// 틀: 왼쪽 사이드바(제품 이름 + 메뉴) + 본문. 휴대폰에서는 사이드바가 접히고 상단 바의 메뉴 버튼으로 연다.
export default function App() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [theme, setTheme] = useState<Theme>(currentTheme);
  const location = useLocation();
  useEffect(() => setMenuOpen(false), [location.pathname]);

  return (
    <div className="app">
      <aside className={`sidebar${menuOpen ? " open" : ""}`} aria-label="메뉴">
        <div className="product">
          <BrandMark />
          <div>
            <b>금연존 지킴이</b>
            <span>교내 금연구역 계도</span>
          </div>
        </div>
        <button type="button" className="btn ghost close" onClick={() => setMenuOpen(false)}>닫기</button>
        <nav aria-label="화면">
          <NavLink className="nav-item" to="/" end>홈</NavLink>
          <div className="group-label">현장 운영</div>
          <NavLink className="nav-item" to="/live">실시간</NavLink>
          <NavLink className="nav-item" to="/records">판정 기록</NavLink>
          <div className="group-label">검증</div>
          <NavLink className="nav-item" to="/analyze">영상 분석</NavLink>
        </nav>
        <div className="bottom">
          <button
            type="button" className="btn small" style={{ marginBottom: 12 }}
            onClick={() => { const next = theme === "dark" ? "light" : "dark"; applyTheme(next); setTheme(next); }}
          >
            {theme === "dark" ? "밝은 화면으로" : "어두운 화면으로"}
          </button>
          <p>영상은 저장하지 않습니다. 이 PC 안에서만 동작합니다.</p>
        </div>
      </aside>

      <div className="phone-bar">
        <b>{SCREEN_NAMES[location.pathname] ?? "금연존 지킴이"}</b>
        <button type="button" className="btn menu-toggle" aria-expanded={menuOpen} onClick={() => setMenuOpen(true)}>메뉴</button>
      </div>

      <main className="canvas">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/live" element={<LivePage />} />
          <Route path="/analyze" element={<AnalyzePage />} />
          <Route path="/records" element={<RecordsPage />} />
        </Routes>
      </main>
    </div>
  );
}
