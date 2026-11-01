import { NavLink, Route, Routes } from "react-router-dom";
import AnalyzePage from "./pages/AnalyzePage";
import HomePage from "./pages/HomePage";
import LivePage from "./pages/LivePage";

// 모든 페이지 위에 고정되는 메뉴. 홈을 거치지 않고도 페이지를 오갈 수 있다.
export default function App() {
  return (
    <>
      <header className="header">
        <NavLink to="/" className="brand">🚭 금연존 지킴이</NavLink>
        <nav>
          <NavLink to="/" end>홈</NavLink>
          <NavLink to="/live">실시간</NavLink>
          <NavLink to="/analyze">영상 분석</NavLink>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/live" element={<LivePage />} />
          <Route path="/analyze" element={<AnalyzePage />} />
        </Routes>
      </main>
    </>
  );
}
