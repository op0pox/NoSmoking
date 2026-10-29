import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchStatus, type StatusResponse } from "../api";

// 홈: 프로젝트 소개 + 원칙 + 페이지 선택 카드 + 백엔드 상태 (시연 직전 점검용)
export default function HomePage() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);

  useEffect(() => {
    fetchStatus().then(setStatus).catch((e: Error) => setStatusError(e.message));
  }, []);

  return (
    <>
      <h1 className="brand-title">🚭 교내 금연구역 흡연 계도 시스템</h1>
      <p className="secondary">
        카메라 영상에서 흡연 행동을 감지하면 안내 음성으로 계도합니다. 사람을 식별하거나 단속하지 않습니다.
      </p>
      <span className="privacy-note">🔒 영상 무저장 · 이 PC 안에서만 동작</span>

      <div className="page-cards">
        <Link to="/live" className="card page-card">
          <div className="icon">📷</div>
          <h3>실시간</h3>
          <p className="secondary">웹캠으로 바로 흡연 동작을 감지합니다.</p>
        </Link>
        <Link to="/analyze" className="card page-card">
          <div className="icon">🎞</div>
          <h3>영상 분석</h3>
          <p className="secondary">영상을 올려 언제, 왜 그렇게 판정했는지 단계별로 확인합니다.</p>
        </Link>
      </div>

      <div className="status-row">
        {statusError && <span className="error">⚠️ 백엔드에 연결할 수 없습니다 — {statusError}</span>}
        {status && (
          <>
            <span>✅ 백엔드 연결됨</span>
            <span>포즈 모델 <b>{status.pose_model}</b></span>
            <span>담배·연기 모델 <b>{status.cigarette_model}</b></span>
            <span>디바이스 <b>{status.device}</b></span>
            {!status.models_found && <span className="error">⚠️ 모델 가중치 파일이 없습니다</span>}
          </>
        )}
        {!status && !statusError && <span className="muted">백엔드 상태 확인 중…</span>}
      </div>
    </>
  );
}
