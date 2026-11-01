// 실시간 페이지는 다음 작업에서 구현한다 (기능 명세서 7.2).
export default function LivePage() {
  return (
    <>
      <h1>실시간</h1>
      <div className="card">
        <p>준비 중입니다. 지금은 터미널에서 실행할 수 있습니다.</p>
        <pre>python edge/motion_detector.py</pre>
        <p className="muted">
          이 페이지에서는 [시작] 버튼을 눌러야 카메라가 켜지고, 페이지를 떠나면 카메라가 자동으로 꺼지도록 만들 예정입니다.
        </p>
      </div>
    </>
  );
}
