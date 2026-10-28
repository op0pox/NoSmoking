# 웹 데모 (localhost 전용)

영상 분석 페이지와 (예정) 실시간 페이지. 같은 PC의 브라우저에서만 열린다.
영상은 분석하는 동안만 임시 파일로 두고 끝나면 바로 지운다. 프레임은 저장하지 않는다.

## 처음 한 번 설치

```bash
# 저장소 루트에서
NS-venv/bin/pip install fastapi "uvicorn[standard]" python-multipart
cd demo/frontend && npm install
```

## 실행 (터미널 2개)

```bash
# 1) 백엔드 — 저장소 루트에서
NS-venv/bin/uvicorn demo.backend.app:app --host 127.0.0.1 --port 8000

# 2) 프론트엔드 — demo/frontend에서
npm run dev
```

브라우저에서 http://127.0.0.1:5173 을 연다.

- 담배·연기 모델은 기본으로 저장소 루트의 `cigarette-v5-best.pt`를 쓴다.
  바꾸려면 백엔드 실행 전에 `NOSMOKING_CIGARETTE_MODEL=cigarette-v4-best.pt` 처럼 지정한다.
- 분석 결과는 백엔드 메모리에 최근 10개까지만 남는다(숫자만). 백엔드를 끄면 사라진다.

## 구성

| 경로 | 내용 |
|---|---|
| `backend/app.py` | FastAPI. `/api/status`, `/api/analyze`(업로드), `/api/analyze/{job_id}`(진행·결과) |
| `frontend/src/pages/` | 홈, 실시간(준비 중), 영상 분석 |
| `frontend/src/components/Timeline.tsx` | 사람별 3단 타임라인 (손-입 거리 / 2차 확인 신뢰도 / 시간 필터 비율) |
| `frontend/src/api.ts` | 백엔드 응답 타입 (파이썬 dataclass와 필드 이름을 맞춘다) |

분석 로직은 평가 도구(`eval/`)와 같은 `edge/video_analysis.py`를 쓴다.
