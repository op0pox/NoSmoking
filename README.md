# 금연존 지킴이 (NoSmoking)

금연구역 카메라 영상에서 **흡연 동작을 실시간으로 감지**하고, 스피커로 **안내 음성**을 내보내는 엣지 AI 시스템입니다.
목적은 단속이 아니라 **계도**입니다. 영상은 저장하지 않고, 사람을 식별하지 않습니다.

팀 사다리 · 연성대학교 컴퓨터소프트웨어과 2026학년도 캡스톤 디자인

---

## 목차

1. [핵심 원칙](#1-핵심-원칙)
2. [어떻게 판정하나](#2-어떻게-판정하나)
3. [저장소 구조](#3-저장소-구조)
4. [설치](#4-설치)
5. [실행 — 웹 데모](#5-실행--웹-데모)
6. [실행 — 터미널 감지 화면](#6-실행--터미널-감지-화면)
7. [실행 — 정량 평가 도구](#7-실행--정량-평가-도구)
8. [모델 학습](#8-모델-학습)
9. [현재 성능과 한계](#9-현재-성능과-한계)
10. [데이터·개인정보 규칙](#10-데이터개인정보-규칙)
11. [문서](#11-문서)

---

## 1. 핵심 원칙

| 원칙 | 내용 |
|---|---|
| **영상·음성 무저장** | 카메라 프레임은 추론 직후 메모리에서 버립니다. 녹화·저장·외부 전송을 하지 않고, 마이크도 쓰지 않습니다. 남기는 것은 시각·판정 비율 같은 숫자뿐입니다. |
| **정밀도 우선** | 비흡연자에게 잘못 방송하는 것을 가장 나쁜 실패로 봅니다. 기준을 보수적으로 잡고, 놓치는 것은 어느 정도 허용합니다. |
| **표준 모델 구조 유지** | YOLOv8n 구조를 바꾸지 않습니다. 나중에 라즈베리파이5 + Hailo 가속기로 옮길 때 이식 위험을 줄이기 위해서입니다. |

## 2. 어떻게 판정하나

```
영상 입력 ─▶ 사람·포즈 감지 ─▶ 손-입 모션 ─┐
(웹캠·영상 파일·           (YOLOv8n-pose      (1차 확인)  ├─▶ 판정 ─▶ 10초 시간 필터 ─▶ 음성 안내
 네트워크 카메라)           + ByteTrack)  ─▶ 담배·연기 ─┘                              (1차 → 2차)
                                              (2차 확인)
```

| 단계 | 하는 일 |
|---|---|
| 사람·포즈 감지 | YOLOv8n-pose로 여러 사람의 관절 17개를 찾고, ByteTrack으로 사람별로 추적합니다. 사람이 없으면 이후 단계는 건너뜁니다. |
| 손-입 모션 (1차 확인) | 손목과 입 사이 거리를 어깨너비로 나눠 비교합니다. 이 값이 0.6 미만으로 1~6초 유지되면 한 모금(puff)으로 세고, 60초 안에 3회 이상이면 **흡연 의심**입니다. |
| 담배·연기 확인 (2차 확인) | 직접 학습한 YOLOv8n(클래스: `cigarette`, `smoke`)으로 입·손목 주변만 잘라 검출합니다. 평소에는 1초마다, 흡연 의심 상태면 매 프레임 확인합니다. |
| 판정 | 손-입 동작과 담배·연기를 짝지은 점수가 0.5 이상이거나, 입 근처에서 담배가 신뢰도 0.4 이상으로 보이면 그 순간을 "흡연"으로 판정합니다. |
| 시간 필터 | 최근 10초 중 50% 이상이 흡연 판정일 때만 **이벤트를 확정**합니다. 한순간의 잘못된 검출로는 방송하지 않습니다. |
| 음성 안내 | 확정되면 1차 완곡한 안내 → 30초 뒤에도 계속되면 2차 안내 → 60초 쿨다운. 다른 안내가 재생 중이면 새 안내는 건너뛰어 소리가 겹치지 않습니다. |

기준값은 모두 각 파일 위쪽 상수로 모여 있습니다(`edge/motion_gate.py`, `edge/pipeline.py`, `edge/event_filter.py`, `edge/voice_announcer.py`).
자세한 설명은 [기능 명세서](docs/기능명세서.md)에 있습니다.

## 3. 저장소 구조

```
edge/                감지 파이프라인 (현장 장치에서 돌아갈 코드)
  motion_detector.py   터미널 실행 진입점 (OpenCV 창)
  pipeline.py          사람·포즈 감지 → 2차 확인 → 판정 → 시간 필터 → 안내를 묶는 파이프라인
  motion_gate.py       손-입 모션 판정 (1차 확인)
  cigarette_detector.py 담배·연기 감지 (2차 확인)
  event_filter.py      10초 시간 필터
  voice_announcer.py   음성 안내 상태 머신
  overlay.py           화면 그리기
  video_analysis.py    영상 분석 (웹 데모·평가 도구 공용)
demo/                웹 데모 (같은 PC에서만 동작)
  backend/             FastAPI
  frontend/            React + TypeScript (Vite)
eval/                정량 평가 도구 (python -m eval.run), 정답 라벨 labels.csv
training/            모델 학습 노트북(Colab), 라벨링용 프레임 추출 도구
assets/audio/        안내 음성 mp3
docs/                기능 명세서
```

## 4. 설치

### 준비물

| 항목 | 버전 |
|---|---|
| Python | 3.12 (확인한 버전) |
| Node.js | 18 이상 (웹 데모 화면) |
| 운영체제 | macOS, Windows, Linux. Windows는 [Windows에서 설치](#windows에서-설치) 참고 |
| 카메라 | 노트북 웹캠, 또는 같은 공유기 안의 IP 카메라·폰 IP 카메라 앱 |
| 음성 재생 | macOS는 기본 `afplay`, Windows는 기본 PowerShell(추가 설치 없음), Linux는 `mpg123` 설치 필요 (`sudo apt install mpg123`) |

### 1) 코드 받기

```bash
git clone https://github.com/op0pox/NoSmoking.git
cd NoSmoking
```

### 2) 파이썬 환경

```bash
python3 -m venv NS-venv
source NS-venv/bin/activate          # Windows: NS-venv\Scripts\activate
python -m pip install -r requirements.txt
```

### 3) 모델 파일

모델 가중치는 저장소에 넣지 않고 [GitHub Releases](https://github.com/op0pox/NoSmoking/releases)로 공유합니다.
아래 두 파일을 **저장소 루트**(이 README가 있는 폴더)에 둡니다.

| 파일 | 내용 | 받는 곳 |
|---|---|---|
| `cigarette-v5-best.pt` | 직접 학습한 담배·연기 감지 모델 (현재 기본값) | [model-cig-v5](https://github.com/op0pox/NoSmoking/releases/tag/model-cig-v5) |
| `yolov8n-pose.pt` | 사람·포즈 모델 (Ultralytics 공개 모델) | 없으면 처음 실행할 때 자동으로 내려받습니다 |

```bash
# 예: GitHub CLI로 받기
gh release download model-cig-v5 --repo op0pox/NoSmoking --pattern "cigarette-v5-best.pt"
```

### 4) 웹 데모 화면 (처음 한 번)

```bash
cd demo/frontend
npm install
cd ../..
```

### Windows에서 설치

명령은 **PowerShell**에서 실행합니다. 위 1)~4)와 순서는 같고, 명령만 조금 다릅니다.

1. 미리 설치할 것
   - [Python 3.12](https://www.python.org/downloads/) — 설치 첫 화면에서 **"Add python.exe to PATH"를 체크**합니다
   - [Git](https://git-scm.com/download/win), [Node.js LTS](https://nodejs.org/)
2. 코드와 파이썬 환경
   ```powershell
   git clone https://github.com/op0pox/NoSmoking.git
   cd NoSmoking
   py -3.12 -m venv NS-venv
   NS-venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```
   `activate`에서 "스크립트를 실행할 수 없습니다" 오류가 나면 한 번만 아래를 실행하고 다시 시도합니다.
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```
3. **NVIDIA 그래픽카드가 있으면** (선택, 속도가 크게 빨라집니다)
   기본 설치는 CPU용입니다. [pytorch.org](https://pytorch.org/get-started/locally/)에서 Windows · Pip · CUDA를 고르면 나오는 설치 명령을 실행합니다.
   그 뒤 `python -c "import torch; print(torch.cuda.is_available())"`가 `True`면 자동으로 GPU를 씁니다.
4. 모델 파일: [model-cig-v5](https://github.com/op0pox/NoSmoking/releases/tag/model-cig-v5)에서 `cigarette-v5-best.pt`를 받아 `NoSmoking` 폴더 바로 안에 둡니다.
5. 웹 데모 화면: `cd demo\frontend` → `npm install` → `cd ..\..`
6. 실행 (터미널 두 개)
   ```powershell
   # 터미널 1 — NoSmoking 폴더에서
   NS-venv\Scripts\activate
   python -m uvicorn demo.backend.app:app --host 127.0.0.1 --port 8000

   # 터미널 2 — demo\frontend 폴더에서
   npm run dev
   ```
   브라우저에서 http://127.0.0.1:5173 을 엽니다.

- 카메라가 안 열리면 **설정 > 개인 정보 및 보안 > 카메라**에서 "데스크톱 앱이 카메라에 액세스하도록 허용"을 켜고, 카메라를 쓰는 다른 앱(Zoom 등)을 닫습니다.
- 안내 음성은 Windows 기본 PowerShell로 재생합니다(추가 설치 없음).

## 5. 실행 — 웹 데모

발표 시연과 검증에 쓰는 화면입니다. **같은 PC의 브라우저에서만** 열립니다. 터미널 두 개가 필요합니다.

```bash
# 터미널 1 — 백엔드 (저장소 루트에서)
NS-venv/bin/uvicorn demo.backend.app:app --host 127.0.0.1 --port 8000

# 터미널 2 — 화면 (demo/frontend에서)
cd demo/frontend
npm run dev
```

브라우저에서 **http://127.0.0.1:5173** 을 엽니다.

| 페이지 | 용도 |
|---|---|
| 홈 | 오늘 확정된 이벤트, 안내 방송 횟수, 현장 정밀도, 최근 판정 기록 |
| 실시간 | 웹캠·네트워크 카메라 영상에 사람 박스와 판정 상태를 겹쳐 보여주고, 확정되면 안내 음성을 재생합니다. [시작]을 눌러야 카메라가 켜집니다. |
| 영상 분석 | 영상 파일을 넣으면 사람별로 손-입 거리, 담배·연기 신뢰도, 판정 비율을 타임라인으로 보여줍니다. 기준값을 바꿔 다시 돌려볼 수 있습니다. 올린 영상은 분석이 끝나면 바로 지웁니다. |
| 판정 기록 | 실시간에서 확정된 이벤트를 판정 근거와 함께 보여줍니다. 맞음/틀림을 누르면 현장 정밀도가 계산됩니다. |

- macOS에서 처음 [시작]을 누르면 카메라 권한 팝업이 뜹니다. 백엔드를 실행한 터미널에 권한을 허용하세요.
- 아이폰 연속성 카메라가 0번으로 잡히면 실시간 페이지에서 카메라 1번을 고르세요.
- 네트워크 카메라 연결, 환경 변수, 화면 구성은 [demo/README.md](demo/README.md)에 자세히 있습니다.

## 6. 실행 — 터미널 감지 화면

웹 데모 없이 OpenCV 창으로 바로 확인할 때 씁니다. 저장소 루트에서 실행합니다.

```bash
python edge/motion_detector.py                                   # 웹캠 (카메라 0번)
python edge/motion_detector.py --camera 1                        # 다른 카메라
python edge/motion_detector.py --source 영상.mp4                  # 영상 파일
python edge/motion_detector.py --source 영상.mp4 --pose-imgsz 1280 # 사람이 작게 찍힌 영상
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--source` | `webcam` | `webcam` 또는 영상 파일 경로 |
| `--camera` | `0` | 웹캠 번호 |
| `--model` | `yolov8n-pose.pt` | 포즈 모델 |
| `--cigarette-model` | `cigarette-v5-best.pt` | 담배·연기 모델 |
| `--pose-imgsz` | `640` | 포즈 추론 해상도. 원거리 영상은 1280~1920 |
| `--first-mp3` / `--second-mp3` | `assets/audio/…` | 1차·2차 안내 음성 |

창이 선택된 상태에서 `q`는 종료, `r`은 전체 상태 초기화입니다.
화면에는 스켈레톤, 손-입 거리, puff 수, 담배·연기 박스, 최근 10초 판정 비율, 방송 상태가 표시됩니다.

## 7. 실행 — 정량 평가 도구

평가 영상 전체를 화면 없이 돌려 정밀도·재현율·오탐·지연·처리 속도를 표로 냅니다.
정답 라벨은 `eval/labels.csv`, 영상은 `test_videos/`에 둡니다. 평가 영상은 개인정보라 저장소에 포함되어 있지 않습니다.

```bash
python -m eval.run                                                       # 기본: v5, 포즈 640
python -m eval.run --weights cigarette-v4-best.pt cigarette-v5-best.pt   # 모델 비교
python -m eval.run --pose-imgsz 640 1280                                 # 포즈 해상도 비교
python -m eval.run --set pipeline.OBJECT_CHECK_INTERVAL_SEC=0.2          # 기준값 바꿔서 실행
python -m eval.run --include-reference --out eval/results/v5.md          # 참고 영상 포함, 결과 저장
```

`labels.csv` 열: `video, label(smoking/non_smoking), category, smoke_start_sec, smoke_end_sec, split(main/reference), note`

## 8. 모델 학습

- **학습**: [training/notebooks/yolo_train_colab.ipynb](training/notebooks/yolo_train_colab.ipynb)를 Google Colab(T4 GPU)에서 위에서부터 실행합니다. Roboflow에서 데이터셋을 받아 COCO 사전학습 YOLOv8n에서 시작해 학습합니다(imgsz 640, batch 16, 최대 100 에포크, 20 에포크 동안 개선이 없으면 조기 종료).
  Roboflow API 키는 노트북에 직접 넣고, **키가 든 노트북은 커밋하지 않습니다.**
- **라벨링 기준**: 담배·연기가 실제로 보이는 곳에만 박스를 그립니다. 흡연 행동 자체에는 박스를 그리지 않습니다(행동은 포즈 모델과 규칙이 판단). 연초·전자담배 기기는 `cigarette`, 연기·에어로졸은 `smoke`로 묶습니다. 물 마시기처럼 담배가 없는 사진은 박스 없는 부정 샘플로 넣습니다.
- **라벨링용 사진 추출**: 영상에서 1초에 한 장씩 사진을 뽑고, 거의 같은 사진은 건너뜁니다. 저장 폴더가 `.gitignore`에 없으면 멈춥니다.
  ```bash
  python training/extract_frames.py 영상1.mp4 영상2.mp4 --fps 1
  ```
- **학습·검증 분리**: 같은 영상에서 나온 사진이 학습과 검증에 섞이지 않게 영상(촬영 세션) 단위로 나눕니다. 학습에 쓴 영상은 평가 세트에 넣지 않습니다.

## 9. 현재 성능과 한계

v5 모델 검증 결과(두 클래스 평균): Precision 0.90 · Recall 0.75 · mAP50 0.84 · mAP50-95 0.54.
같은 평가 영상으로 판정 파이프라인 전체를 돌린 결과, 물 마시기 함정 영상에서 오탐이 없어 v5를 채택했습니다.
**평가 영상 수가 적어** 모든 수치는 참고값입니다.

알려진 한계:
- 사람이 멀리 작게 찍히거나 여러 명이 겹치는 장면에서는 흡연을 놓칠 수 있습니다. 3fps 이하 영상은 동작 조건 밖입니다.
- Windows 실행(안내 음성 재생, NVIDIA GPU 사용)은 코드는 준비했지만 실제 Windows 노트북에서의 확인은 아직입니다.
- 실제 IP 카메라·라즈베리파이에서는 아직 시험하지 않았습니다. 하드웨어 이식은 향후 과제입니다.
- 남은 개선 과제와 원인 분석은 [기능 명세서 8장](docs/기능명세서.md)에 있습니다.

## 10. 데이터·개인정보 규칙

- 영상·프레임을 저장하거나 외부로 보내는 코드를 만들지 않습니다. 디버깅용이라도 기본은 꺼 둡니다.
- 다음은 **저장소에 커밋하지 않습니다**: 데이터셋(이미지·영상·라벨), 모델 가중치(`.pt` `.onnx` `.hef`), `.env`, `runs/`, 판정 기록 DB와 카메라 등록 파일(`demo/backend/data/`).
- 팀원이 직접 촬영한 영상에는 얼굴이 담겨 있어 개인정보입니다. 팀원이 아닌 사람이 나오는 영상은 로컬 평가에만 쓰고, 학습·공유·발표 자료에는 쓰지 않습니다.
- 네트워크 카메라는 같은 공유기 안의 주소(사설 IP)만 연결할 수 있습니다.

## 11. 문서

| 문서 | 내용 |
|---|---|
| [docs/기능명세서.md](docs/기능명세서.md) | 기능별 동작, 기준값과 그 근거, 평가 결과, 개선 검토 내용 |
| [demo/README.md](demo/README.md) | 웹 데모 상세 (네트워크 카메라, 판정 기록, 환경 변수, 화면 구성) |
| [CLAUDE.md](CLAUDE.md) | 개발 규칙 요약 (설계 원칙, 기술 스택, 협업 규칙) |
