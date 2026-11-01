"""
assets/audio/make_placeholder_audio.py — 임시 안내 음성 mp3 생성 (macOS 전용)

팀이 직접 녹음한 안내 음성이 준비되기 전까지 쓸 임시 파일을 만든다.
macOS 기본 음성 합성(say)으로 한 번 만들어 두는 "사전 녹음 파일"이라,
실시간 TTS를 쓰지 않는다는 원칙과 어긋나지 않는다.

라즈베리파이의 재생기(mpg123)에 맞춰 mp3로 저장한다.
macOS에는 mp3 인코더가 없어서 lameenc 패키지로 변환한다.

사용 (저장소 루트에서):
    NS-venv/bin/pip install lameenc     # 처음 한 번
    NS-venv/bin/python assets/audio/make_placeholder_audio.py

문구를 바꾸려면 ANNOUNCEMENTS를 고치고 다시 실행한다.
"""

from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path

import lameenc

VOICE = "Yuna"          # macOS 한국어 음성
SPEAKING_RATE = 165     # 분당 단어 수. 기본(약 175)보다 조금 천천히
SAMPLE_RATE = 22050
MP3_BITRATE_KBPS = 64   # 음성 안내에는 충분한 음질

OUT_DIR = Path(__file__).resolve().parent

# 1차: 완곡한 안내 / 2차: 30초 뒤에도 흡연이 계속될 때, 법적 고지 포함
# ⚠️ 2차 문구의 법적 근거(과태료 부과 대상 여부)는 실제 설치 장소 기준으로 팀이 확인할 것
ANNOUNCEMENTS = {
    "first_announce.mp3": "이곳은 금연구역입니다. 흡연은 지정된 흡연 구역을 이용해 주세요. 협조해 주셔서 감사합니다.",
    "second_announce.mp3": (
        "이곳은 금연구역입니다. 금연구역에서 흡연하면 관련 법령에 따라 과태료가 부과될 수 있습니다. "
        "지금 흡연을 중단해 주세요."
    ),
}


def synthesize_wav(text: str, wav_path: Path):
    subprocess.run(
        ["say", "-v", VOICE, "-r", str(SPEAKING_RATE), "-o", str(wav_path),
         f"--data-format=LEI16@{SAMPLE_RATE}", text],
        check=True,
    )


def wav_to_mp3(wav_path: Path, mp3_path: Path):
    with wave.open(str(wav_path), "rb") as wav:
        pcm = wav.readframes(wav.getnframes())
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(MP3_BITRATE_KBPS)
        encoder.set_in_sample_rate(wav.getframerate())
        encoder.set_channels(wav.getnchannels())
        encoder.set_quality(2)  # 2 = 고음질(느림). 파일이 작아 상관없음
    mp3_path.write_bytes(encoder.encode(pcm) + encoder.flush())


def main():
    with tempfile.TemporaryDirectory() as tmp:
        for file_name, text in ANNOUNCEMENTS.items():
            wav_path = Path(tmp) / "speech.wav"
            synthesize_wav(text, wav_path)
            wav_to_mp3(wav_path, OUT_DIR / file_name)
            print(f"생성: {OUT_DIR / file_name} — {text}")


if __name__ == "__main__":
    main()
