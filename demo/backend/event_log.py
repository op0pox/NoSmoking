"""
demo/backend/event_log.py — 판정 기록 (SQLite)

실시간 감지에서 이벤트가 확정될 때마다 한 줄씩 남긴다. 민원이 왔을 때 "왜 그때 방송이
나갔는지" 설명하고, 현장 사람이 누른 맞음/틀림으로 현장 정밀도를 재기 위한 기록이다.

개인정보 원칙: 영상·사진·얼굴 정보는 남기지 않는다. 사람 번호는 그 실시간 세션 안에서만
쓰는 임시 번호라 누구인지 알 수 없다. 남는 것은 시각·카메라 이름·점수 같은 숫자와 글자뿐이다.
영상 분석(검증 도구)의 결과는 기록하지 않는다 — 기록은 현장 운영(실시간)만 대상으로 한다.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

FEEDBACK_VALUES = ("correct", "wrong")   # 맞음 / 틀림

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,        -- 이벤트 확정 시각 (로컬 시간, ISO 형식)
    source_label    TEXT NOT NULL,        -- 카메라(설치 위치) 이름
    session_id      TEXT NOT NULL,        -- 실시간 세션 (같은 세션 안에서만 사람 번호가 의미 있음)
    track_id        INTEGER NOT NULL,     -- 세션 안의 임시 사람 번호
    first_announced INTEGER NOT NULL DEFAULT 0,  -- 1차 안내 방송 여부
    second_announced INTEGER NOT NULL DEFAULT 0, -- 2차 안내 방송 여부
    evidence        TEXT NOT NULL,        -- 판정 근거 JSON (비율·경로별 횟수·신뢰도 등)
    settings        TEXT NOT NULL,        -- 당시 설정 JSON (모델·기준값)
    feedback        TEXT,                 -- correct / wrong / NULL(미확인)
    feedback_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_created ON events (created_at);
"""


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class EventLog:
    """판정 기록 저장소. 여러 스레드(실시간 작업 스레드, API 요청)에서 함께 쓰므로 잠금을 건다."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def record_confirmed(self, source_label: str, session_id: str, track_id: int,
                         evidence: dict, settings: dict) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (created_at, source_label, session_id, track_id, evidence, settings) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (_now_iso(), source_label, session_id, track_id,
                 json.dumps(evidence, ensure_ascii=False), json.dumps(settings, ensure_ascii=False)),
            )
            self._conn.commit()
            return cur.lastrowid

    def mark_announced(self, session_id: str, track_id: int, stage: str):
        """같은 세션·사람의 가장 최근 확정 기록에 방송 여부를 표시한다 (stage: "1차" / "2차")."""
        column = "first_announced" if stage == "1차" else "second_announced"
        with self._lock:
            self._conn.execute(
                f"UPDATE events SET {column} = 1 WHERE id = ("
                "SELECT id FROM events WHERE session_id = ? AND track_id = ? ORDER BY id DESC LIMIT 1)",
                (session_id, track_id),
            )
            self._conn.commit()

    def set_feedback(self, event_id: int, value: Optional[str]) -> bool:
        if value is not None and value not in FEEDBACK_VALUES:
            raise ValueError(f"피드백은 {FEEDBACK_VALUES} 또는 null이어야 합니다.")
        with self._lock:
            cur = self._conn.execute(
                "UPDATE events SET feedback = ?, feedback_at = ? WHERE id = ?",
                (value, _now_iso() if value else None, event_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def list(self, date_from: Optional[str] = None, date_to: Optional[str] = None,
             source_label: Optional[str] = None, feedback: Optional[str] = None, limit: int = 500) -> list:
        """date_from·date_to는 YYYY-MM-DD(포함). feedback: correct / wrong / none(미확인)."""
        where, args = [], []
        if date_from:
            where.append("created_at >= ?")
            args.append(f"{date_from}T00:00:00")
        if date_to:
            where.append("created_at <= ?")
            args.append(f"{date_to}T23:59:59")
        if source_label:
            where.append("source_label = ?")
            args.append(source_label)
        if feedback == "none":
            where.append("feedback IS NULL")
        elif feedback in FEEDBACK_VALUES:
            where.append("feedback = ?")
            args.append(feedback)
        sql = "SELECT * FROM events" + (" WHERE " + " AND ".join(where) if where else "")
        sql += " ORDER BY id DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, (*args, limit)).fetchall()
        return [self._to_dict(row) for row in rows]

    def sources(self) -> list:
        with self._lock:
            rows = self._conn.execute("SELECT DISTINCT source_label FROM events ORDER BY source_label").fetchall()
        return [row[0] for row in rows]

    def summary(self) -> dict:
        """전체 기록 수, 피드백 수, 현장 정밀도(맞음 / (맞음 + 틀림))."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS total, "
                "SUM(feedback = 'correct') AS correct, SUM(feedback = 'wrong') AS wrong, "
                "SUM(first_announced) AS announced FROM events"
            ).fetchone()
        correct, wrong = row["correct"] or 0, row["wrong"] or 0
        return {
            "total": row["total"],
            "announced": row["announced"] or 0,
            "correct": correct,
            "wrong": wrong,
            "unreviewed": row["total"] - correct - wrong,
            "field_precision": correct / (correct + wrong) if correct + wrong else None,
        }

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> dict:
        record = dict(row)
        record["evidence"] = json.loads(record["evidence"])
        record["settings"] = json.loads(record["settings"])
        record["first_announced"] = bool(record["first_announced"])
        record["second_announced"] = bool(record["second_announced"])
        return record
