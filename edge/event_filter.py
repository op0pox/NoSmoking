"""
edge/event_filter.py — 시간 필터 (10초 윈도, 60% 판정 비율)

1차(모션)+2차(객체 감지) 합산 판정을 프레임 단위로 그대로 이벤트화하면
순간적인 오탐(각도·블러로 한 프레임만 잘못 걸린 경우 등)에도 이벤트가
확정될 수 있다. 최근 EVENT_WINDOW_SEC초 동안의 프레임별 판정 중
흡연 판정 비율이 EVENT_CONFIRM_RATIO 이상일 때만 이벤트를 확정한다.

EventTimeFilter 클래스로 분리되어 있어 다른 파이프라인에서도
`from event_filter import EventTimeFilter` 로 재사용 가능하다.
사람(track_id) 1명당 인스턴스를 1개씩 만들어 쓴다.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

EVENT_WINDOW_SEC = 10.0      # 판정 비율을 계산할 슬라이딩 윈도 길이
EVENT_CONFIRM_RATIO = 0.6    # 윈도 내 흡연 판정 비율이 이 값 이상이면 이벤트 확정


class EventTimeFilter:
    """사람 1명에 대한 10초 슬라이딩 윈도 흡연 판정 비율 필터.

    update()는 "이 프레임을 흡연으로 판정했는가"(bool)를 받아 누적하고,
    최근 EVENT_WINDOW_SEC초 안에서의 판정 비율이 EVENT_CONFIRM_RATIO
    이상인지를 self.confirmed로 반영한다. 판정을 주는 쪽(파이프라인)이
    언제 update()를 호출할지 결정한다 — 예를 들어 SMOKING_SUSPECTED가
    아닌 사람은 애초에 호출하지 않는 식으로 쓸 수 있다.

    관측을 시작한 지 EVENT_WINDOW_SEC초가 지나기 전에는 confirmed가
    True가 될 수 없다 — 표본이 1~2개뿐인 상태에서 비율만 보면(예: 판정
    1개 중 1개가 True → 100%) 순간적인 오탐 하나로 즉시 확정돼버려서
    "시간 필터"의 존재 의미가 없어지기 때문이다. update() 호출이
    EVENT_WINDOW_SEC초 이상 끊기면(예: SUSPECTED에서 빠져나갔다가
    한참 후 재진입) 새로 관측을 시작한 것으로 보고 리셋한다.

    confirmed는 래칭되지 않는 실시간 상태다. 비율이 다시 떨어지면
    False로 돌아간다 — "이벤트 확정" 신호(콜러가 False→True 전환을
    감지)와 "안내 방송·쿨다운" 같은 후속 처리는 이 클래스의 책임이 아니다.
    """

    def __init__(self, track_id: int):
        self.track_id = track_id
        self._judgments: deque = deque()  # (timestamp, is_smoking_frame)
        self._observation_start: Optional[float] = None
        self._last_update_time: Optional[float] = None
        self.confirmed = False
        self.last_ratio = 0.0

    def reset(self):
        self._judgments.clear()
        self._observation_start = None
        self._last_update_time = None
        self.confirmed = False
        self.last_ratio = 0.0

    def update(self, now: float, is_smoking_frame: bool) -> bool:
        """새 프레임 판정을 추가하고 이벤트 확정 여부를 반환한다."""
        if self._last_update_time is not None and now - self._last_update_time > EVENT_WINDOW_SEC:
            # 한참 끊겼다가 재진입 — 이전 관측은 버리고 새로 시작
            self._judgments.clear()
            self._observation_start = None
        self._last_update_time = now

        if self._observation_start is None:
            self._observation_start = now

        self._judgments.append((now, is_smoking_frame))
        while self._judgments and now - self._judgments[0][0] > EVENT_WINDOW_SEC:
            self._judgments.popleft()

        positive = sum(1 for _, v in self._judgments if v)
        self.last_ratio = positive / len(self._judgments) if self._judgments else 0.0

        has_enough_history = (now - self._observation_start) >= EVENT_WINDOW_SEC
        self.confirmed = has_enough_history and self.last_ratio >= EVENT_CONFIRM_RATIO
        return self.confirmed
