"""
edge/cigarette_detector.py — 담배/연기 객체 2차 확인 모듈

1차 게이트(SmokingMotionDetector)가 SMOKING_SUSPECTED로 판정한 사람에
한해서만 돌리는 2차 확인 모델. 커스텀 YOLOv8n(cigarette=0, smoke=1)으로
그 사람의 bbox를 크롭해 추론한다. 표준 YOLOv8n 구조 그대로 사용
(Hailo Model Zoo 호환 — 레이어 변경 금지).

CigaretteDetector 클래스로 분리되어 있어 다른 파이프라인에서도
`from cigarette_detector import CigaretteDetector` 로 재사용 가능하다.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

CIGARETTE_CLASS = "cigarette"
SMOKE_CLASS = "smoke"


class CigaretteDetector:
    """담배/연기 2차 확인 모델 래퍼.

    크롭 이미지 하나를 받아 cigarette/smoke 클래스별 최고 신뢰도를 반환한다.
    상시 실행하지 않고, 호출하는 쪽(파이프라인)에서 SMOKING_SUSPECTED인
    사람에 대해서만 detect()를 호출하는 것을 전제로 설계했다.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, crop: np.ndarray) -> dict:
        """crop에서 cigarette/smoke 각각의 최고 신뢰도(0.0~1.0)를 반환한다.
        검출이 없거나 crop이 비어 있으면 둘 다 0.0."""
        result = {CIGARETTE_CLASS: 0.0, SMOKE_CLASS: 0.0}
        if crop is None or crop.size == 0:
            return result

        results = self.model.predict(crop, verbose=False, device=self.device)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return result

        for cls_id, conf in zip(boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
            name = self.model.names[int(cls_id)]
            if name in result and conf > result[name]:
                result[name] = float(conf)
        return result
