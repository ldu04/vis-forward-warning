"""타임스탬프가 있는 이벤트 로깅 (YOLO → UDP → UI 추적)."""
from __future__ import annotations

import csv
import logging
import os
import threading
import time
import uuid
from datetime import datetime
from typing import Any, List, Optional

from config import LOG_DIR


def setup_logger(name: str = "vis", log_file: Optional[str] = None) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    path = log_file or os.path.join(LOG_DIR, "vis_events.log")
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log_emergency_brake_detection(
    logger: logging.Logger,
    methods: List[str],
    ttc: float,
    dttc_dt: Optional[float],
    bbox_area_rate: Optional[float],
    bbox_area: Optional[float],
) -> None:
    """급정거 감지 시각·방법·TTC·bbox 변화율 로그."""
    logger.info(
        "EMERGENCY_BRAKE methods=%s ttc=%.4f dttc_dt=%s bbox_area_rate=%s bbox_area=%s",
        "+".join(methods),
        ttc,
        f"{dttc_dt:.4f}" if dttc_dt is not None else "—",
        f"{bbox_area_rate:.4f}" if bbox_area_rate is not None else "—",
        f"{bbox_area:.1f}" if bbox_area is not None else "—",
    )


def log_rear_brake_start(
    logger: logging.Logger, wall_time: float, ttc_at_start: Optional[float]
) -> None:
    """뒤차 제동 시작(인지 지연 측정용)."""
    logger.info(
        "REAR_BRAKE_START wall=%.4f ttc_at_start=%s",
        wall_time,
        f"{ttc_at_start:.4f}" if ttc_at_start is not None else "—",
    )


def log_scenario_result(logger: logging.Logger, payload: Any) -> None:
    import json

    logger.info("SCENARIO_RESULT %s", json.dumps(payload, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 파이프라인 단계별 지연시간(ms) 측정용 CSV 로깅.
# sender/receiver가 별도 프로세스(경우에 따라 별도 기기)로 실행되므로 각자
# 자기 역할의 CSV에만 기록하고, event_id로 두 CSV를 나중에 조인해서 분석한다.
# ---------------------------------------------------------------------------

LATENCY_CSV_HEADER = ["event_id", "stage", "wall_time_s", "mono_time_s", "role", "extra"]

_latency_csv_lock = threading.Lock()


def new_event_id() -> str:
    """단계별 로그를 조인하기 위한 짧은 고유 ID."""
    return uuid.uuid4().hex[:12]


def default_latency_csv_path(role: str) -> str:
    return os.path.join(LOG_DIR, f"latency_{role}.csv")


def log_latency_stage(
    csv_path: str,
    event_id: str,
    stage: str,
    role: str,
    extra: str = "",
) -> float:
    """
    한 단계(stage)의 발생 시각을 CSV에 append.
    - wall_time_s: time.time() (다른 프로세스/기기와 벽시계 비교용)
    - mono_time_s: time.perf_counter() (같은 프로세스 내 구간 차 계산용, 더 정밀)
    반환값은 이 호출에서 기록한 mono_time_s (호출부에서 구간 소요시간 계산에 재사용 가능).
    """
    wall = time.time()
    mono = time.perf_counter()
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with _latency_csv_lock:
        is_new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(LATENCY_CSV_HEADER)
            w.writerow([event_id, stage, f"{wall:.6f}", f"{mono:.6f}", role, extra])
    return mono
