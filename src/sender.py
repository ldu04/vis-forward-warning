"""
VIS 송신측: CARLA 동기 모드 + RGB/Depth + YOLO(스로틀) + UDP 브로드캐스트.
목 모드: 로컬 영상 파일로 동일 경로 검증 ( CARLA 미실행 ).

실행:
  set PYTHONPATH=CARLA PythonAPI 경로
  python sender.py
  set VIS_MOCK=1  (목 영상)
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from brake_detector import EmergencyBrakeDetector, EmergencyBrakeSignal
from config import (
    CAMERA_FOV_DEG,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CARLA_HOST,
    CARLA_PORT,
    EMERGENCY_BBOX_AREA_RATE_THRESHOLD,
    EMERGENCY_DETECTION_COOLDOWN_S,
    EMERGENCY_DTTC_DT_THRESHOLD,
    EMERGENCY_TTC_WINDOW_FRAMES,
    MOCK_VIDEO_PATH,
    SENDER_VEHICLE_ID,
    TTC_CRITICAL,
    TTC_WARNING,
    UDP_BROADCAST_ADDR,
    UDP_PORT,
    USE_MOCK_VIDEO,
    YOLO_TARGET_HZ,
)
from geometry import (
    closing_speed_ms,
    compute_ttc,
    depth_buffer_to_meters,
    longitudinal_distance_m,
    risk_tier_from_ttc,
    unproject_pixel_ray_to_world,
)
from lead_vehicle import compute_lead_vehicle_metrics
from vis_logger import (
    default_latency_csv_path,
    log_emergency_brake_detection,
    log_latency_stage,
    new_event_id,
    setup_logger,
)
from yolo_risk import DetectionRisk, YoloRiskPipeline

logger = setup_logger("sender")

# 파이프라인 단계별(프레임 캡처 → YOLO 완료 → UDP 송신) 지연시간 CSV
# 벤치마크 모드에서 --csv 로 덮어쓸 수 있도록 모듈 전역으로 둔다.
SENDER_LATENCY_CSV = default_latency_csv_path("sender")


class LatestPair:
    """센서 콜백은 최신 프레임만 유지 (큐 적체 방지)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.rgb_bgr: Optional[np.ndarray] = None
        self.depth_bgra: Optional[np.ndarray] = None

    def set_rgb(self, arr: np.ndarray) -> None:
        with self._lock:
            self.rgb_bgr = arr

    def set_depth(self, arr: np.ndarray) -> None:
        with self._lock:
            self.depth_bgra = arr

    def get_copy(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            r = None if self.rgb_bgr is None else self.rgb_bgr.copy()
            d = None if self.depth_bgra is None else self.depth_bgra.copy()
            return r, d


def carla_image_to_bgr(image: Any) -> np.ndarray:
    """CARLA ColorConverter Raw: BGRA -> BGR (OpenCV/ YOLO 입력)."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = np.reshape(arr, (image.height, image.width, 4))
    return arr[:, :, :3]


def carla_depth_to_bgra(image: Any) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    return np.reshape(arr, (image.height, image.width, 4))


def make_udp_socket() -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    return s


def build_packet(
    risk_type: str,
    ttc: float,
    loc: np.ndarray,
    extra: Optional[Dict[str, Any]] = None,
    event_id: Optional[str] = None,
) -> Dict[str, Any]:
    pkt: Dict[str, Any] = {
        "v_id": SENDER_VEHICLE_ID,
        "risk_type": risk_type,
        "ttc": round(float(ttc), 3),
        "location": {
            "x": round(float(loc[0]), 2),
            "y": round(float(loc[1]), 2),
            "z": round(float(loc[2]), 2),
        },
    }
    if extra:
        pkt["meta"] = extra
    if event_id:
        # 수신측(receiver.py)이 지연시간 CSV를 같은 event_id로 조인할 수 있도록 포함.
        pkt["event_id"] = event_id
    return pkt


def sample_depth_at_bbox_center(
    depth_meters: np.ndarray, xyxy: Tuple[float, float, float, float]
) -> float:
    x1, y1, x2, y2 = xyxy
    u = int(round((x1 + x2) / 2.0))
    v = int(round((y1 + y2) / 2.0))
    h, w = depth_meters.shape[:2]
    u = max(0, min(w - 1, u))
    v = max(0, min(h - 1, v))
    d = float(depth_meters[v, u])
    if d <= 0.05 or np.isnan(d):
        # 폴백: bbox 영역 중앙값
        patch = depth_meters[
            max(0, v - 2) : min(h, v + 3), max(0, u - 2) : min(w, u + 3)
        ]
        if patch.size:
            d = float(np.median(patch))
    return max(d, 0.1)


def process_risks_and_send(
    risks: List[DetectionRisk],
    depth_meters: Optional[np.ndarray],
    cam_matrix: np.ndarray,
    ego_loc: np.ndarray,
    ego_fwd: np.ndarray,
    ego_vel: np.ndarray,
    udp_sock: socket.socket,
    mock_fixed_depth: Optional[float] = None,
    sender_heading_deg: float = 0.0,
    event_id: Optional[str] = None,
) -> None:
    for r in risks:
        x1, y1, x2, y2 = r.xyxy
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0

        if mock_fixed_depth is not None:
            depth_ray = mock_fixed_depth
            loc = ego_loc + ego_fwd * depth_ray
        else:
            if depth_meters is None:
                continue
            depth_ray = sample_depth_at_bbox_center(depth_meters, r.xyxy)
            loc = unproject_pixel_ray_to_world(
                u,
                v,
                depth_ray,
                CAMERA_WIDTH,
                CAMERA_HEIGHT,
                CAMERA_FOV_DEG,
                cam_matrix,
            )

        dist_long = longitudinal_distance_m(ego_loc, ego_fwd, loc)
        cs = closing_speed_ms(ego_vel, ego_fwd, None)
        ttc = compute_ttc(dist_long, cs)
        tier = risk_tier_from_ttc(ttc, TTC_CRITICAL, TTC_WARNING)
        if tier == "info":
            continue

        pkt = build_packet(
            r.risk_type,
            ttc,
            loc,
            extra={
                "tier": tier,
                "score": round(float(r.score), 3),
                "sender_heading_deg": round(sender_heading_deg, 2),
                "sender_location": {
                    "x": round(float(ego_loc[0]), 2),
                    "y": round(float(ego_loc[1]), 2),
                    "z": round(float(ego_loc[2]), 2),
                },
            },
            event_id=event_id,
        )
        data = json.dumps(pkt, ensure_ascii=False).encode("utf-8")
        udp_sock.sendto(data, (UDP_BROADCAST_ADDR, UDP_PORT))
        if event_id:
            log_latency_stage(
                SENDER_LATENCY_CSV, event_id, "udp_send", "sender", extra=r.risk_type
            )
        logger.info(
            "UDP 전송 risk=%s ttc=%.2f tier=%s bytes=%d",
            r.risk_type,
            ttc,
            tier,
            len(data),
        )


def process_emergency_brake_send(
    sig: EmergencyBrakeSignal,
    xyxy: Optional[Tuple[float, float, float, float]],
    depth_meters: Optional[np.ndarray],
    cam_matrix: np.ndarray,
    ego_loc: np.ndarray,
    ego_fwd: np.ndarray,
    ego_vel: np.ndarray,
    udp_sock: socket.socket,
    mock_fixed_depth: Optional[float],
    sender_heading_deg: float,
    event_id: Optional[str] = None,
) -> None:
    """급정거(emergency_brake) UDP 전송 + 로그."""
    if mock_fixed_depth is not None:
        loc = ego_loc + ego_fwd * float(mock_fixed_depth)
    elif xyxy is not None and depth_meters is not None:
        x1, y1, x2, y2 = xyxy
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        depth_ray = sample_depth_at_bbox_center(depth_meters, xyxy)
        loc = unproject_pixel_ray_to_world(
            u,
            v,
            depth_ray,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            CAMERA_FOV_DEG,
            cam_matrix,
        )
    else:
        loc = ego_loc + ego_fwd * 12.0

    cs = closing_speed_ms(ego_vel, ego_fwd, None)
    dist_long = longitudinal_distance_m(ego_loc, ego_fwd, loc)
    ttc_use = compute_ttc(dist_long, cs)
    tier = risk_tier_from_ttc(ttc_use, TTC_CRITICAL, TTC_WARNING)
    if tier == "info":
        tier = "warning"

    pkt = build_packet(
        "emergency_brake",
        float(sig.ttc),
        loc,
        extra={
            "tier": tier,
            "score": 1.0,
            "sender_heading_deg": round(sender_heading_deg, 2),
            "sender_location": {
                "x": round(float(ego_loc[0]), 2),
                "y": round(float(ego_loc[1]), 2),
                "z": round(float(ego_loc[2]), 2),
            },
            "emergency_methods": sig.methods,
            "detection_dttc_dt": sig.dttc_dt,
            "bbox_area_rate": sig.bbox_area_rate,
            "bbox_area": sig.bbox_area,
        },
        event_id=event_id,
    )
    data = json.dumps(pkt, ensure_ascii=False).encode("utf-8")
    udp_sock.sendto(data, (UDP_BROADCAST_ADDR, UDP_PORT))
    if event_id:
        log_latency_stage(
            SENDER_LATENCY_CSV, event_id, "udp_send", "sender", extra="emergency_brake"
        )
    log_emergency_brake_detection(
        logger,
        sig.methods,
        float(sig.ttc),
        sig.dttc_dt,
        sig.bbox_area_rate,
        sig.bbox_area,
    )
    logger.info(
        "UDP 전송 risk=emergency_brake ttc=%.2f tier=%s bytes=%d",
        sig.ttc,
        tier,
        len(data),
    )


def carla_inference_tick(
    rgb: np.ndarray,
    depth_bgra: np.ndarray,
    truck: Any,
    rgb_actor: Any,
    yolo: YoloRiskPipeline,
    eb_detector: EmergencyBrakeDetector,
    udp_sock: socket.socket,
    tick_time_s: float,
) -> Optional[float]:
    """
    한 번의 YOLO 추론 + 일반 위험 UDP + 급정거 감지 UDP.
    급정거 패킷을 보냈으면 tick_time_s(시뮬/틱 시간), 아니면 None.
    """
    event_id = new_event_id()
    log_latency_stage(SENDER_LATENCY_CSV, event_id, "frame_capture", "sender")

    depth_m = depth_buffer_to_meters(depth_bgra)
    tf = rgb_actor.get_transform()
    cam_matrix = np.array(tf.get_matrix(), dtype=np.float64).reshape(4, 4)

    loc = truck.get_transform().location
    ego_loc = np.array([loc.x, loc.y, loc.z], dtype=np.float64)
    fwd = tf.get_forward_vector()
    ego_fwd = np.array([fwd.x, fwd.y, fwd.z], dtype=np.float64)
    ego_fwd /= np.linalg.norm(ego_fwd) + 1e-9
    vel = truck.get_velocity()
    ego_vel = np.array([vel.x, vel.y, vel.z], dtype=np.float64)

    risks, res_yolo = yolo.infer_frame(rgb)
    log_latency_stage(SENDER_LATENCY_CSV, event_id, "yolo_detect_done", "sender")
    if risks:
        logger.info("탐지 %d건", len(risks))
    yaw = truck.get_transform().rotation.yaw
    process_risks_and_send(
        risks,
        depth_m,
        cam_matrix,
        ego_loc,
        ego_fwd,
        ego_vel,
        udp_sock,
        mock_fixed_depth=None,
        sender_heading_deg=float(yaw),
        event_id=event_id,
    )

    lt, larea, lxy = compute_lead_vehicle_metrics(
        res_yolo,
        depth_bgra,
        cam_matrix,
        ego_loc,
        ego_fwd,
        ego_vel,
    )
    # 동기 모드 재현성: 벽시계 대신 tick_time_s 사용
    sig_eb = eb_detector.update_and_evaluate(lt, larea, tick_time_s)
    if sig_eb is not None:
        process_emergency_brake_send(
            sig_eb,
            lxy,
            depth_m,
            cam_matrix,
            ego_loc,
            ego_fwd,
            ego_vel,
            udp_sock,
            None,
            float(yaw),
            event_id=event_id,
        )
        return tick_time_s
    return None


def run_carla_sender() -> None:
    import carla

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 Hz 틱
    world.apply_settings(settings)

    try:
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
    except Exception as exc:
        logger.warning("Traffic Manager 동기화 생략: %s", exc)

    map_spawn = world.get_map().get_spawn_points()
    spawn_a = map_spawn[10] if len(map_spawn) > 10 else map_spawn[0]
    spawn_b = carla.Transform(
        spawn_a.location + carla.Location(x=-25.0, y=4.0),
        spawn_a.rotation,
    )

    def try_spawn_vehicle(filter_substr: str, tf: carla.Transform) -> Any:
        bps = bp_lib.filter(filter_substr)
        if not bps:
            bps = bp_lib.filter("vehicle.*")
        bp = bps[0]
        v = world.try_spawn_actor(bp, tf)
        if v is None:
            v = world.try_spawn_actor(bp, spawn_a)
        return v

    truck = try_spawn_vehicle("vehicle.carlamotors.*", spawn_a)
    sedan = try_spawn_vehicle("vehicle.audi.*", spawn_b)
    if truck is None or sedan is None:
        raise RuntimeError("차량 스폰 실패 — 맵·충돌 확인")

    cam_rgb_bp = bp_lib.find("sensor.camera.rgb")
    cam_rgb_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
    cam_rgb_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
    cam_rgb_bp.set_attribute("fov", str(CAMERA_FOV_DEG))

    cam_depth_bp = bp_lib.find("sensor.camera.depth")
    cam_depth_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
    cam_depth_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
    cam_depth_bp.set_attribute("fov", str(CAMERA_FOV_DEG))

    cam_tf = carla.Transform(carla.Location(x=2.3, z=1.6), carla.Rotation(pitch=-8.0))

    latest = LatestPair()

    def on_rgb(image: Any) -> None:
        latest.set_rgb(carla_image_to_bgr(image))

    def on_depth(image: Any) -> None:
        latest.set_depth(carla_depth_to_bgra(image))

    rgb_actor = world.spawn_actor(cam_rgb_bp, cam_tf, attach_to=truck)
    depth_actor = world.spawn_actor(cam_depth_bp, cam_tf, attach_to=truck)
    rgb_actor.listen(on_rgb)
    depth_actor.listen(on_depth)

    yolo = YoloRiskPipeline()
    udp_sock = make_udp_socket()
    fixed_dt = float(settings.fixed_delta_seconds or 0.05)
    infer_every_n = max(
        1, int(round(1.0 / (max(YOLO_TARGET_HZ, 0.1) * fixed_dt)))
    )
    infer_dt_s = infer_every_n * fixed_dt
    eb_detector = EmergencyBrakeDetector(
        EMERGENCY_TTC_WINDOW_FRAMES,
        EMERGENCY_DTTC_DT_THRESHOLD,
        EMERGENCY_BBOX_AREA_RATE_THRESHOLD,
        infer_dt_s,
        EMERGENCY_DETECTION_COOLDOWN_S,
    )

    logger.info("CARLA 송신 시작 sync tick=20Hz YOLO≈%.1fHz", YOLO_TARGET_HZ)

    try:
        tick_idx = 0
        while True:
            world.tick()
            tick_idx += 1
            snap = world.get_snapshot()
            tick_time_s = float(snap.timestamp.elapsed_seconds)
            rgb, depth_bgra = latest.get_copy()
            if rgb is None or depth_bgra is None:
                continue
            if tick_idx % infer_every_n != 0:
                continue

            carla_inference_tick(
                rgb,
                depth_bgra,
                truck,
                rgb_actor,
                yolo,
                eb_detector,
                udp_sock,
                tick_time_s,
            )
    except KeyboardInterrupt:
        logger.info("사용자 중단 (KeyboardInterrupt)")
    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        rgb_actor.destroy()
        depth_actor.destroy()
        truck.destroy()
        sedan.destroy()
        udp_sock.close()


def run_mock_sender() -> None:
    import cv2

    cap = cv2.VideoCapture(MOCK_VIDEO_PATH)
    if not cap.isOpened():
        logger.warning(
            "목 영상 없음: %s — 더미 프레임으로 UDP 테스트만 수행",
            MOCK_VIDEO_PATH,
        )

    yolo = YoloRiskPipeline()
    udp_sock = make_udp_socket()
    last_infer = 0.0
    period = 1.0 / max(YOLO_TARGET_HZ, 0.1)

    ego_loc = np.array([0.0, 0.0, 0.5])
    ego_fwd = np.array([1.0, 0.0, 0.0])
    ego_vel = np.array([15.0, 0.0, 0.0])
    cam_matrix = np.eye(4)

    logger.info("목 송신 시작 (깊이 고정 18m, 월드 좌표 근사)")

    frame_idx = 0
    try:
        while True:
            if cap.isOpened():
                ok, fr = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                bgr = fr
            else:
                bgr = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(
                    bgr,
                    "NO MOCK VIDEO",
                    (80, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                )
                time.sleep(0.05)

            now = time.perf_counter()
            if now - last_infer < period:
                continue
            last_infer = now

            event_id = new_event_id()
            log_latency_stage(SENDER_LATENCY_CSV, event_id, "frame_capture", "sender")

            risks, _ = yolo.infer_frame(bgr)
            log_latency_stage(SENDER_LATENCY_CSV, event_id, "yolo_detect_done", "sender")
            frame_idx += 1
            if risks:
                logger.info("[목] 탐지 %d건 frame=%d", len(risks), frame_idx)
            process_risks_and_send(
                risks,
                None,
                cam_matrix,
                ego_loc,
                ego_fwd,
                ego_vel,
                udp_sock,
                mock_fixed_depth=18.0,
                sender_heading_deg=0.0,
                event_id=event_id,
            )
    except KeyboardInterrupt:
        logger.info("목 송신 종료")
    finally:
        if cap.isOpened():
            cap.release()
        udp_sock.close()


# ---------------------------------------------------------------------------
# 지연시간 측정 전용 경로 (트랙 A: 단일 기기 구간분해 / 트랙 B: 2기기 RTT)
#
# 주의 — 이 경로는 "파이프라인 지연시간"만 재기 위한 것이다.
# 실시간 재생 속도를 따르지 않고 프레임을 최대 속도로 처리하므로,
# 재생속도가 병목이 되지 않는 대신 실제 카메라 캡처 지연은 포함되지 않는다.
# ---------------------------------------------------------------------------


def start_ack_listener(
    sock: socket.socket, csv_path: str, stop_evt: threading.Event
) -> threading.Thread:
    """
    트랙 B: 수신측이 되돌려준 ACK를 같은 소켓/같은 클럭으로 받아 기록.
    송신 시각과 ACK 수신 시각을 한 기기에서 재므로 시계 동기화가 필요 없다.
    """

    def _run() -> None:
        sock.settimeout(0.3)
        while not stop_evt.is_set():
            try:
                data, _addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                pkt = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            eid = pkt.get("ack_event_id")
            if eid:
                log_latency_stage(csv_path, eid, "ack_recv", "sender", extra="trackB")

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return th


def run_video_sender(
    input_path: str,
    target_addr: str,
    target_port: int,
    max_events: int,
    warmup: int,
    force_emit: bool,
    csv_path: str,
    enable_ack: bool,
) -> None:
    """
    영상 파일을 최대 속도로 돌려 프레임당 (capture → YOLO → UDP송신) 구간을 기록.

    force_emit=True 이면 YOLO가 위험을 못 찾아도 매 프레임 패킷을 1개 보낸다.
    ★ 이는 지연시간 표본 수를 확보하기 위한 측정 전용 모드이며,
      위험 판단(TTC/tier) 정확도와는 무관하다. 이 모드의 결과를 탐지 성능
      근거로 인용해서는 안 된다.
    """
    import cv2

    global SENDER_LATENCY_CSV
    SENDER_LATENCY_CSV = csv_path

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"입력 영상을 열 수 없음: {input_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(
        "입력 영상 %s (%dx%d, %.2f fps, %d frames) → %s:%d",
        os.path.basename(input_path),
        src_w,
        src_h,
        src_fps,
        src_frames,
        target_addr,
        target_port,
    )

    yolo = YoloRiskPipeline()
    udp_sock = make_udp_socket()
    stop_evt = threading.Event()
    if enable_ack:
        start_ack_listener(udp_sock, csv_path, stop_evt)
        logger.info("ACK 리스너 시작 (트랙 B RTT 측정)")

    ego_loc = np.array([0.0, 0.0, 0.5])
    ego_fwd = np.array([1.0, 0.0, 0.0])

    emitted = 0
    frame_idx = 0
    loops = 0
    t_start = time.perf_counter()

    try:
        while emitted < max_events + warmup:
            t_read0 = time.perf_counter()
            ok, bgr = cap.read()
            if not ok:
                # 영상 끝 → 자동 루프해서 목표 이벤트 수를 채운다.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                loops += 1
                logger.info("영상 루프 #%d (누적 이벤트 %d)", loops, emitted)
                continue
            decode_ms = (time.perf_counter() - t_read0) * 1000.0
            frame_idx += 1

            is_warmup = emitted < warmup
            phase = "warmup" if is_warmup else "measure"

            event_id = new_event_id()
            log_latency_stage(
                csv_path,
                event_id,
                "frame_capture",
                "sender",
                extra=f"{phase};decode_ms={decode_ms:.3f};frame={frame_idx}",
            )

            risks, _ = yolo.infer_frame(bgr)
            log_latency_stage(
                csv_path,
                event_id,
                "yolo_detect_done",
                "sender",
                extra=f"{phase};n_risk={len(risks)}",
            )

            if risks:
                r = risks[0]
                risk_type = r.risk_type
                score = float(r.score)
            elif force_emit:
                risk_type = "bench_synthetic"
                score = 0.0
            else:
                continue  # 위험 없음 → 패킷 없음 (이벤트로 세지 않음)

            pkt = build_packet(
                risk_type,
                3.0,
                ego_loc + ego_fwd * 18.0,
                extra={
                    "tier": "warning",
                    "score": round(score, 3),
                    "sender_heading_deg": 0.0,
                    "sender_location": {"x": 0.0, "y": 0.0, "z": 0.5},
                    "bench_phase": phase,
                    # 지연측정 전용 합성 패킷임을 수신측/분석측에서 식별 가능하게.
                    "bench_synthetic": risk_type == "bench_synthetic",
                },
                event_id=event_id,
            )
            data = json.dumps(pkt, ensure_ascii=False).encode("utf-8")
            udp_sock.sendto(data, (target_addr, target_port))
            log_latency_stage(
                csv_path,
                event_id,
                "udp_send",
                "sender",
                extra=f"{phase};{risk_type};bytes={len(data)}",
            )
            emitted += 1

            if emitted % 25 == 0:
                logger.info("이벤트 %d/%d", emitted, max_events + warmup)

        elapsed = time.perf_counter() - t_start
        logger.info(
            "측정 완료: %d 이벤트 (워밍업 %d 포함), %d 프레임, %.1fs, 루프 %d회",
            emitted,
            warmup,
            frame_idx,
            elapsed,
            loops,
        )
        if enable_ack:
            # 마지막 ACK가 도착할 여유를 준다.
            time.sleep(1.5)
    except KeyboardInterrupt:
        logger.info("사용자 중단")
    finally:
        stop_evt.set()
        cap.release()
        time.sleep(0.4)
        udp_sock.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VIS 송신측 / 지연시간 측정")
    p.add_argument(
        "--input",
        default=None,
        help="입력 영상 경로. 지정하면 지연시간 측정 모드로 동작 (파일명 하드코딩 금지)",
    )
    p.add_argument(
        "--target",
        default=UDP_BROADCAST_ADDR,
        help="UDP 목적지 IP. 루프백 우회를 위해 실제 WiFi 인터페이스 IP를 쓸 것",
    )
    p.add_argument("--port", type=int, default=UDP_PORT, help="UDP 목적지 포트")
    p.add_argument(
        "--max-events", type=int, default=200, help="워밍업 제외 목표 이벤트 수"
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="통계에서 제외할 워밍업 이벤트 수 (raw CSV에는 남김)",
    )
    p.add_argument(
        "--force-emit",
        action="store_true",
        help="위험 미검출 프레임도 패킷 송신 (측정 전용, 탐지 정확도와 무관)",
    )
    p.add_argument("--csv", default=None, help="지연시간 CSV 출력 경로")
    p.add_argument(
        "--ack", action="store_true", help="트랙 B: ACK 회신을 수신해 RTT 기록"
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.input:
        run_video_sender(
            input_path=args.input,
            target_addr=args.target,
            target_port=args.port,
            max_events=args.max_events,
            warmup=args.warmup,
            force_emit=args.force_emit,
            csv_path=args.csv or default_latency_csv_path("sender"),
            enable_ack=args.ack,
        )
    elif USE_MOCK_VIDEO:
        run_mock_sender()
    else:
        run_carla_sender()


if __name__ == "__main__":
    main()
