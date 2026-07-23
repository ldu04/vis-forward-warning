"""
블랙박스 사본 YOLO 스캔 — 제안서 삽입용 장면 후보 탐색.

2단계 방식:
  1단계(탐색) : 초당 2프레임 샘플링으로 전체 훑기
  2단계(정밀) : 1단계 후보 구간 ±3초만 전 프레임 재스캔 + 대표 PNG 추출

찾는 것 4종:
  (a) occlusion   : truck/bus bbox 면적 15%+ & 화면 중앙하단 & 2초+ 지속 (시야차단)
  (b) reveal      : (a) 직후 대형차가 빠져 전방이 드러나는 전환 구간
  (c) brake_light : 앞차 브레이크등 점등 (붉은픽셀 8%+; 5~8%는 near 후보)
  (d) person      : 보행자 검출

EVENT 폴더는 충격·급감속 트리거 저장분이므로, YOLO와 별개로
광학흐름 기반 자기운동(ego-motion) 분석을 돌려 무슨 일이었는지 분류한다.

카드는 사용하지 않는다. 전부 로컬 사본(C:\\Users\\...\\blackbox_full) 대상.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# COCO (yolov8n.pt)
CLS_PERSON, CLS_CAR, CLS_MOTO, CLS_BUS, CLS_TRUCK = 0, 2, 3, 5, 7
BIG_VEHICLE = {CLS_BUS, CLS_TRUCK}
ANY_VEHICLE = {CLS_CAR, CLS_BUS, CLS_TRUCK}

# 판정 임계값 (요청 사양)
OCCL_AREA = 0.15          # 화면 대비 bbox 면적비
OCCL_MIN_SEC = 2.0        # 최소 지속시간
REVEAL_AREA = 0.05        # 이 아래로 떨어지면 전방이 드러난 것으로 간주
REVEAL_WINDOW_S = 3.0
BRAKE_HI = 0.08           # 브레이크등 확정 후보
BRAKE_NEAR = 0.05         # 임계 근처 후보 (2fps에서 순간이벤트 누락 방지)
CONF = 0.35
IMGSZ = 640


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """cv2.imwrite는 Windows에서 한글 경로에 실패하므로 imencode로 우회."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def red_ratio_in_box(bgr: np.ndarray, xyxy: Tuple[float, float, float, float]) -> float:
    """bbox 내 붉은 픽셀 비율 (기존 yolo_risk.red_tail_ratio_in_bbox 로직 재사용)."""
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    h, w = bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    return float(np.count_nonzero(mask)) / float(mask.size + 1e-6)


def parse_dets(res: Any, W: int, H: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return out
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    area_img = float(W * H)
    for i in range(len(cls)):
        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
        bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
        out.append(
            {
                "cls": int(cls[i]),
                "conf": float(conf[i]),
                "xyxy": (x1, y1, x2, y2),
                "area_ratio": (bw * bh) / area_img,
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
            }
        )
    return out


def is_center_bottom(d: Dict[str, Any], W: int, H: int) -> bool:
    """화면 중앙 하단부에 위치하는가 (전방 시야를 실제로 가리는 위치인지)."""
    return (0.25 * W <= d["cx"] <= 0.75 * W) and (d["cy"] >= 0.40 * H)


def stage1_scan_video(model, path: str, sample_fps: float) -> Dict[str, Any]:
    """2fps 샘플링으로 프레임별 지표 수집."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"ok": False, "reason": "열 수 없음"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0:
        cap.release()
        return {"ok": False, "reason": "프레임 없음"}

    step = max(1, int(round(fps / sample_fps)))
    samples = []
    idx = 0
    n_infer = 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            idx += step
            continue
        res = model.predict(frame, imgsz=IMGSZ, conf=CONF, verbose=False,
                            classes=[CLS_PERSON, CLS_CAR, CLS_MOTO, CLS_BUS, CLS_TRUCK])[0]
        n_infer += 1
        dets = parse_dets(res, W, H)

        big = [d for d in dets if d["cls"] in BIG_VEHICLE and is_center_bottom(d, W, H)]
        big_max = max([d["area_ratio"] for d in big], default=0.0)
        big_best = max(big, key=lambda d: d["area_ratio"], default=None)
        persons = [d for d in dets if d["cls"] == CLS_PERSON]
        # 브레이크등: 차량 bbox 중 붉은비율 최대
        red_max, red_src = 0.0, None
        for d in dets:
            if d["cls"] in ANY_VEHICLE and d["area_ratio"] >= 0.01:
                r = red_ratio_in_box(frame, d["xyxy"])
                if r > red_max:
                    red_max, red_src = r, d
        samples.append(
            {
                "t": idx / fps,
                "frame": idx,
                "big_area": big_max,
                "big_conf": (big_best or {}).get("conf", 0.0),
                "big_cls": (big_best or {}).get("cls", -1),
                "n_person": len(persons),
                "person_conf": max([d["conf"] for d in persons], default=0.0),
                "red": red_max,
                "red_conf": (red_src or {}).get("conf", 0.0),
            }
        )
        idx += step
    cap.release()
    return {"ok": True, "fps": fps, "W": W, "H": H, "total": total,
            "samples": samples, "n_infer": n_infer, "dur": total / fps}


def find_segments(samples: List[dict], key: str, thr: float, min_sec: float,
                  max_gap_s: float = 0.6) -> List[Tuple[float, float, float, float]]:
    """key >= thr 인 연속 구간을 (시작t, 끝t, 최대값, 평균값)으로 반환."""
    segs = []
    cur: List[dict] = []
    last_t = None
    for s in samples:
        if s[key] >= thr:
            if last_t is not None and (s["t"] - last_t) > max_gap_s and cur:
                segs.append(cur)
                cur = []
            cur.append(s)
            last_t = s["t"]
        else:
            if cur:
                segs.append(cur)
                cur = []
            last_t = None
    if cur:
        segs.append(cur)
    out = []
    for seg in segs:
        t0, t1 = seg[0]["t"], seg[-1]["t"]
        if (t1 - t0) >= min_sec - 1e-6:
            vals = [x[key] for x in seg]
            out.append((t0, t1, max(vals), float(np.mean(vals))))
    return out


def analyze_event_motion(path: str, probe_fps: float = 5.0) -> Dict[str, Any]:
    """
    EVENT 트리거 원인 분류용 자기운동 분석.
    광학흐름 크기로 전진속도 대용치를, 수직성분 급변으로 노면충격을 추정.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"ok": False}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 1:
        cap.release()
        return {"ok": False}
    step = max(1, int(round(fps / probe_fps)))

    mags, vys, ts = [], [], []
    prev = None
    idx = 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok or fr is None:
            idx += step
            continue
        g = cv2.cvtColor(cv2.resize(fr, (320, 180)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            roi = flow[90:, :]  # 하단 절반 = 노면, 자기운동 반영
            mag = float(np.mean(np.linalg.norm(roi, axis=2)))
            vy = float(np.mean(roi[..., 1]))
            mags.append(mag)
            vys.append(vy)
            ts.append(idx / fps)
        prev = g
        idx += step
    cap.release()
    if len(mags) < 4:
        return {"ok": False}

    m = np.array(mags)
    v = np.array(vys)
    n = len(m)
    head = float(np.mean(m[: max(1, n // 3)]))
    tail = float(np.mean(m[-max(1, n // 3):]))
    peak = float(np.max(m))
    drop = (head - tail) / head if head > 1e-6 else 0.0
    vy_spike = float(np.max(np.abs(np.diff(v)))) if n > 1 else 0.0

    # 분류 규칙
    if head < 0.35 and peak < 0.9:
        kind, why = "정지/주차중", f"전 구간 정지 수준 (평균흐름 {head:.2f})"
    elif drop >= 0.45 and head >= 0.8:
        kind, why = "급감속(위험상황 유력)", f"흐름 {head:.2f}→{tail:.2f} ({drop*100:.0f}% 감소)"
    elif vy_spike >= 0.55 and drop < 0.30:
        kind, why = "노면충격(방지턱 등)", f"수직성분 급변 {vy_spike:.2f}, 감속 미미({drop*100:.0f}%)"
    elif drop >= 0.25:
        kind, why = "판단애매", f"완만한 감속 {drop*100:.0f}%"
    else:
        kind, why = "판단애매", f"뚜렷한 감속·충격 없음 (감속 {drop*100:.0f}%, 수직 {vy_spike:.2f})"

    return {"ok": True, "kind": kind, "why": why, "head": head, "tail": tail,
            "drop": drop, "peak": peak, "vy_spike": vy_spike}


def main() -> None:
    ap = argparse.ArgumentParser(description="블랙박스 사본 YOLO 스캔")
    ap.add_argument("--src", required=True, help="스캔할 폴더 (사본)")
    ap.add_argument("--out", required=True, help="결과 출력 폴더")
    ap.add_argument("--label", required=True, help="결과 파일 접두사 (EVENT/NORMAL 등)")
    ap.add_argument("--sample-fps", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0, help="파일 수 제한 (0=전체)")
    ap.add_argument("--min-duration", type=float, default=0.0,
                    help="이 길이 미만 파일은 건너뜀 (NORMAL 60초 완전본 선별용)")
    ap.add_argument("--classify-motion", action="store_true",
                    help="EVENT 트리거 원인 분류 수행")
    ap.add_argument("--no-stage2", action="store_true", help="2단계 정밀 스캔 생략")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cand_dir = os.path.join(args.out, "candidates")
    os.makedirs(cand_dir, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

    files = sorted(f for f in os.listdir(args.src) if f.lower().endswith(".avi"))
    if args.limit:
        files = files[: args.limit]

    rows: List[dict] = []
    ev_rows: List[dict] = []
    t_start = time.perf_counter()
    total_infer = 0
    skipped = 0

    for fi, name in enumerate(files, 1):
        path = os.path.join(args.src, name)
        cap = cv2.VideoCapture(path)
        dur = 0.0
        if cap.isOpened():
            _fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / _fps
        cap.release()
        if args.min_duration and dur < args.min_duration:
            skipped += 1
            continue

        r = stage1_scan_video(model, path, args.sample_fps)
        if not r.get("ok"):
            print(f"[{fi}/{len(files)}] {name} — 스캔 실패: {r.get('reason')}", flush=True)
            continue
        total_infer += r["n_infer"]
        S = r["samples"]

        # (a) 시야차단
        occl = find_segments(S, "big_area", OCCL_AREA, OCCL_MIN_SEC)
        for (t0, t1, mx, mean) in occl:
            rows.append({"file": name, "type": "a_occlusion", "t_start": round(t0, 2),
                         "t_end": round(t1, 2), "dur_s": round(t1 - t0, 2),
                         "area_ratio_max": round(mx, 4), "area_ratio_mean": round(mean, 4),
                         "conf": round(max([s["big_conf"] for s in S
                                            if t0 <= s["t"] <= t1] or [0]), 3),
                         "stage": 1})
            # (b) 직후 전방 드러남
            after = [s for s in S if t1 < s["t"] <= t1 + REVEAL_WINDOW_S]
            if after and min(s["big_area"] for s in after) <= REVEAL_AREA:
                rv = next(s for s in after if s["big_area"] <= REVEAL_AREA)
                rows.append({"file": name, "type": "b_reveal", "t_start": round(t1, 2),
                             "t_end": round(rv["t"], 2), "dur_s": round(rv["t"] - t1, 2),
                             "area_ratio_max": round(mx, 4),
                             "area_ratio_mean": round(rv["big_area"], 4),
                             "conf": 0.0, "stage": 1})

        # (c) 브레이크등 (확정 / 임계근처)
        for (t0, t1, mx, mean) in find_segments(S, "red", BRAKE_HI, 0.0):
            rows.append({"file": name, "type": "c_brake", "t_start": round(t0, 2),
                         "t_end": round(t1, 2), "dur_s": round(t1 - t0, 2),
                         "area_ratio_max": round(mx, 4), "area_ratio_mean": round(mean, 4),
                         "conf": round(max([s["red_conf"] for s in S
                                            if t0 <= s["t"] <= t1] or [0]), 3),
                         "stage": 1})
        for (t0, t1, mx, mean) in find_segments(S, "red", BRAKE_NEAR, 0.0):
            if mx < BRAKE_HI:
                rows.append({"file": name, "type": "c_brake_near", "t_start": round(t0, 2),
                             "t_end": round(t1, 2), "dur_s": round(t1 - t0, 2),
                             "area_ratio_max": round(mx, 4),
                             "area_ratio_mean": round(mean, 4),
                             "conf": 0.0, "stage": 1})

        # (d) 보행자
        for (t0, t1, mx, mean) in find_segments(S, "n_person", 1, 0.0):
            rows.append({"file": name, "type": "d_person", "t_start": round(t0, 2),
                         "t_end": round(t1, 2), "dur_s": round(t1 - t0, 2),
                         "area_ratio_max": round(mx, 4), "area_ratio_mean": round(mean, 4),
                         "conf": round(max([s["person_conf"] for s in S
                                            if t0 <= s["t"] <= t1] or [0]), 3),
                         "stage": 1})

        if args.classify_motion:
            mo = analyze_event_motion(path)
            ev_rows.append({"file": name, "dur_s": round(dur, 1),
                            "kind": mo.get("kind", "분석실패"),
                            "why": mo.get("why", ""),
                            "flow_head": round(mo.get("head", 0), 3),
                            "flow_tail": round(mo.get("tail", 0), 3),
                            "drop_pct": round(mo.get("drop", 0) * 100, 1),
                            "vy_spike": round(mo.get("vy_spike", 0), 3),
                            "has_occlusion": any(
                                x["file"] == name and x["type"] == "a_occlusion" for x in rows)})

        if fi % 10 == 0 or fi == len(files):
            el = time.perf_counter() - t_start
            print(f"[{fi}/{len(files)}] 후보 {len(rows)}건, 추론 {total_infer}프레임, "
                  f"{el:.0f}s 경과", flush=True)

    # ---------- 2단계: 후보 구간 ±3초 정밀 재스캔 + 대표 프레임 ----------
    if not args.no_stage2:
        print("\n[2단계] 후보 구간 정밀 스캔 + 대표 프레임 추출", flush=True)
        # 유형별 상위 후보만 정밀 처리 (면적비/신뢰도 기준)
        prio = {"a_occlusion": 0, "b_reveal": 1, "c_brake": 2, "d_person": 3,
                "c_brake_near": 4}
        ranked = sorted(rows, key=lambda r: (prio.get(r["type"], 9),
                                             -r["area_ratio_max"], -r["dur_s"]))
        picked = ranked[:60]
        for k, r in enumerate(picked, 1):
            path = os.path.join(args.src, r["file"])
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            # 구간 중앙 프레임을 대표로
            t_mid = (r["t_start"] + r["t_end"]) / 2.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_mid * fps))
            ok, fr = cap.read()
            cap.release()
            if not ok or fr is None:
                continue
            stem = os.path.splitext(r["file"])[0]
            png = f"{k:03d}_{r['type']}_{stem}_t{t_mid:.1f}s.png"
            if imwrite_unicode(os.path.join(cand_dir, png), fr):
                r["frame_png"] = png
                r["stage"] = 2

    # ---------- 저장 ----------
    csv_path = os.path.join(args.out, f"{args.label}_candidates.csv")
    cols = ["file", "type", "t_start", "t_end", "dur_s", "area_ratio_max",
            "area_ratio_mean", "conf", "stage", "frame_png"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if ev_rows:
        ev_path = os.path.join(args.out, f"{args.label}_event_classification.csv")
        with open(ev_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(ev_rows[0].keys()))
            w.writeheader()
            w.writerows(ev_rows)
        print(f"[저장] {ev_path}")

    el = time.perf_counter() - t_start
    summary = {
        "src": args.src, "files_scanned": len(files) - skipped, "files_skipped": skipped,
        "sample_fps": args.sample_fps, "frames_inferred": total_infer,
        "elapsed_s": round(el, 1), "candidates": len(rows),
        "by_type": dict(defaultdict(int, {k: sum(1 for r in rows if r["type"] == k)
                                          for k in set(r["type"] for r in rows)})),
    }
    with open(os.path.join(args.out, f"{args.label}_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[저장] {csv_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if ev_rows:
        print("\n=== EVENT 분류 요약 ===")
        cnt = defaultdict(int)
        for e in ev_rows:
            cnt[e["kind"]] += 1
        for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
            print(f"  {k:<22} {v}개")


if __name__ == "__main__":
    main()
