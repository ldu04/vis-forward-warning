"""
NORMAL 주행분 스캔 — (a) 대형차 시야차단 탐색에 최적화된 버전.

scan_blackbox.py 대비 개선점
---------------------------
1. **seek 제거**: cap.set(POS_FRAMES)는 키프레임까지 되감아 재디코딩하므로 매우 비싸다.
   순차 grab()으로 흘리고 필요한 프레임만 retrieve() 한다.
2. **추론 전 다운스케일**: YOLO가 어차피 imgsz=640으로 리사이즈하므로 미리 줄여도
   정확도 손실이 없고, 1080p→640 리사이즈 비용이 추론 입력 처리보다 싸다.
   면적비는 정규화 값이라 스케일 불변.
3. **위치 필터 완화**: 블랙박스 시점에서 대형차는 화면 중앙에 오지 않는 경우가 많다
   (EVENT 실측 cx 0.129~0.207). cx 조건을 대폭 완화하고 cy(세로) 조건만 유지해
   하늘·간판 오검출을 막는다. 면적 임계는 0.12로 낮춰 후보를 넓게 잡고,
   최종 선별은 사람이 PNG를 보고 한다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

CLS_PERSON, CLS_CAR, CLS_MOTO, CLS_BUS, CLS_TRUCK = 0, 2, 3, 5, 7
BIG_VEHICLE = {CLS_BUS, CLS_TRUCK}
ANY_VEHICLE = {CLS_CAR, CLS_BUS, CLS_TRUCK}
INFER_W = 640


def imwrite_unicode(path: str, img) -> bool:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def red_ratio_in_box(bgr, xyxy) -> float:
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


def scan_video(model, path: str, sample_fps: float, args) -> Dict[str, Any]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"ok": False, "reason": "열 수 없음"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0 or W0 <= 0:
        cap.release()
        return {"ok": False, "reason": "메타데이터 없음"}

    step = max(1, int(round(fps / sample_fps)))
    scale = INFER_W / float(W0)
    W, H = INFER_W, int(round(H0 * scale))

    samples: List[dict] = []
    i = 0
    n_infer = 0
    while True:
        if not cap.grab():          # 디코드만, 변환 없음 → seek 대비 훨씬 저렴
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                small = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
                res = model.predict(small, imgsz=INFER_W, conf=args.conf, verbose=False,
                                    classes=[CLS_PERSON, CLS_CAR, CLS_MOTO,
                                             CLS_BUS, CLS_TRUCK])[0]
                n_infer += 1
                big_area, big_conf, big_cls, big_cx, big_cy = 0.0, 0.0, -1, 0.0, 0.0
                big_area_raw, raw_cx, raw_cy = 0.0, 0.0, 0.0
                n_person, person_conf = 0, 0.0
                red_max, red_conf = 0.0, 0.0
                b = res.boxes
                if b is not None and len(b) > 0:
                    xy = b.xyxy.cpu().numpy()
                    cl = b.cls.cpu().numpy().astype(int)
                    cf = b.conf.cpu().numpy()
                    for k in range(len(cl)):
                        x1, y1, x2, y2 = [float(v) for v in xy[k]]
                        ar = max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(W * H)
                        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                        c = int(cl[k])
                        if c in BIG_VEHICLE:
                            # 진단용: 위치 필터를 적용하지 않은 원시 최대 면적비.
                            # 후보가 0건일 때 "장면이 없는 것"인지 "필터가 막은 것"인지
                            # 구분하려면 이 값이 필요하다.
                            if ar > big_area_raw:
                                big_area_raw, raw_cx, raw_cy = ar, cx, cy
                            # 위치 필터: cx는 넓게, cy만 유지 (하늘/간판 배제)
                            if args.cx_min <= cx <= args.cx_max and cy >= args.cy_min:
                                if ar > big_area:
                                    big_area, big_conf, big_cls = ar, float(cf[k]), c
                                    big_cx, big_cy = cx, cy
                        if c == CLS_PERSON:
                            n_person += 1
                            person_conf = max(person_conf, float(cf[k]))
                        if c in ANY_VEHICLE and ar >= 0.01:
                            r = red_ratio_in_box(small, (x1, y1, x2, y2))
                            if r > red_max:
                                red_max, red_conf = r, float(cf[k])
                samples.append({"t": i / fps, "frame": i, "big_area": big_area,
                                "big_conf": big_conf, "big_cls": big_cls,
                                "big_cx": big_cx, "big_cy": big_cy,
                                "big_area_raw": big_area_raw,
                                "raw_cx": raw_cx, "raw_cy": raw_cy,
                                "n_person": n_person, "person_conf": person_conf,
                                "red": red_max, "red_conf": red_conf})
        i += 1
    cap.release()
    return {"ok": True, "fps": fps, "total": total, "dur": total / fps,
            "samples": samples, "n_infer": n_infer}


def find_segments(samples, key, thr, min_sec, max_gap_s=0.6):
    segs, cur, last_t = [], [], None
    for s in samples:
        if s[key] >= thr:
            if last_t is not None and (s["t"] - last_t) > max_gap_s and cur:
                segs.append(cur); cur = []
            cur.append(s); last_t = s["t"]
        else:
            if cur:
                segs.append(cur); cur = []
            last_t = None
    if cur:
        segs.append(cur)
    out = []
    for seg in segs:
        t0, t1 = seg[0]["t"], seg[-1]["t"]
        if (t1 - t0) >= min_sec - 1e-6:
            vals = [x[key] for x in seg]
            out.append((t0, t1, max(vals), float(np.mean(vals)), seg))
    return out


def grab_full_frame(path: str, t: float):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--sample-fps", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-duration", type=float, default=0.0)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--occl-area", type=float, default=0.12, help="시야차단 면적비 임계")
    ap.add_argument("--occl-min-sec", type=float, default=2.0)
    ap.add_argument("--cx-min", type=float, default=0.10)
    ap.add_argument("--cx-max", type=float, default=0.90)
    ap.add_argument("--cy-min", type=float, default=0.40)
    ap.add_argument("--reveal-area", type=float, default=0.05)
    ap.add_argument("--brake-hi", type=float, default=0.08)
    ap.add_argument("--brake-near", type=float, default=0.05)
    ap.add_argument("--save-all-occlusion", action="store_true",
                    help="(a) 후보를 전부 PNG로 저장 (육안 선별용)")
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
    t0 = time.perf_counter()
    n_infer_tot = 0
    scanned = 0
    skipped = 0
    per_file_times = []
    diag_raw: List[float] = []
    diag_over_012 = 0
    diag_over_008 = 0
    diag_pos_at_max = [0.0, 0.0]

    for fi, name in enumerate(files, 1):
        path = os.path.join(args.src, name)
        # 길이 필터는 스캔 '전에' 본다. 스캔 후에 버리면 그 시간이 통째로 낭비된다.
        if args.min_duration:
            probe = cv2.VideoCapture(path)
            d = 0.0
            if probe.isOpened():
                pf = probe.get(cv2.CAP_PROP_FPS) or 30.0
                d = (probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / pf
            probe.release()
            if d < args.min_duration:
                skipped += 1
                continue
        ft = time.perf_counter()
        r = scan_video(model, path, args.sample_fps, args)
        if not r.get("ok"):
            print(f"[{fi}/{len(files)}] {name} 실패: {r.get('reason')}", flush=True)
            continue
        scanned += 1
        n_infer_tot += r["n_infer"]
        per_file_times.append(time.perf_counter() - ft)
        S = r["samples"]

        for s in S:
            ra = s.get("big_area_raw", 0.0)
            if ra > 0:
                diag_raw.append(ra)
                if ra >= 0.12:
                    diag_over_012 += 1
                if ra >= 0.08:
                    diag_over_008 += 1
                if ra >= max(diag_raw, default=0.0):
                    diag_pos_at_max = [round(s.get("raw_cx", 0), 3),
                                       round(s.get("raw_cy", 0), 3)]

        occl = find_segments(S, "big_area", args.occl_area, args.occl_min_sec)
        for (a0, a1, mx, mean, seg) in occl:
            peak = max(seg, key=lambda x: x["big_area"])
            row = {"file": name, "type": "a_occlusion", "t_start": round(a0, 2),
                   "t_end": round(a1, 2), "dur_s": round(a1 - a0, 2),
                   "area_ratio_max": round(mx, 4), "area_ratio_mean": round(mean, 4),
                   "conf": round(peak["big_conf"], 3),
                   "cls": "truck" if peak["big_cls"] == CLS_TRUCK else "bus",
                   "cx": round(peak["big_cx"], 3), "cy": round(peak["big_cy"], 3),
                   "stage": 1}
            if args.save_all_occlusion:
                fr = grab_full_frame(path, peak["t"])
                if fr is not None:
                    png = (f"a_{os.path.splitext(name)[0]}_t{peak['t']:.1f}s"
                           f"_ar{mx:.3f}.png")
                    if imwrite_unicode(os.path.join(cand_dir, png), fr):
                        row["frame_png"] = png
                        row["stage"] = 2
            rows.append(row)
            after = [s for s in S if a1 < s["t"] <= a1 + 3.0]
            if after and min(s["big_area"] for s in after) <= args.reveal_area:
                rv = next(s for s in after if s["big_area"] <= args.reveal_area)
                rows.append({"file": name, "type": "b_reveal", "t_start": round(a1, 2),
                             "t_end": round(rv["t"], 2), "dur_s": round(rv["t"] - a1, 2),
                             "area_ratio_max": round(mx, 4),
                             "area_ratio_mean": round(rv["big_area"], 4),
                             "conf": 0.0, "cls": "", "cx": 0.0, "cy": 0.0, "stage": 1})

        for (b0, b1, mx, mean, seg) in find_segments(S, "red", args.brake_hi, 0.0):
            rows.append({"file": name, "type": "c_brake", "t_start": round(b0, 2),
                         "t_end": round(b1, 2), "dur_s": round(b1 - b0, 2),
                         "area_ratio_max": round(mx, 4), "area_ratio_mean": round(mean, 4),
                         "conf": round(max(x["red_conf"] for x in seg), 3),
                         "cls": "", "cx": 0.0, "cy": 0.0, "stage": 1})
        for (b0, b1, mx, mean, seg) in find_segments(S, "red", args.brake_near, 0.0):
            if mx < args.brake_hi:
                rows.append({"file": name, "type": "c_brake_near", "t_start": round(b0, 2),
                             "t_end": round(b1, 2), "dur_s": round(b1 - b0, 2),
                             "area_ratio_max": round(mx, 4),
                             "area_ratio_mean": round(mean, 4), "conf": 0.0,
                             "cls": "", "cx": 0.0, "cy": 0.0, "stage": 1})
        for (p0, p1, mx, mean, seg) in find_segments(S, "n_person", 1, 0.0):
            rows.append({"file": name, "type": "d_person", "t_start": round(p0, 2),
                         "t_end": round(p1, 2), "dur_s": round(p1 - p0, 2),
                         "area_ratio_max": round(mx, 4), "area_ratio_mean": round(mean, 4),
                         "conf": round(max(x["person_conf"] for x in seg), 3),
                         "cls": "", "cx": 0.0, "cy": 0.0, "stage": 1})

        if fi % 5 == 0 or fi == len(files):
            el = time.perf_counter() - t0
            avg = float(np.mean(per_file_times)) if per_file_times else 0.0
            na = sum(1 for x in rows if x["type"] == "a_occlusion")
            print(f"[{fi}/{len(files)}] 후보 {len(rows)}건 (a={na}), "
                  f"파일당 {avg:.1f}s, {el:.0f}s 경과", flush=True)

    cols = ["file", "type", "t_start", "t_end", "dur_s", "area_ratio_max",
            "area_ratio_mean", "conf", "cls", "cx", "cy", "stage", "frame_png"]
    csv_path = os.path.join(args.out, f"{args.label}_candidates.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    by_type = {k: sum(1 for x in rows if x["type"] == k)
               for k in sorted(set(x["type"] for x in rows))}
    # 진단: (a)가 0건일 때 원인을 가리기 위한 원시 관측값
    diag = {
        "max_big_area_raw": round(max(diag_raw, default=0.0), 4),
        "n_frames_raw_over_012": diag_over_012,
        "n_frames_raw_over_008": diag_over_008,
        "raw_pos_at_max": diag_pos_at_max,
    }
    summary = {"src": args.src, "files_scanned": scanned, "files_skipped": skipped,
               "frames_inferred": n_infer_tot,
               "elapsed_s": round(time.perf_counter() - t0, 1),
               "sec_per_file": round(float(np.mean(per_file_times)), 2)
               if per_file_times else 0,
               "candidates": len(rows), "by_type": by_type,
               "diagnostics": diag,
               "filter": {"occl_area": args.occl_area, "cx": [args.cx_min, args.cx_max],
                          "cy_min": args.cy_min, "min_sec": args.occl_min_sec}}
    with open(os.path.join(args.out, f"{args.label}_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {csv_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
