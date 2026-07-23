"""
브레이크등 후보 정밀 평가 — 제안서 4-4 "실사 축" 핵심 근거.

왜 실사여야 하는가: 브레이크등은 시뮬레이션과 실사의 색감·노출·반사 특성이 달라
CARLA 렌더링으로 대체 검증할 수 없다. 실제 주행영상에서 붉은픽셀 비율이 임계값
근처에서 어떻게 분포하는지가 임계값 타당성의 유일한 실측 근거다.

하는 일
-------
1. 기존 스캔이 뽑은 c_brake / c_brake_near 후보를 읽는다.
2. 각 후보의 피크 시각에서 **원본 해상도(1920x1080)** 프레임을 다시 열어
   차량 bbox 안의 붉은픽셀 비율을 재측정한다.
   (기존 값은 NORMAL의 경우 640 축소 프레임에서 잰 것이라 재측정이 필요하다.)
3. 임계 0.08 대비 판정: detected(>=0.08) / near(0.05~0.08) / miss(<0.05)
4. bbox와 측정값을 그려 넣은 PNG를 저장한다.

출력 파일은 항상 새 이름으로 쓴다 (덮어쓰기 금지 원칙).
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional

import cv2
import numpy as np

BRAKE_HI = 0.08
BRAKE_NEAR = 0.05
ANY_VEHICLE = {2, 5, 7}  # car, bus, truck


def imwrite_unicode(path: str, img) -> bool:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def red_ratio(bgr, xyxy) -> float:
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


def classify(r: float) -> str:
    if r >= BRAKE_HI:
        return "detected"
    if r >= BRAKE_NEAR:
        return "near"
    return "miss"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True,
                    help="후보 CSV 경로들 (라벨=경로 형식)")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--png-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.png_dir, exist_ok=True)
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

    rows: List[dict] = []
    idx = 0
    for spec in args.sources:
        label, path, srcdir = spec.split("=", 2)
        if not os.path.exists(path):
            print(f"[skip] {path} 없음")
            continue
        with open(path, newline="", encoding="utf-8-sig") as f:
            cands = [r for r in csv.DictReader(f)
                     if r.get("type", "").startswith("c_brake")]
        print(f"[{label}] 브레이크등 후보 {len(cands)}건", flush=True)

        for c in cands:
            idx += 1
            vid = os.path.join(srcdir, c["file"])
            if not os.path.exists(vid):
                rows.append({"idx": idx, "set": label, "file": c["file"],
                             "type_orig": c["type"], "status": "video_missing"})
                continue
            t_peak = (float(c["t_start"]) + float(c["t_end"])) / 2.0
            cap = cv2.VideoCapture(vid)
            if not cap.isOpened():
                rows.append({"idx": idx, "set": label, "file": c["file"],
                             "type_orig": c["type"], "status": "open_failed"})
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_peak * fps))
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                rows.append({"idx": idx, "set": label, "file": c["file"],
                             "type_orig": c["type"], "status": "frame_read_failed"})
                continue

            H, W = frame.shape[:2]
            res = model.predict(frame, imgsz=640, conf=0.35, verbose=False,
                                classes=[2, 5, 7])[0]
            best_r, best_box, best_cls, best_conf = 0.0, None, -1, 0.0
            if res.boxes is not None and len(res.boxes) > 0:
                xy = res.boxes.xyxy.cpu().numpy()
                cl = res.boxes.cls.cpu().numpy().astype(int)
                cf = res.boxes.conf.cpu().numpy()
                for k in range(len(cl)):
                    ar = ((xy[k][2] - xy[k][0]) * (xy[k][3] - xy[k][1])) / (W * H)
                    if ar < 0.01:
                        continue
                    r = red_ratio(frame, xy[k])
                    if r > best_r:
                        best_r, best_box = r, xy[k]
                        best_cls, best_conf = int(cl[k]), float(cf[k])

            verdict = classify(best_r)
            png_name = ""
            if best_box is not None:
                vis = frame.copy()
                x1, y1, x2, y2 = [int(v) for v in best_box]
                col = ((0, 0, 255) if verdict == "detected"
                       else (0, 165, 255) if verdict == "near" else (160, 160, 160))
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
                txt = f"red={best_r*100:.2f}%  thr=8%  {verdict}"
                cv2.rectangle(vis, (x1, max(0, y1 - 34)), (x1 + 460, y1), col, -1)
                cv2.putText(vis, txt, (x1 + 6, max(16, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.putText(vis, f"{c['file']}  t={t_peak:.1f}s",
                            (12, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2, cv2.LINE_AA)
                png_name = (f"{idx:03d}_{verdict}_{best_r*100:05.2f}pct_"
                            f"{os.path.splitext(c['file'])[0]}_t{t_peak:.1f}s.png")
                imwrite_unicode(os.path.join(args.png_dir, png_name), vis)

            rows.append({
                "idx": idx, "set": label, "file": c["file"],
                "type_orig": c["type"],
                "t_start": c["t_start"], "t_end": c["t_end"],
                "t_peak_s": round(t_peak, 2),
                "red_ratio_rescan": round(best_r, 5),
                "red_pct_rescan": round(best_r * 100, 2),
                "red_ratio_orig": c.get("area_ratio_max", ""),
                "threshold": BRAKE_HI,
                "verdict": verdict,
                "vehicle_cls": {2: "car", 5: "bus", 7: "truck"}.get(best_cls, ""),
                "det_conf": round(best_conf, 3),
                "frame_png": png_name,
                "status": "ok",
            })
            print(f"  [{idx}] {c['file']} t={t_peak:.1f}s "
                  f"red={best_r*100:.2f}% → {verdict}", flush=True)

    cols = ["idx", "set", "file", "type_orig", "t_start", "t_end", "t_peak_s",
            "red_ratio_rescan", "red_pct_rescan", "red_ratio_orig", "threshold",
            "verdict", "vehicle_cls", "det_conf", "frame_png", "status"]
    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    print(f"\n=== 브레이크등 평가 요약 (임계 {BRAKE_HI*100:.0f}%) ===")
    for v in ("detected", "near", "miss"):
        n = sum(1 for r in ok_rows if r.get("verdict") == v)
        print(f"  {v:<9} {n}건")
    if ok_rows:
        vals = [r["red_ratio_rescan"] for r in ok_rows]
        print(f"  붉은픽셀 비율: 최소 {min(vals)*100:.2f}% / "
              f"중앙 {float(np.median(vals))*100:.2f}% / 최대 {max(vals)*100:.2f}%")
    print(f"[저장] {args.out_csv}")


if __name__ == "__main__":
    main()
