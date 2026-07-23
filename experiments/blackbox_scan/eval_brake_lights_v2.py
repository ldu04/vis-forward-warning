"""
브레이크등 재추출 v2 — '내 앞차' 한정 + naive/roi/bright/dRed 4중 신호 + A/B/C 분류.

목적(S18): (A) 실제 점등 후보, (B) naive 방식(bbox 전체 붉은비율) 오검출 사례
(빨간 차체 → 단순 방식의 한계 전시용), (C) 근접 비점등 대비 프레임을 뽑는다.
정밀 검출기 완성이 목적이 아니라, '단순 붉은비율 → 개선된 다중신호'라는
스토리를 뒷받침할 근거 프레임 확보가 목적이다.

기존 eval_brake_lights.py 대비 변경점
------------------------------------
1. 선행차 한정: bbox 중심이 프레임 중앙 ±15% 이내인 차량 중 면적 최대(=가장 가까운)
   1대만 '선행차'로 채택. 옆차선 차량이 선행차로 오검출되던 문제를 없앤다.
2. red% 를 3가지로 계산:
   - naive_pct  : 기존 방식(bbox 전체 붉은 비율)
   - roi_pct    : bbox 후미 하단(세로 하위 40%, 좌우 폭 전체)만의 붉은 비율
   - bright_pct : roi 안에서 고휘도·고채도(S/V 높음) 붉은픽셀만의 비율 — 차체
     빨강(매트/음영)을 배제하고 실제 점등된 램프에 가깝게 좁힌 값
3. dRed: roi_pct 의 직전 N샘플(N_HIST) 대비 증가량. 점등 순간은 급증하지만
   빨간 차체는 시간에 따라 거의 변하지 않는다는 점을 이용.
4. 분류: A(점등 후보)=bright 높음 AND dRed 급증 / B(naive 오검출)=naive는
   높은데 bright는 낮음 / C(근접 비점등)=그 외. 후보(원본 이벤트)당 대표
   프레임 1장만 저장(우선순위 A>B>C, 각 카테고리 내 최댓값).

개인정보 처리는 하지 않는다 — 이 출력은 선별용이며, 실제 채택 프레임만
수동으로 블러 처리한다 (지시사항).
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import deque

import cv2
import numpy as np

# ---- 임계값 (정밀 검출기가 아니므로 대략치 — 필요시 재조정) ----
NAIVE_THR = 0.08        # 기존 방식 임계(8%), naive_pct 판정 기준 그대로 유지
BRIGHT_THR = 0.05       # roi 내 고휘도·고채도 붉은픽셀 비율 임계(점등 판정)
DRED_THR = 0.03         # roi_pct 의 직전 N샘플 대비 증가량 임계(급점등 판정)
N_HIST = 3              # dRed 비교 대상: 직전 N 샘플(= STEP_NATIVE_FRAMES * N_HIST 프레임 전)
STEP_NATIVE_FRAMES = 3  # 원본 프레임 중 몇 프레임마다 1회 샘플링(성능/정밀도 균형)
CENTER_TOL = 0.15       # 선행차 판정: bbox 중심 x 가 프레임 중앙 ±15% 이내
MIN_AREA_RATIO = 0.01   # 프레임 대비 bbox 면적이 이보다 작으면(원거리) 제외

ANY_VEHICLE = {2, 5, 7}  # YOLO: car, bus, truck


def imwrite_unicode(path: str, img) -> bool:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def _red_mask(hsv, s_lo: int, v_lo: int):
    m1 = cv2.inRange(hsv, (0, s_lo, v_lo), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, s_lo, v_lo), (180, 255, 255))
    return cv2.bitwise_or(m1, m2)


def region_red_ratio(bgr, x1, y1, x2, y2, s_lo=70, v_lo=50) -> float:
    h, w = bgr.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = _red_mask(hsv, s_lo, v_lo)
    return float(np.count_nonzero(mask)) / float(mask.size + 1e-6)


def leading_vehicle_box(res, W, H):
    """중앙 ±CENTER_TOL 이내 차량 중 면적 최대(=가장 가까운) 1대만 반환. 없으면 None."""
    if res.boxes is None or len(res.boxes) == 0:
        return None
    xy = res.boxes.xyxy.cpu().numpy()
    cl = res.boxes.cls.cpu().numpy().astype(int)
    best, best_area = None, 0.0
    for k in range(len(cl)):
        if cl[k] not in ANY_VEHICLE:
            continue
        x1, y1, x2, y2 = xy[k]
        area = (x2 - x1) * (y2 - y1) / (W * H)
        if area < MIN_AREA_RATIO:
            continue
        cx = (x1 + x2) / 2.0 / W
        if abs(cx - 0.5) > CENTER_TOL:
            continue  # 옆차선 등 중앙에서 벗어난 차량 제외 → '내 앞차'만
        if area > best_area:
            best_area, best = area, (x1, y1, x2, y2)
    return best


def taillight_roi(x1, y1, x2, y2, y_frac=0.40):
    """bbox 후미 하단(세로 하위 y_frac), 좌우 폭은 그대로(램프가 보통 좌우 끝에 있음)."""
    h = y2 - y1
    ry1 = y1 + h * (1 - y_frac)
    return x1, ry1, x2, y2


def classify(naive_pct, bright_pct, dred_pct) -> str:
    if bright_pct >= BRIGHT_THR * 100 and dred_pct >= DRED_THR * 100:
        return "A"
    if naive_pct >= NAIVE_THR * 100 and bright_pct < BRIGHT_THR * 100:
        return "B"
    return "C"


def process_candidate(model, vid, t_start, t_end, min_window=2.0):
    if t_end <= t_start:
        t_start, t_end = max(0.0, t_start - min_window / 2), t_start + min_window / 2
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        return None, "open_failed"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_f, end_f = int(t_start * fps), int(t_end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    hist = deque(maxlen=N_HIST)
    best = {"A": None, "B": None, "C": None}
    fidx = start_f
    while fidx <= end_f:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if (fidx - start_f) % STEP_NATIVE_FRAMES == 0:
            H, W = frame.shape[:2]
            res = model.predict(frame, imgsz=640, conf=0.30, verbose=False,
                                classes=list(ANY_VEHICLE))[0]
            box = leading_vehicle_box(res, W, H)
            if box is not None:
                x1, y1, x2, y2 = box
                naive = region_red_ratio(frame, x1, y1, x2, y2) * 100
                rx1, ry1, rx2, ry2 = taillight_roi(x1, y1, x2, y2)
                roi_pct = region_red_ratio(frame, rx1, ry1, rx2, ry2) * 100
                bright_pct = region_red_ratio(frame, rx1, ry1, rx2, ry2,
                                              s_lo=150, v_lo=180) * 100
                prev = hist[0] if len(hist) == N_HIST else None
                dred = (roi_pct - prev) if prev is not None else 0.0
                hist.append(roi_pct)
                verdict = classify(naive, bright_pct, dred)
                rec = dict(naive=naive, roi=roi_pct, bright=bright_pct, dred=dred,
                          verdict=verdict, t=fidx / fps, box=(x1, y1, x2, y2),
                          roi_box=(rx1, ry1, rx2, ry2), frame=frame)
                if verdict == "A":
                    if best["A"] is None or bright_pct > best["A"]["bright"]:
                        best["A"] = rec
                elif verdict == "B":
                    if best["B"] is None or naive > best["B"]["naive"]:
                        best["B"] = rec
                else:
                    if best["C"] is None or roi_pct > best["C"]["roi"]:
                        best["C"] = rec
        fidx += 1
    cap.release()
    chosen = best["A"] or best["B"] or best["C"]
    if chosen is None:
        return None, "no_lead_vehicle"
    return chosen, "ok"


def draw_and_save(chosen, out_path, file_label):
    vis = chosen["frame"].copy()
    x1, y1, x2, y2 = [int(v) for v in chosen["box"]]
    rx1, ry1, rx2, ry2 = [int(v) for v in chosen["roi_box"]]
    col = {"A": (0, 0, 255), "B": (0, 165, 255), "C": (160, 160, 160)}[chosen["verdict"]]
    cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
    line1 = (f"naive={chosen['naive']:.1f}%  roi={chosen['roi']:.1f}%  "
             f"bright={chosen['bright']:.1f}%  dRed={chosen['dred']:+.1f}%p")
    line2 = f"verdict={chosen['verdict']}  t={chosen['t']:.1f}s"
    cv2.rectangle(vis, (x1, max(0, y1 - 56)), (x1 + 640, y1), col, -1)
    cv2.putText(vis, line1, (x1 + 6, max(20, y1 - 34)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, line2, (x1 + 6, max(40, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, file_label, (12, vis.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    imwrite_unicode(out_path, vis)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True,
                    help="label=csv_path=srcdir 형식")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--png-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.png_dir, exist_ok=True)
    from ultralytics import YOLO
    model = YOLO(os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt"))

    rows = []
    idx = 0
    for spec in args.sources:
        label, path, srcdir = spec.split("=", 2)
        if not os.path.exists(path):
            print(f"[skip] {path} 없음")
            continue
        with open(path, newline="", encoding="utf-8-sig") as f:
            cands = [r for r in csv.DictReader(f)
                     if r.get("type", "").startswith("c_brake")]
        print(f"[{label}] 후보 {len(cands)}건", flush=True)

        for c in cands:
            idx += 1
            vid = os.path.join(srcdir, c["file"])
            if not os.path.exists(vid):
                rows.append({"idx": idx, "set": label, "file": c["file"], "status": "video_missing"})
                print(f"  [{idx}] {c['file']} 원본 없음 → skip")
                continue

            chosen, status = process_candidate(model, vid, float(c["t_start"]), float(c["t_end"]))
            if status != "ok":
                rows.append({"idx": idx, "set": label, "file": c["file"], "status": status})
                print(f"  [{idx}] {c['file']} → {status}")
                continue

            png_name = f"{idx:03d}_{chosen['verdict']}_{os.path.splitext(c['file'])[0]}_t{chosen['t']:.1f}s.png"
            draw_and_save(chosen, os.path.join(args.png_dir, png_name), c["file"])

            rows.append({
                "idx": idx, "set": label, "file": c["file"],
                "t_start": c["t_start"], "t_end": c["t_end"], "t_peak_s": round(chosen["t"], 2),
                "naive_pct": round(chosen["naive"], 2),
                "roi_pct": round(chosen["roi"], 2),
                "bright_pct": round(chosen["bright"], 2),
                "dred_pct": round(chosen["dred"], 2),
                "verdict": chosen["verdict"],
                "frame_png": png_name,
                "status": "ok",
            })
            print(f"  [{idx}] {c['file']} t={chosen['t']:.1f}s "
                  f"naive={chosen['naive']:.1f}% roi={chosen['roi']:.1f}% "
                  f"bright={chosen['bright']:.1f}% dRed={chosen['dred']:+.1f}%p "
                  f"-> {chosen['verdict']}", flush=True)

    cols = ["idx", "set", "file", "t_start", "t_end", "t_peak_s",
            "naive_pct", "roi_pct", "bright_pct", "dred_pct", "verdict",
            "frame_png", "status"]
    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    print(f"\n=== 요약 (전체 {len(rows)}건 중 처리 {len(ok_rows)}건) ===")
    for v in ("A", "B", "C"):
        n = sum(1 for r in ok_rows if r.get("verdict") == v)
        print(f"  {v} {n}건")
    n_skip = len(rows) - len(ok_rows)
    print(f"  skip(원본없음/선행차미검출/열기실패) {n_skip}건")
    print(f"[저장] {args.out_csv}")


if __name__ == "__main__":
    main()
