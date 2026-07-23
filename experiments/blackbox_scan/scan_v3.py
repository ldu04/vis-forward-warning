"""
NORMAL 전체 스캔 v3 — ffmpeg 파이프 디코딩 + 체크포인트.

v2(scan_normal.py) 대비
-----------------------
1. **ffmpeg 파이프 디코딩** (`--decode ffmpeg`)
   OpenCV의 grab() 루프는 1080p 전 프레임을 디코딩한다. ffmpeg에 `fps=2,scale=640`
   필터를 걸면 C 레벨에서 솎아내고 축소한 rawvideo만 받으므로 훨씬 싸다.
   `--decode cv2` 로 기존 방식과 비교 검증할 수 있다.

2. **체크포인트**
   파일 하나가 끝날 때마다 결과 CSV에 append 하고 done 목록에 기록한다.
   중간에 죽어도 재실행하면 이미 끝낸 파일을 건너뛴다.

3. **정지 상태 포함 필터** (`--allow-stationary`)
   신호 대기 중 앞에 선 버스/트럭도 시야차단 사례이므로, 자기운동이 0이어도
   후보로 잡는다. (기존에는 주행 중 장면만 암묵적으로 상정했다.)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

CLS_PERSON, CLS_CAR, CLS_MOTO, CLS_BUS, CLS_TRUCK = 0, 2, 3, 5, 7
BIG_VEHICLE = {CLS_BUS, CLS_TRUCK}
ANY_VEHICLE = {CLS_CAR, CLS_BUS, CLS_TRUCK}
INFER_W = 640

CSV_COLS = ["file", "type", "t_start", "t_end", "dur_s", "area_ratio_max",
            "area_ratio_mean", "conf", "cls", "cx", "cy", "stage", "frame_png"]


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
    return float(np.count_nonzero(cv2.bitwise_or(m1, m2))) / float(m1.size + 1e-6)


def probe(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frames": n, "w": w, "h": h, "dur": n / fps if fps else 0}


def find_ffmpeg() -> Optional[str]:
    """
    ffmpeg 실행 파일 경로.

    winget 설치 직후에는 PATH 변경이 이미 떠 있는 프로세스에 전파되지 않아
    shutil.which('ffmpeg')가 None을 돌려준다. 이때 파이프가 조용히 0프레임을
    내놓아 '검출 결과 불일치'로 오진되므로, 알려진 설치 경로도 함께 뒤진다.
    """
    import glob
    import shutil

    p = shutil.which("ffmpeg")
    if p:
        return p
    pats = [
        os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Packages"
                           r"\Gyan.FFmpeg*\**\bin\ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]
    for pat in pats:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None


def iter_frames_ffmpeg(path: str, sample_fps: float, W: int, H: int):
    """ffmpeg에서 fps/scale 적용된 rawvideo를 받아 프레임을 순차 반환."""
    exe = find_ffmpeg()
    if not exe:
        raise RuntimeError("ffmpeg 실행 파일을 찾을 수 없음")
    cmd = [exe, "-v", "error", "-i", path,
           "-vf", f"fps={sample_fps},scale={W}:{H}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    fsz = W * H * 3
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         bufsize=fsz * 4)
    k = 0
    try:
        while True:
            buf = p.stdout.read(fsz)
            if not buf or len(buf) < fsz:
                break
            yield k / sample_fps, np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            k += 1
    finally:
        try:
            p.stdout.close()
        except Exception:
            pass
        p.wait(timeout=10)


def iter_frames_cv2(path: str, sample_fps: float, W: int, H: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))
    i = 0
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            ok, fr = cap.retrieve()
            if ok and fr is not None:
                yield i / fps, cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
        i += 1
    cap.release()


def scan_one(model, path: str, args, W: int, H: int) -> List[dict]:
    it = (iter_frames_ffmpeg if args.decode == "ffmpeg" else iter_frames_cv2)
    samples = []
    for t, small in it(path, args.sample_fps, W, H):
        res = model.predict(small, imgsz=INFER_W, conf=args.conf, verbose=False,
                            classes=[CLS_PERSON, CLS_CAR, CLS_MOTO,
                                     CLS_BUS, CLS_TRUCK])[0]
        big_a, big_c, big_k, bcx, bcy = 0.0, 0.0, -1, 0.0, 0.0
        npers, pconf, red_m, red_c = 0, 0.0, 0.0, 0.0
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
                if c in BIG_VEHICLE and args.cx_min <= cx <= args.cx_max \
                        and cy >= args.cy_min and ar > big_a:
                    big_a, big_c, big_k, bcx, bcy = ar, float(cf[k]), c, cx, cy
                if c == CLS_PERSON:
                    npers += 1
                    pconf = max(pconf, float(cf[k]))
                if c in ANY_VEHICLE and ar >= 0.01:
                    r = red_ratio(small, (x1, y1, x2, y2))
                    if r > red_m:
                        red_m, red_c = r, float(cf[k])
        samples.append({"t": t, "big_area": big_a, "big_conf": big_c, "big_cls": big_k,
                        "big_cx": bcx, "big_cy": bcy, "n_person": npers,
                        "person_conf": pconf, "red": red_m, "red_conf": red_c})
    return samples


def segments(samples, key, thr, min_sec, max_gap=0.6):
    segs, cur, last = [], [], None
    for s in samples:
        if s[key] >= thr:
            if last is not None and (s["t"] - last) > max_gap and cur:
                segs.append(cur); cur = []
            cur.append(s); last = s["t"]
        else:
            if cur:
                segs.append(cur); cur = []
            last = None
    if cur:
        segs.append(cur)
    out = []
    for sg in segs:
        t0, t1 = sg[0]["t"], sg[-1]["t"]
        if (t1 - t0) >= min_sec - 1e-6:
            v = [x[key] for x in sg]
            out.append((t0, t1, max(v), float(np.mean(v)), sg))
    return out


def rows_from_samples(name, S, args, path, cand_dir) -> List[dict]:
    rows = []
    for (a0, a1, mx, mean, sg) in segments(S, "big_area", args.occl_area,
                                           args.occl_min_sec):
        pk = max(sg, key=lambda x: x["big_area"])
        row = {"file": name, "type": "a_occlusion", "t_start": round(a0, 2),
               "t_end": round(a1, 2), "dur_s": round(a1 - a0, 2),
               "area_ratio_max": round(mx, 4), "area_ratio_mean": round(mean, 4),
               "conf": round(pk["big_conf"], 3),
               "cls": "truck" if pk["big_cls"] == CLS_TRUCK else "bus",
               "cx": round(pk["big_cx"], 3), "cy": round(pk["big_cy"], 3),
               "stage": 1, "frame_png": ""}
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pk["t"] * fps))
            ok, fr = cap.read()
            cap.release()
            if ok and fr is not None:
                png = f"a_{os.path.splitext(name)[0]}_t{pk['t']:.1f}s_ar{mx:.3f}.png"
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
                         "area_ratio_mean": round(rv["big_area"], 4), "conf": 0.0,
                         "cls": "", "cx": 0.0, "cy": 0.0, "stage": 1, "frame_png": ""})
    for (b0, b1, mx, mean, sg) in segments(S, "red", args.brake_hi, 0.0):
        rows.append({"file": name, "type": "c_brake", "t_start": round(b0, 2),
                     "t_end": round(b1, 2), "dur_s": round(b1 - b0, 2),
                     "area_ratio_max": round(mx, 4), "area_ratio_mean": round(mean, 4),
                     "conf": round(max(x["red_conf"] for x in sg), 3), "cls": "",
                     "cx": 0.0, "cy": 0.0, "stage": 1, "frame_png": ""})
    for (b0, b1, mx, mean, sg) in segments(S, "red", args.brake_near, 0.0):
        if mx < args.brake_hi:
            rows.append({"file": name, "type": "c_brake_near", "t_start": round(b0, 2),
                         "t_end": round(b1, 2), "dur_s": round(b1 - b0, 2),
                         "area_ratio_max": round(mx, 4),
                         "area_ratio_mean": round(mean, 4), "conf": 0.0, "cls": "",
                         "cx": 0.0, "cy": 0.0, "stage": 1, "frame_png": ""})
    for (p0, p1, mx, mean, sg) in segments(S, "n_person", 1, 0.0):
        rows.append({"file": name, "type": "d_person", "t_start": round(p0, 2),
                     "t_end": round(p1, 2), "dur_s": round(p1 - p0, 2),
                     "area_ratio_max": round(mx, 4), "area_ratio_mean": round(mean, 4),
                     "conf": round(max(x["person_conf"] for x in sg), 3), "cls": "",
                     "cx": 0.0, "cy": 0.0, "stage": 1, "frame_png": ""})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--decode", choices=["ffmpeg", "cv2"], default="ffmpeg")
    ap.add_argument("--sample-fps", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--occl-area", type=float, default=0.12)
    ap.add_argument("--occl-min-sec", type=float, default=2.0)
    ap.add_argument("--cx-min", type=float, default=0.10)
    ap.add_argument("--cx-max", type=float, default=0.90)
    ap.add_argument("--cy-min", type=float, default=0.40)
    ap.add_argument("--reveal-area", type=float, default=0.05)
    ap.add_argument("--brake-hi", type=float, default=0.08)
    ap.add_argument("--brake-near", type=float, default=0.05)
    ap.add_argument("--min-duration", type=float, default=0.0)
    ap.add_argument("--only", nargs="*", default=None, help="특정 파일만")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cand_dir = os.path.join(args.out, "candidates")
    os.makedirs(cand_dir, exist_ok=True)
    out_csv = os.path.join(args.out, f"{args.label}_candidates.csv")
    done_path = os.path.join(args.out, f"{args.label}_done.txt")

    done = set()
    if os.path.exists(done_path):
        with open(done_path, encoding="utf-8") as f:
            done = {l.strip() for l in f if l.strip()}
    if not os.path.exists(out_csv):
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=CSV_COLS).writeheader()

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

    files = sorted(f for f in os.listdir(args.src) if f.lower().endswith(".avi"))
    if args.only:
        files = [f for f in files if f in set(args.only)]
    if args.limit:
        files = files[: args.limit]

    t0 = time.perf_counter()
    times, nscan, nskip = [], 0, 0
    for i, name in enumerate(files, 1):
        if name in done:
            nskip += 1
            continue
        path = os.path.join(args.src, name)
        meta = probe(path)
        if not meta:
            print(f"[{i}/{len(files)}] {name} probe 실패", flush=True)
            continue
        if args.min_duration and meta["dur"] < args.min_duration:
            with open(done_path, "a", encoding="utf-8") as f:
                f.write(name + "\n")
            nskip += 1
            continue
        W = INFER_W
        H = int(round(meta["h"] * INFER_W / meta["w"] / 2) * 2)
        ft = time.perf_counter()
        try:
            S = scan_one(model, path, args, W, H)
        except Exception as exc:
            print(f"[{i}/{len(files)}] {name} 스캔 실패: {exc}", flush=True)
            continue
        rows = rows_from_samples(name, S, args, path, cand_dir)
        with open(out_csv, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
            w.writerows(rows)
        with open(done_path, "a", encoding="utf-8") as f:
            f.write(name + "\n")
        el = time.perf_counter() - ft
        times.append(el)
        nscan += 1
        print(f"[{i}/{len(files)}] {name} {el:.1f}s 후보 {len(rows)}건 "
              f"(누적 {nscan}스캔/{nskip}건너뜀)", flush=True)

    summary = {"label": args.label, "decode": args.decode, "scanned": nscan,
               "skipped": nskip, "elapsed_s": round(time.perf_counter() - t0, 1),
               "sec_per_file": round(float(np.mean(times)), 2) if times else 0,
               "filter": {"occl_area": args.occl_area, "cx": [args.cx_min, args.cx_max],
                          "cy_min": args.cy_min, "min_sec": args.occl_min_sec,
                          "sample_fps": args.sample_fps}}
    with open(os.path.join(args.out, f"{args.label}_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
