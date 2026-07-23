"""
EVENT 감속률 상위 N개의 대표 프레임 추출 — 육안 확인 전용.

자동 분류에서 '급감속(위험상황)' 판정이 0건이었으나, 그 판정은 임의로 정한
임계(감속 45%)에 따른 것이다. 임계를 재조정하지는 않되(사용자 지시),
감속률이 가장 큰 파일들이 실제로 어떤 상황이었는지는 사람이 확인할 수 있어야 한다.
그래서 각 파일의 시작·중간·끝 구간 프레임을 한 장으로 묶어 저장한다.
"""
from __future__ import annotations

import argparse
import csv
import os

import cv2
import numpy as np


def imwrite_unicode(path: str, img) -> bool:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", required=True)
    ap.add_argument("--srcdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.classification, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: -float(r.get("drop_pct") or 0))
    top = rows[: args.top]

    summary = []
    for rank, r in enumerate(top, 1):
        vid = os.path.join(args.srcdir, r["file"])
        if not os.path.exists(vid):
            continue
        cap = cv2.VideoCapture(vid)
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(total - 1, 0), args.frames).astype(int)
        thumbs = []
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, fr = cap.read()
            if not ok or fr is None:
                thumbs.append(np.zeros((360, 640, 3), np.uint8))
                continue
            t = cv2.resize(fr, (640, 360), interpolation=cv2.INTER_AREA)
            cv2.putText(t, f"{fi / fps:5.1f}s", (8, 348),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            thumbs.append(t)
        cap.release()

        grid = np.vstack([np.hstack(thumbs[:3]), np.hstack(thumbs[3:6])]) \
            if len(thumbs) >= 6 else np.hstack(thumbs)
        band = np.full((44, grid.shape[1], 3), 25, np.uint8)
        txt = (f"#{rank}  {r['file']}   drop={r['drop_pct']}%  "
               f"flow {r['flow_head']} -> {r['flow_tail']}  "
               f"vy_spike={r['vy_spike']}  [{r['kind']}]")
        cv2.putText(band, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (255, 255, 255), 2, cv2.LINE_AA)
        out_img = np.vstack([band, grid])
        name = f"decel_{rank:02d}_drop{float(r['drop_pct']):05.1f}_{os.path.splitext(r['file'])[0]}.png"
        imwrite_unicode(os.path.join(args.out, name), out_img)
        summary.append({"rank": rank, "file": r["file"], "drop_pct": r["drop_pct"],
                        "kind": r["kind"], "why": r.get("why", ""), "png": name})
        print(f"[{rank}] {r['file']} drop={r['drop_pct']}% → {name}", flush=True)

    with open(os.path.join(args.out, "decel_top.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        if summary:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
    print(f"\n[저장] {args.out}")


if __name__ == "__main__":
    main()
