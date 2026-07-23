"""
NORMAL 주행영상 컨택트시트 생성 — 육안 선별용.

182개를 전부 재생하지 않고도 "대형차가 앞을 가린 구간"이 있는지 눈으로 찾을 수 있게,
파일당 프레임을 균등 추출해 격자 이미지로 묶는다.
파일명·타임코드를 각 썸네일에 새겨두어, 쓸 만한 장면을 찾으면 바로 원본을 열 수 있다.
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

TH_W, TH_H = 400, 225        # 썸네일 크기 (트럭 식별 가능한 최소선)
COLS = 6                     # 파일당 추출 프레임 수 = 가로 칸 수
FILES_PER_SHEET = 8


def imwrite_unicode(path: str, img) -> bool:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def build_row(path: str, name: str, cols: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    idxs = np.linspace(0, max(total - 1, 0), cols).astype(int)
    thumbs = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if not ok or fr is None:
            thumbs.append(np.zeros((TH_H, TH_W, 3), np.uint8))
            continue
        t = cv2.resize(fr, (TH_W, TH_H), interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"{fi / fps:5.1f}s", (6, TH_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(t)
    cap.release()
    row = np.hstack(thumbs)
    # 파일명 띠
    band = np.full((26, row.shape[1], 3), 30, np.uint8)
    cv2.putText(band, name, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([band, row])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-duration", type=float, default=55.0,
                    help="이 길이 이상만 (60초 완전본 선별)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cols", type=int, default=COLS)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = sorted(f for f in os.listdir(args.src) if f.lower().endswith(".avi"))

    keep = []
    for n in files:
        cap = cv2.VideoCapture(os.path.join(args.src, n))
        d = 0.0
        if cap.isOpened():
            fp = cap.get(cv2.CAP_PROP_FPS) or 30.0
            d = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fp
        cap.release()
        if d >= args.min_duration:
            keep.append(n)
    if args.limit:
        keep = keep[: args.limit]
    print(f"대상 {len(keep)}개 (전체 {len(files)}개 중 {args.min_duration}s 이상)", flush=True)

    rows, sheet_i, made = [], 0, 0
    for i, n in enumerate(keep, 1):
        r = build_row(os.path.join(args.src, n), n, args.cols)
        if r is not None:
            rows.append(r)
        if len(rows) >= FILES_PER_SHEET or i == len(keep):
            if rows:
                sheet_i += 1
                sheet = np.vstack(rows)
                p = os.path.join(args.out, f"sheet_{sheet_i:02d}.png")
                imwrite_unicode(p, sheet)
                made += len(rows)
                print(f"  [{made}/{len(keep)}] {p}", flush=True)
                rows = []
    print(f"\n완료: 시트 {sheet_i}장")


if __name__ == "__main__":
    main()
