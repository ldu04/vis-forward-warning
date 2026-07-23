"""
rebuild_full A 프레임(검출 성공) 2장을 개인정보 처리해 selected/d_brake_detected/ 로 승격.

B/C 승격(promote_brake_limitation.py)과 동일 방식:
 raw 프레임에 먼저 블러(OSD·번호판·도로표지 지명·인물·상호·앞유리 각인) → 저장된 검출
 수치로 오버레이(bbox·램프쌍·lamp%/dLamp/sym/verdict) 재작도 → 1280px 축소 → OCR 재검사.

대상: 204건 전수 재실행에서 나온 실제 점등(brake_on=Y) 4건 중 대표 2장.
 A01 20260720-11h47m23s_N t14.3s (dLamp 44.9x, sym=Y — 램프쌍 좌우 미등 정확 안착)
 A03 20260720-11h46m23s_N t59.3s (sym=Y)

★ 하드룰: 처리 후 덱에 바로 넣지 말 것. 블러 결과는 사용자 육안 검수 후 반영.
"""
from __future__ import annotations

import os
import re
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from rebuild_brake_detect import process_window, bucket_of
import promote_brake_limitation as pbl

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "candidates", "selected", "d_brake_detected")
MODEL = os.path.join(HERE, "yolov8n.pt")

# A 프레임 전용 수동 백스톱(육안 확인된 도로표지·번호판 위치). 자동 YOLO+OCR 블러에 더해 적용.
MANUAL_BLUR = {
    "a01_brake_detected_dLamp44.9x_symY": [
        (0.68, 0.00, 1.00, 0.17),   # 우상단 파란 도로표지(성림1로/지명)
        (0.82, 0.15, 0.96, 0.36),   # 우측 세로 파란 도로표지
        (0.39, 0.59, 0.56, 0.71),   # 선행차 후면 번호판(69마 3417) 백업
        (0.23, 0.48, 0.39, 0.60),   # 좌측 검정 세단 후면 번호판(자동블러 미흡분)
        (0.35, 0.74, 0.50, 0.87),   # 앞유리 각인 번호
    ],
    "a03_brake_detected_symY": [
        (0.68, 0.00, 1.00, 0.17),   # 우상단 파란 도로표지(성림1로/지명)
        (0.82, 0.15, 0.96, 0.36),   # 우측 세로 파란 도로표지
        (0.40, 0.61, 0.57, 0.72),   # 선행차 후면 번호판(69마 341x) 백업
        (0.58, 0.53, 0.75, 0.65),   # 우측 흰 박스카(레이) 후면 번호판
        (0.35, 0.74, 0.50, 0.87),   # 앞유리 각인 번호
    ],
}

TARGETS = [
    dict(out="a01_brake_detected_dLamp44.9x_symY",
         vid=r"C:\Users\이동욱\blackbox_full\NORMAL\20260720-11h47m23s_N.avi", t=14.3,
         note="빗길 적신호 정차 선행차 미등 점등 — dLamp 44.9x, 램프쌍 좌우 정확 안착"),
    dict(out="a03_brake_detected_symY",
         vid=r"C:\Users\이동욱\blackbox_full\NORMAL\20260720-11h46m23s_N.avi", t=59.3,
         note="빗길 적신호 정차 선행차 미등 점등 — sym=Y"),
]


def privacy_blur(frame, yolo, reader, manual_key):
    out = frame.copy()
    H, W = out.shape[:2]
    log = {"osd": 1, "plates": 0, "persons": 0, "texts": 0, "manual": 0}
    pbl.blur_region(out, 0, 0, W, int(H * pbl.OSD_FRAC))

    res = yolo.predict(frame, conf=pbl.YOLO_CONF, verbose=False,
                       classes=[pbl.PERSON_CLS, 2, 3, 5, 7])[0]
    if res.boxes is not None:
        for b in res.boxes:
            cls = int(b.cls[0])
            x0, y0, x1, y1 = [float(v) for v in b.xyxy[0]]
            bw, bh = x1 - x0, y1 - y0
            if cls in pbl.VEHICLE_CLS:
                cx = (x0 + x1) / 2
                pbl.blur_region(out, cx - bw * 0.32, y1 - bh / 3, cx + bw * 0.32, y1)
                log["plates"] += 1
            elif cls == pbl.PERSON_CLS:
                pbl.blur_region(out, x0, y0, x1, y1)
                log["persons"] += 1

    for box, text, conf in reader.readtext(frame, low_text=0.3, text_threshold=0.4):
        if conf < pbl.OCR_CONF:
            continue
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        if (max(xs) - min(xs)) < pbl.OCR_MIN_W:
            continue
        pbl.blur_region(out, min(xs), min(ys), max(xs), max(ys))
        log["texts"] += 1

    for (bx0, by0, bx1, by1) in MANUAL_BLUR.get(manual_key, []):
        pbl.blur_region(out, bx0 * W, by0 * H, bx1 * W, by1 * H)
        log["manual"] += 1
    return out, log


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    from ultralytics import YOLO
    import easyocr
    yolo = YOLO(MODEL)
    reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)

    saved = []
    for tg in TARGETS:
        if not os.path.exists(tg["vid"]):
            print(f"[skip] 원본 없음: {tg['vid']}"); continue
        recs, status = process_window(yolo, tg["vid"], tg["t"], tg["t"])
        r = pbl.pick_rec(recs, tg["t"])
        if r is None:
            print(f"[skip] {tg['out']} 선행차 미검출({status})"); continue
        blurred, log = privacy_blur(r["frame"], yolo, reader, tg["out"])
        pbl.draw_overlay(blurred, r)
        name = f"{tg['out']}.png"
        pbl.downscale_save(blurred, os.path.join(OUTDIR, name))
        saved.append(dict(name=name, rec=r, note=tg["note"]))
        dl = r["dlamp"]
        dl_s = "inf" if dl == float("inf") else f"{dl:.1f}x"
        print(f"[저장] {name}  verdict={bucket_of(r)} lamp={r['lamp']:.1f}% "
              f"dLamp={dl_s} sym={r['sym']}  블러(plate{log['plates']}/person{log['persons']}/"
              f"text{log['texts']}/manual{log['manual']})", flush=True)

    print("\n=== OCR 재검사 (번호판/전화) ===")
    lines = ["# d_brake_detected OCR 재검사 로그", ""]
    total = 0
    for s in saved:
        path = os.path.join(OUTDIR, s["name"])
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        hits = reader.readtext(img, low_text=0.3, text_threshold=0.4)
        plates = [t for _, t, c in hits if c >= 0.05 and pbl.PLATE_RE.search(t)]
        phones = [t for _, t, c in hits if c >= 0.05 and pbl.PHONE_RE.search(t.replace(" ", ""))]
        residual = [t for _, t, c in hits if c >= 0.45 and len(t.strip()) >= 2
                    and not re.fullmatch(r"[\d\s:.\-/%=]+", t)]
        total += len(plates) + len(phones)
        print(f"  [{'위험' if (plates or phones) else '안전'}] {s['name']}  "
              f"plate={plates} phone={phones}")
        lines += [f"## {s['name']}", f"- plate 정규식: {plates or '0건'}",
                  f"- phone 정규식: {phones or '0건'}",
                  f"- 잔여 판독텍스트(육안확인용): {residual[:12]}", ""]
    lines.insert(2, f"번호판/전화 합계: {total}건 ({'이상 없음' if total == 0 else '★확인 필요'})\n")
    with open(os.path.join(OUTDIR, "_OCR_VERIFY.log.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n번호판/전화 합계 {total}건. 저장 {len(saved)}장 → {OUTDIR}")


if __name__ == "__main__":
    main()
