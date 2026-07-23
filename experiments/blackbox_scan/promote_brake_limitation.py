"""
rebuild B/C 대표 프레임을 개인정보 처리해 selected/d_brake_limitation/ 로 승격.

절차(프레임마다)
1. 원본 영상에서 t_peak 의 raw 프레임을 재추출하고, 커밋된 rebuild_brake_detect 로직으로
   동일 검출 레코드(box/roi/lamp쌍/naive·lamp·dLamp·sym·TTC·verdict)를 재계산한다.
2. **raw 프레임에 먼저 개인정보 블러**를 적용한다(오버레이 아래 개인정보가 남지 않게):
   - 상단 OSD 띠 블러
   - YOLO(person·car·bus·truck·motorcycle, 저임계) → 차량 번호판부(하단1/3 중앙60%) +
     사람 전체 bbox 블러
   - OCR(ko+en) → 판독 가능한 모든 텍스트(상호·지명·번호판문자·전화) 블러
3. 블러된 프레임 위에 1의 레코드로 동일 오버레이를 다시 그린다.
4. 긴 변 1280px 로 축소 저장(저장소 표준).
5. 전량 저장 후 OCR 전수 재검사(번호판/전화 정규식) → 0건 확인, 로그 저장.

블러는 되돌릴 수 없는 파괴(축소→NEAREST 확대→가우시안). 원본 영상/PNG 는 건드리지 않는다.
출력 폴더는 selected/ (gitignore 예외 = 커밋 대상)이므로 처리를 보수적으로 한다.
"""
from __future__ import annotations

import os
import re
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from rebuild_brake_detect import (
    process_window, bucket_of, imwrite_unicode, naive_verdict,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "candidates", "selected", "d_brake_limitation")
MODEL = os.path.join(HERE, "yolov8n.pt")
LONG_EDGE = 1280

OSD_FRAC = 0.055           # 상단 OSD 띠 블러 비율
YOLO_CONF = 0.05           # 낮게 — 놓치는 것보다 과검출
OCR_CONF = 0.30
OCR_MIN_W = 14             # 이보다 좁은 텍스트는 애초에 판독 불가 → 건너뜀
VEHICLE_CLS = {2, 3, 5, 7}
PERSON_CLS = 0

PLATE_RE = re.compile(r"\d{2,3}\s?[가-힣]\s?\d{4}")
PHONE_RE = re.compile(r"(01[016-9]|0\d{1,2})[\s\-.]?\d{3,4}[\s\-.]?\d{4}")

# 자동 블러(YOLO+OCR)가 놓치는 항목의 수동 백스톱 — out 이름 키, 프레임 대비 비율 좌표.
# 육안 검수로 확인된 잔여 개인정보(스타일 로고 상호·광고 인물 얼굴·차체 상호)를 보수적으로 가린다.
MANUAL_BLUR = {
    "d01_redsportscar_naive-hi_lamp00": [
        (0.11, 0.29, 0.34, 0.43),   # KB손해보험 상호 밴드(좌 건물)
        (0.25, 0.43, 0.40, 0.54),   # 중앙 건물 유리문 영문 상호
        (0.35, 0.74, 0.49, 0.86),   # 앞유리 각인 번호
    ],
    # 간판 밀집 도심 교차로 — 사용자 필수요청 프레임이라 교체 대신 '키홀' 블러로 처리:
    # 중앙 버스+도로 창만 남기고 상단 전폭·좌우 여백을 통째로 파괴한다(모든 간판·포스터·측면차 포함).
    "d02_redbus_bright-fooled_corrected": [
        (0.00, 0.00, 1.00, 0.43),   # 상단 전폭(모든 건물 간판·지명·광고 포스터·인물)
        (0.00, 0.43, 0.20, 1.00),   # 좌측 여백 컬럼
        (0.66, 0.43, 1.00, 1.00),   # 우측 여백 컬럼
        (0.20, 0.43, 0.285, 0.53),  # 버스 좌측 포켓(약국 등 노면 간판)
        (0.51, 0.43, 0.66, 0.53),   # 버스 우측 포켓(예산치과의원 등 지명·상호)
        (0.35, 0.74, 0.49, 0.86),   # 앞유리 각인 번호
    ],
    "d03_chungnam_bus_naive-mid_lamp01": [
        (0.38, 0.26, 0.54, 0.42),   # 선행 버스 후면 상호(충남고속/우등)
        (0.35, 0.74, 0.49, 0.86),   # 앞유리 각인 번호
    ],
}

# 승격 대상 (C 는 유일 후보가 검출 아티팩트[전폭 허위 bbox]라 제외 — 보고서에 사유 명기)
TARGETS = [
    dict(out="d01_redsportscar_naive-hi_lamp00", label="B",
         vid=r"C:\Users\이동욱\blackbox_full\NORMAL\20260719-17h38m27s_N.avi", t=56.6,
         note="빨간 스포츠카 — naive 는 detected, lamp 0% → 단순 방식만 속은 전형적 오검출"),
    dict(out="d02_redbus_bright-fooled_corrected", label="B",
         vid=r"C:\Users\이동욱\blackbox_full\EVENT\20260719-08h32m05s_E.avi", t=16.9,
         note="빨간 시내버스 — 밝기 필터(lamp%)까지 속지만 시간(dLamp)·대칭(sym)에서 정정"),
    dict(out="d03_chungnam_bus_naive-mid_lamp01", label="B",
         vid=r"C:\Users\이동욱\blackbox_full\EVENT\20260719-08h36m31s_E.avi", t=13.4,
         note="선행 우등버스(내 차선 정면) — naive 중간(주황 차체), lamp 0.9% → 점등 아님으로 정정"),
]


def blur_region(bgr, x0, y0, x1, y1):
    h, w = bgr.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    roi = bgr[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    small = cv2.resize(roi, (max(1, rw // 22), max(1, rh // 22)), interpolation=cv2.INTER_LINEAR)
    up = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    up = cv2.GaussianBlur(up, (0, 0), 6)
    bgr[y0:y1, x0:x1] = up


def privacy_blur(frame, yolo, reader, manual_key=None):
    """raw 프레임(복사본)에 개인정보 블러 적용. 반환: (블러본, 처리내역 dict)."""
    out = frame.copy()
    H, W = out.shape[:2]
    log = {"osd": 1, "plates": 0, "persons": 0, "texts": 0, "manual": 0}

    # 1) 상단 OSD 띠
    blur_region(out, 0, 0, W, int(H * OSD_FRAC))

    # 2) YOLO 차량 번호판부 + 사람 전체
    res = yolo.predict(frame, conf=YOLO_CONF, verbose=False,
                       classes=[PERSON_CLS, 2, 3, 5, 7])[0]
    if res.boxes is not None:
        for b in res.boxes:
            cls = int(b.cls[0])
            x0, y0, x1, y1 = [float(v) for v in b.xyxy[0]]
            bw, bh = x1 - x0, y1 - y0
            if cls in VEHICLE_CLS:
                cx = (x0 + x1) / 2
                blur_region(out, cx - bw * 0.32, y1 - bh / 3, cx + bw * 0.32, y1)
                log["plates"] += 1
            elif cls == PERSON_CLS:
                blur_region(out, x0, y0, x1, y1)  # 사람은 전체 블러(초상권)
                log["persons"] += 1

    # 3) OCR 텍스트 전량(상호·지명·번호판문자·전화)
    for box, text, conf in reader.readtext(frame, low_text=0.3, text_threshold=0.4):
        if conf < OCR_CONF:
            continue
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        if (max(xs) - min(xs)) < OCR_MIN_W:
            continue
        blur_region(out, min(xs), min(ys), max(xs), max(ys))
        log["texts"] += 1

    # 4) 수동 백스톱(육안 확인된 잔여 항목)
    for (bx0, by0, bx1, by1) in MANUAL_BLUR.get(manual_key, []):
        blur_region(out, bx0 * W, by0 * H, bx1 * W, by1 * H)
        log["manual"] += 1

    return out, log


def draw_overlay(vis, r):
    x1, y1, x2, y2 = [int(v) for v in r["box"]]
    rx1, ry1, rx2, ry2 = [int(v) for v in r["roi_box"]]
    b = bucket_of(r)
    col = {"A": (0, 0, 255), "B": (0, 165, 255), "C": (160, 160, 160)}.get(b, (200, 200, 200))
    cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
    ox, oy = r["roi_origin"]
    for (lx, ly, lw, lh) in r["pair_boxes"]:
        cv2.rectangle(vis, (ox + lx, oy + ly), (ox + lx + lw, oy + ly + lh), (0, 255, 0), 2)
    ttc = f"{r['ttc']:.2f}s" if r["ttc"] is not None else "-"
    dl = "inf" if r["dlamp"] == float("inf") else f"{r['dlamp']:.1f}x"
    l1 = (f"naive={r['naive']:.1f}%({naive_verdict(r)})  lamp={r['lamp']:.1f}%  "
          f"dLamp={dl}  sym={'Y' if r['sym'] else 'N'}  TTC={ttc}")
    l2 = (f"IMPROVED verdict={b}  brake_on={'Y' if r['brake_on'] else 'N'}  "
          f"looming={'Y' if r['looming'] else 'N'}")
    cv2.rectangle(vis, (x1, max(0, y1 - 56)), (min(vis.shape[1], x1 + 720), y1), col, -1)
    cv2.putText(vis, l1, (x1 + 6, max(20, y1 - 34)), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, l2, (x1 + 6, max(42, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)


def downscale_save(vis, path):
    h, w = vis.shape[:2]
    s = min(1.0, LONG_EDGE / max(h, w))
    if s < 1.0:
        vis = cv2.resize(vis, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    imwrite_unicode(path, vis)


def pick_rec(recs, t):
    return min(recs, key=lambda r: abs(r["t"] - t)) if recs else None


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    from ultralytics import YOLO
    import easyocr
    yolo = YOLO(MODEL)
    reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)

    saved = []
    for tg in TARGETS:
        if not os.path.exists(tg["vid"]):
            print(f"[skip] 원본 없음: {tg['vid']}")
            continue
        recs, status = process_window(yolo, tg["vid"], tg["t"], tg["t"])
        r = pick_rec(recs, tg["t"])
        if r is None:
            print(f"[skip] {tg['out']} 선행차 미검출 ({status})")
            continue
        blurred, log = privacy_blur(r["frame"], yolo, reader, manual_key=tg["out"])
        draw_overlay(blurred, r)
        name = f"{tg['out']}.png"
        downscale_save(blurred, os.path.join(OUTDIR, name))
        saved.append(dict(name=name, rec=r, log=log, note=tg["note"]))
        print(f"[저장] {name}  verdict={bucket_of(r)} naive={r['naive']:.1f}% "
              f"lamp={r['lamp']:.1f}%  블러(plate{log['plates']}/person{log['persons']}/"
              f"text{log['texts']}/manual{log['manual']})", flush=True)

    # OCR 전수 재검사 (번호판/전화 정규식) — 최종 축소본 대상
    print("\n=== OCR 재검사 (번호판/전화) ===")
    veri_lines = ["# d_brake_limitation OCR 재검사 로그", ""]
    total_hit = 0
    for s in saved:
        path = os.path.join(OUTDIR, s["name"])
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        hits = reader.readtext(img, low_text=0.3, text_threshold=0.4)
        plates = [t for _, t, c in hits if c >= 0.05 and PLATE_RE.search(t)]
        phones = [t for _, t, c in hits if c >= 0.05 and PHONE_RE.search(t.replace(" ", ""))]
        # 참고: 판독 가능한 잔여 텍스트(상호·지명 육안 확인용) 목록
        residual = [t for _, t, c in hits if c >= 0.45 and len(t.strip()) >= 2
                    and not re.fullmatch(r"[\d\s:.\-/%=]+", t)]
        status = "위험" if (plates or phones) else "안전"
        total_hit += len(plates) + len(phones)
        print(f"  [{status}] {s['name']}  plate={plates} phone={phones}")
        veri_lines += [f"## {s['name']}", f"- plate 정규식: {plates or '0건'}",
                       f"- phone 정규식: {phones or '0건'}",
                       f"- 잔여 판독텍스트(육안확인용): {residual[:12]}", ""]
    veri_lines.insert(2, f"번호판/전화 검출 합계: {total_hit}건 "
                         f"({'이상 없음' if total_hit == 0 else '★확인 필요'})\n")
    with open(os.path.join(OUTDIR, "_OCR_VERIFY.log.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(veri_lines))
    print(f"\n번호판/전화 합계 {total_hit}건. 로그: {os.path.join(OUTDIR, '_OCR_VERIFY.log.md')}")
    print(f"저장 {len(saved)}장 → {OUTDIR}")


if __name__ == "__main__":
    main()
