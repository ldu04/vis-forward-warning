"""
S17용 '전방(선행) 차량 검출' 프레임 — ego-lane 선행차 강조본.

기존 s17_vehicle_detection.png(선행차가 옆차선이던 것)을 지우지 않고, ego-lane 선행차
(bbox cx 중앙 ±15%, 근접·최대)가 깨끗이 잡힌 프레임을 새 파일로 추가한다.
라벨은 '차량 검출(class conf)'만 — red%·급정거·brake 금지.

프레임: 20260720-11h49m11s_N t24.2s — 정면 ego-lane 선행 세단(cx 0.49), 옆차선 버스와 구분됨.
개인정보: raw 먼저 블러(OSD·번호판·상호·지명·인물·앞유리 각인) → 라벨 작도 → 1280 축소 → OCR 재검사.
"""
from __future__ import annotations
import os, re, sys
import cv2, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import promote_brake_limitation as pbl
from rebuild_brake_detect import leading_vehicle_box, ANY_VEHICLE

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "candidates", "selected", "s17_detection")
MODEL = os.path.join(HERE, "yolov8n.pt")
VID = r"C:\Users\이동욱\blackbox_full\NORMAL\20260720-11h49m11s_N.avi"
T = 24.2
NAMES = {2: "car", 5: "bus", 7: "truck"}
OUTNAME = "s17_lead_vehicle_detection.png"

MANUAL_BLUR = [
    (0.43, 0.52, 0.57, 0.64),   # 선행 세단 후면 번호판(...3417)
    (0.19, 0.52, 0.31, 0.61),   # 좌측 버스 노란 번호판
    (0.16, 0.32, 0.43, 0.53),   # 좌측 버스 상호(고속관광 텍스트대)
    (0.35, 0.74, 0.50, 0.87),   # 앞유리 각인 번호
]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    from ultralytics import YOLO
    import easyocr
    yolo = YOLO(MODEL)
    reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)

    cap = cv2.VideoCapture(VID)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(T * fps))
    ok, frame = cap.read(); cap.release()
    if not ok:
        print("frame read fail"); return
    H, W = frame.shape[:2]

    # ego-lane 선행차 + class/conf
    res = yolo.predict(frame, imgsz=640, conf=0.30, verbose=False,
                       classes=list(ANY_VEHICLE))[0]
    lead = leading_vehicle_box(res, W, H)
    if lead is None:
        print("선행차 미검출"); return
    lx0, ly0, lx1, ly1 = [float(v) for v in lead]
    # 매칭되는 검출의 class/conf 회수(IoU 최대)
    best_cls, best_cf, best_iou = 2, 0.0, 0.0
    for b in res.boxes:
        x0, y0, x1, y1 = [float(v) for v in b.xyxy[0]]
        ix0, iy0 = max(x0, lx0), max(y0, ly0); ix1, iy1 = min(x1, lx1), min(y1, ly1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        ua = (x1-x0)*(y1-y0) + (lx1-lx0)*(ly1-ly0) - inter
        iou = inter / ua if ua > 0 else 0
        if iou > best_iou:
            best_iou, best_cls, best_cf = iou, int(b.cls[0]), float(b.conf[0])

    # 개인정보 블러 (raw 먼저)
    out = frame.copy()
    pbl.blur_region(out, 0, 0, W, int(H * pbl.OSD_FRAC))
    pres = yolo.predict(frame, conf=pbl.YOLO_CONF, verbose=False,
                        classes=[pbl.PERSON_CLS, 2, 3, 5, 7])[0]
    npn, nt = 0, 0
    for b in pres.boxes:
        cls = int(b.cls[0]); x0, y0, x1, y1 = [float(v) for v in b.xyxy[0]]
        bw, bh = x1 - x0, y1 - y0
        if cls in pbl.VEHICLE_CLS:
            cx = (x0 + x1) / 2
            pbl.blur_region(out, cx - bw * 0.32, y1 - bh / 3, cx + bw * 0.32, y1)
        elif cls == pbl.PERSON_CLS:
            pbl.blur_region(out, x0, y0, x1, y1); npn += 1
    for box, text, conf in reader.readtext(frame, low_text=0.3, text_threshold=0.4):
        if conf < pbl.OCR_CONF:
            continue
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        if (max(xs) - min(xs)) < pbl.OCR_MIN_W:
            continue
        pbl.blur_region(out, min(xs), min(ys), max(xs), max(ys)); nt += 1
    for (bx0, by0, bx1, by1) in MANUAL_BLUR:
        pbl.blur_region(out, bx0 * W, by0 * H, bx1 * W, by1 * H)

    # 검출된 차량 전부 박스: 선행차=초록 굵게 "(lead)", 나머지=회색 얇게.
    def iou_to_lead(x0, y0, x1, y1):
        ix0, iy0 = max(x0, lx0), max(y0, ly0); ix1, iy1 = min(x1, lx1), min(y1, ly1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        ua = (x1-x0)*(y1-y0) + (lx1-lx0)*(ly1-ly0) - inter
        return inter / ua if ua > 0 else 0

    def put_label(p0, text, box_col, txt_col):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (p0[0], max(0, p0[1] - th - 7)),
                      (p0[0] + tw + 6, p0[1]), box_col, -1)
        cv2.putText(out, text, (p0[0] + 3, max(13, p0[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_col, 2, cv2.LINE_AA)

    # 비선행 차량(회색) 먼저
    for b in res.boxes:
        x0, y0, x1, y1 = [float(v) for v in b.xyxy[0]]
        cls = int(b.cls[0]); cf = float(b.conf[0])
        if cf < 0.40 or iou_to_lead(x0, y0, x1, y1) > 0.5:
            continue
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (170, 170, 170), 2)
        put_label((int(x0), int(y0)), f"{NAMES.get(cls,'veh')} {cf:.2f}",
                  (170, 170, 170), (0, 0, 0))
    # 선행차(초록 굵게)
    p0 = (int(lx0), int(ly0)); p1 = (int(lx1), int(ly1))
    cv2.rectangle(out, p0, p1, (0, 220, 0), 3)
    put_label(p0, f"{NAMES.get(best_cls,'car')} {best_cf:.2f} (lead)", (0, 220, 0), (0, 0, 0))
    cv2.putText(out, "YOLOv8n object detection (vehicles) - ego-lane lead highlighted",
                (16, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 0), 2, cv2.LINE_AA)

    pbl.downscale_save(out, os.path.join(OUTDIR, OUTNAME))
    print(f"[저장] {OUTNAME}  lead={NAMES.get(best_cls)} {best_cf:.2f} "
          f"cx={(lx0+lx1)/2/W:.2f}  블러(person{npn}/text{nt}/manual{len(MANUAL_BLUR)})")

    img = cv2.imdecode(np.fromfile(os.path.join(OUTDIR, OUTNAME), np.uint8), cv2.IMREAD_COLOR)
    hits = reader.readtext(img, low_text=0.3, text_threshold=0.4)
    plates = [t for _, t, c in hits if c >= 0.05 and pbl.PLATE_RE.search(t)]
    phones = [t for _, t, c in hits if c >= 0.05 and pbl.PHONE_RE.search(t.replace(" ", ""))]
    residual = [t for _, t, c in hits if c >= 0.45 and len(t.strip()) >= 2
                and not re.fullmatch(r"[\d\s:.\-/%=()]+", t)]
    print(f"  OCR 재검사: plate={plates} phone={phones}")
    with open(os.path.join(OUTDIR, "_OCR_VERIFY_lead.log.md"), "w", encoding="utf-8") as f:
        f.write(f"# s17_lead OCR 재검사\n\n번호판/전화: {len(plates)+len(phones)}건 "
                f"({'이상 없음' if not (plates or phones) else '★확인'})\n\n"
                f"- plate: {plates or '0건'}\n- phone: {phones or '0건'}\n"
                f"- 잔여 판독텍스트: {residual[:12]}\n")
    print(f"저장 → {OUTDIR}")


if __name__ == "__main__":
    main()
