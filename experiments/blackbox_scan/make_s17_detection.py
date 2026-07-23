"""
S17용 '객체 탐지 동작 확인' 프레임 1장 — 선행 차량 YOLO 검출 결과.

★ S18(브레이크등)과 구분: 라벨은 '차량 검출(class conf)'만. red%·급정거·brake 표기 금지.
   오검출 논란을 피하고 S17='파이프라인 동작 확인', S18='브레이크등 검출'을 분리한다.

개인정보: raw 에 먼저 블러(OSD·번호판·인물·상호·지명·앞유리 각인) → 검출 라벨 작도 →
 1280px 축소 → OCR 재검사. B/C·A 승격과 동일 파이프라인.
"""
from __future__ import annotations
import os, re, sys
import cv2, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import promote_brake_limitation as pbl

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "candidates", "selected", "s17_detection")
MODEL = os.path.join(HERE, "yolov8n.pt")
VID = r"C:\Users\이동욱\blackbox_full\NORMAL\20260720-14h42m56s_N.avi"
T = 54.0
DRAW_CONF = 0.45           # 이 이상만 라벨(원거리 저신뢰 오검출 배제)
NAMES = {2: "car", 5: "bus", 7: "truck"}

MANUAL_BLUR = [
    (0.11, 0.53, 0.27, 0.63),   # 선행 적색 세단 후면 번호판(66조 XXXX)
    (0.42, 0.41, 0.55, 0.50),   # 중앙 원거리 녹색 도로표지(지명)
    (0.02, 0.39, 0.10, 0.51),   # 좌측 경고표지(비식별이나 보수적으로)
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

    # 검출 (라벨용) — 원본에서
    res = yolo.predict(frame, conf=0.35, verbose=False, classes=[2, 5, 7])[0]
    dets = []
    for b in res.boxes:
        x0, y0, x1, y1 = [float(v) for v in b.xyxy[0]]
        c = int(b.cls[0]); cf = float(b.conf[0])
        if cf >= DRAW_CONF:
            dets.append((x0, y0, x1, y1, NAMES[c], cf))

    # 개인정보 블러 (raw 먼저)
    out = frame.copy()
    pbl.blur_region(out, 0, 0, W, int(H * pbl.OSD_FRAC))
    pres = yolo.predict(frame, conf=pbl.YOLO_CONF, verbose=False,
                        classes=[pbl.PERSON_CLS, 2, 3, 5, 7])[0]
    np_, nt = 0, 0
    for b in pres.boxes:
        cls = int(b.cls[0]); x0, y0, x1, y1 = [float(v) for v in b.xyxy[0]]
        bw, bh = x1 - x0, y1 - y0
        if cls in pbl.VEHICLE_CLS:
            cx = (x0 + x1) / 2
            pbl.blur_region(out, cx - bw * 0.32, y1 - bh / 3, cx + bw * 0.32, y1)
        elif cls == pbl.PERSON_CLS:
            pbl.blur_region(out, x0, y0, x1, y1); np_ += 1
    for box, text, conf in reader.readtext(frame, low_text=0.3, text_threshold=0.4):
        if conf < pbl.OCR_CONF:
            continue
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        if (max(xs) - min(xs)) < pbl.OCR_MIN_W:
            continue
        pbl.blur_region(out, min(xs), min(ys), max(xs), max(ys)); nt += 1
    for (bx0, by0, bx1, by1) in MANUAL_BLUR:
        pbl.blur_region(out, bx0 * W, by0 * H, bx1 * W, by1 * H)

    # 검출 라벨 작도 (차량만, red%/brake 없음)
    for (x0, y0, x1, y1, name, cf) in dets:
        p0 = (int(x0), int(y0)); p1 = (int(x1), int(y1))
        cv2.rectangle(out, p0, p1, (0, 220, 0), 3)
        label = f"{name} {cf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (p0[0], max(0, p0[1] - th - 8)),
                      (p0[0] + tw + 8, p0[1]), (0, 220, 0), -1)
        cv2.putText(out, label, (p0[0] + 4, max(14, p0[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, "YOLOv8n object detection (vehicle)", (16, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)

    name = "s17_vehicle_detection.png"
    pbl.downscale_save(out, os.path.join(OUTDIR, name))
    print(f"[저장] {name}  검출 {len(dets)}대: {[(d[4], round(d[5],2)) for d in dets]}  "
          f"블러(person{np_}/text{nt}/manual{len(MANUAL_BLUR)})")

    # OCR 재검사
    img = cv2.imdecode(np.fromfile(os.path.join(OUTDIR, name), np.uint8), cv2.IMREAD_COLOR)
    hits = reader.readtext(img, low_text=0.3, text_threshold=0.4)
    plates = [t for _, t, c in hits if c >= 0.05 and pbl.PLATE_RE.search(t)]
    phones = [t for _, t, c in hits if c >= 0.05 and pbl.PHONE_RE.search(t.replace(" ", ""))]
    residual = [t for _, t, c in hits if c >= 0.45 and len(t.strip()) >= 2
                and not re.fullmatch(r"[\d\s:.\-/%=]+", t)]
    print(f"  OCR 재검사: plate={plates} phone={phones}")
    with open(os.path.join(OUTDIR, "_OCR_VERIFY.log.md"), "w", encoding="utf-8") as f:
        f.write(f"# s17_detection OCR 재검사\n\n번호판/전화: "
                f"{len(plates)+len(phones)}건 ({'이상 없음' if not (plates or phones) else '★확인'})\n\n"
                f"- plate: {plates or '0건'}\n- phone: {phones or '0건'}\n"
                f"- 잔여 판독텍스트: {residual[:12]}\n")
    print(f"저장 → {OUTDIR}")


if __name__ == "__main__":
    main()
