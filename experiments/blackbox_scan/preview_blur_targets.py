"""
블러 대상 미리보기 — 박스만 표시, 실제 블러는 적용하지 않는다.

카테고리 (사용자 지정)
  RED    번호판   — OCR 차량번호패턴 확인 + YOLO 차량 하단1/3 휴리스틱(OCR 미확인 대비)
  YELLOW 상호/간판 텍스트 — 상호·전화번호·지명 등 판독된 문자 (OCR 분류)
  BLUE   기타     — OSD 상단 오버레이 띠, 기타 미분류 문자

지금까지 이 프로젝트에서 쓴 방법(audit_privacy.py 의 차량 하단1/3 휴리스틱 +
audit_ocr_face.py 의 OCR 분류)을 그대로 재사용한다.

출력: candidates/final_review/preview/<원본명>_preview.png (박스 오버레이, 블러 없음)
      candidates/final_review/preview/<원본명>_boxes.json (좌표 목록)
"""
from __future__ import annotations
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "candidates", "final_review")
OUT_DIR = os.path.join(SRC_DIR, "preview")
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = [
    "S6_A_stop_t17.0_area0.152.png",
    "S6_C_userpick_t25.0_area0.030.png",
]

VEHICLE = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
OSD_TOP_FRAC = 0.05        # build_selected.py 의 OSD_CROP_FRAC 과 동일 기준

PLATE_RE = re.compile(r"\d{2,3}\s?[가-힣]\s?\d{4}")
PHONE_RE = re.compile(r"(01[016-9]|0\d{1,2})[\s\-.]?\d{3,4}[\s\-.]?\d{4}")
PLACE_KW = ("동", "읍", "면", "리", "로", "길", "사거리", "삼거리", "오거리",
            "육거리", "교차로", "IC", "대교", "터미널", "역", "공항", "시청",
            "군청", "구청")

COLORS = {"plate": (230, 30, 30), "sign": (235, 200, 20), "other": (40, 110, 230)}
LABELS = {"plate": "번호판", "sign": "상호/간판", "other": "기타"}


def classify_text(t: str) -> str:
    t = t.strip()
    if PLATE_RE.search(t):
        return "plate"
    if PHONE_RE.search(t.replace(" ", "")):
        return "sign"          # 전화번호는 상가 텍스트로 묶어 노랑
    if any(k in t for k in PLACE_KW):
        return "sign"          # 지명(도로표지판)도 노랑
    if re.search(r"[가-힣A-Za-z]{2,}", t):
        return "sign"
    return None                # 판정 불가한 잡음은 표시 안 함


def imread_u(path):
    return np.array(Image.open(path).convert("RGB"))


def main():
    import easyocr
    from ultralytics import YOLO
    reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    model = YOLO(os.path.join(HERE, "yolov8n.pt"))

    for name in TARGETS:
        path = os.path.join(SRC_DIR, name)
        if not os.path.exists(path):
            print(f"  [경고] 없음: {name}"); continue
        img_np = imread_u(path)
        H, W = img_np.shape[:2]
        im = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(im, "RGBA")
        try:
            font = ImageFont.truetype("malgun.ttf", 22)
        except Exception:
            font = ImageFont.load_default()

        boxes = []

        # 1) OSD 상단 띠 — 기타(파랑), 항상 고정 표시
        osd_box = (0, 0, W, int(H * OSD_TOP_FRAC))
        boxes.append({"category": "other", "label": "OSD 상단 오버레이",
                     "source": "fixed", "box_px": list(osd_box),
                     "box_frac": [0, 0, 1.0, round(OSD_TOP_FRAC, 3)]})

        # 2) YOLO 차량 검출 → 하단 1/3 = 번호판 후보 (OCR 미확인 대비 휴리스틱)
        res = model.predict(path, conf=0.15, verbose=False,
                            classes=[2, 3, 5, 7])[0]
        for b in res.boxes:
            x0, y0, x1, y1 = [float(v) for v in b.xyxy[0]]
            bh = y1 - y0
            plate_box = (x0, y1 - bh / 3, x1, y1)
            boxes.append({
                "category": "plate", "label": f"번호판후보({VEHICLE.get(int(b.cls[0]),'veh')})",
                "source": "yolo_vehicle_heuristic",
                "conf": round(float(b.conf[0]), 3),
                "box_px": [round(v) for v in plate_box],
                "box_frac": [round(plate_box[0]/W,3), round(plate_box[1]/H,3),
                            round(plate_box[2]/W,3), round(plate_box[3]/H,3)],
            })

        # 3) OCR — 문자 판독 (번호판/상호/전화/지명)
        for box, text, conf in reader.readtext(img_np, low_text=0.3, text_threshold=0.4):
            if conf < 0.05 or not text.strip():
                continue
            cat = classify_text(text)
            if cat is None:
                continue
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            boxes.append({
                "category": cat, "label": text.strip(),
                "source": "ocr", "conf": round(float(conf), 3),
                "box_px": [round(x0), round(y0), round(x1), round(y1)],
                "box_frac": [round(x0/W,3), round(y0/H,3), round(x1/W,3), round(y1/H,3)],
            })

        # 오버레이 그리기 (번호판 우선 표시되도록 plate를 맨 나중에 그림)
        order = {"other": 0, "sign": 1, "plate": 2}
        for b in sorted(boxes, key=lambda b: order[b["category"]]):
            x0, y0, x1, y1 = b["box_px"]
            col = COLORS[b["category"]]
            draw.rectangle([x0, y0, x1, y1], outline=col + (255,), width=3,
                           fill=col + (55,))
            tag = f"{LABELS[b['category']]}"
            if b["source"] == "ocr":
                tag += f":{b['label'][:14]}"
            ty = max(0, y0 - 24)
            tw = draw.textlength(tag, font=font) if hasattr(draw, "textlength") else 8*len(tag)
            draw.rectangle([x0, ty, x0 + tw + 8, ty + 22], fill=(0, 0, 0, 170))
            draw.text((x0 + 4, ty), tag, fill=col + (255,), font=font)

        out_png = os.path.join(OUT_DIR, name.replace(".png", "_preview.png"))
        out_json = os.path.join(OUT_DIR, name.replace(".png", "_boxes.json"))
        im.save(out_png)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"file": name, "width": W, "height": H, "boxes": boxes},
                      f, ensure_ascii=False, indent=2)

        n_plate = sum(1 for b in boxes if b["category"] == "plate")
        n_sign = sum(1 for b in boxes if b["category"] == "sign")
        n_other = sum(1 for b in boxes if b["category"] == "other")
        print(f"  {name}: 번호판{n_plate} 상호/간판{n_sign} 기타{n_other} → {out_png}")


if __name__ == "__main__":
    main()
