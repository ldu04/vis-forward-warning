"""
S6 t17/t25 블러 대상 좌표 v2 — 사용자 피드백 반영, 수기 정밀 조정.

v1(preview_blur_targets.py, YOLO+OCR 자동)에서:
  - OSD 오분류(노랑) 삭제, 좌측 상가 간판(치과 등) 삭제 — 풀네임 노출 없음 확인됨
  - 차량 하단1/3 휴리스틱 박스(느슨함) → 4배 확대 크롭으로 실제 번호판만 타이트하게 재측정
  - "검정 세단 번호판" 자동박스 위치 오류 확인 → OCR이 다른 위치(구 'sign' 오분류 지점)에서
    번호판을 걸러낸 것으로 판명, 그 위치를 번호판(빨강)으로 재지정
  - t25 세단 번호판은 4배 크롭으로 재탐색해 새 좌표 확보(구 박스보다 위쪽, 흰 바탕)

좌표는 전부 4배 확대 크롭(crop_check/*.png)을 육안 실측해 산출했다(자동검출 아님).
출력: candidates/final_review/preview/<이름>_preview_v2.png / _boxes_v2.json
      (v1 파일은 덮어쓰지 않음)
"""
from __future__ import annotations
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "candidates", "final_review")
OUT_DIR = os.path.join(SRC_DIR, "preview")
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {"plate": (230, 30, 30), "sign": (235, 200, 20), "other": (40, 110, 230)}
LABELS = {"plate": "번호판", "sign": "상호/간판", "other": "기타"}

FRAMES = {
    "S6_A_stop_t17.0_area0.152.png": [
        {"category": "other", "label": "OSD 상단 오버레이", "box_px": [0, 0, 1920, 54]},
        {"category": "plate", "label": "번호판(버스, 노란바탕)", "box_px": [890, 522, 1012, 585]},
        {"category": "plate", "label": "번호판(검정세단, 흰바탕) — 위치 재지정", "box_px": [1248, 570, 1352, 608]},
        {"category": "sign", "label": "YONG NAM (버스 상호)", "box_px": [856, 441, 1053, 492]},
    ],
    "S6_C_userpick_t25.0_area0.030.png": [
        {"category": "other", "label": "OSD 상단 오버레이", "box_px": [0, 0, 1920, 54]},
        {"category": "plate", "label": "번호판(버스, 노란바탕)", "box_px": [915, 525, 987, 562]},
        {"category": "plate", "label": "번호판(검정세단, 흰바탕) — 위치 재지정(구박스보다 위쪽)", "box_px": [1075, 543, 1133, 568]},
        {"category": "sign", "label": "YONG NAM (버스 상호)", "box_px": [896, 486, 1004, 510]},
    ],
}


def main():
    for name, boxes in FRAMES.items():
        path = os.path.join(SRC_DIR, name)
        im = Image.open(path).convert("RGB")
        W, H = im.size
        draw = ImageDraw.Draw(im, "RGBA")
        try:
            font = ImageFont.truetype("malgun.ttf", 22)
        except Exception:
            font = ImageFont.load_default()

        for b in boxes:
            b["box_frac"] = [round(b["box_px"][0]/W, 3), round(b["box_px"][1]/H, 3),
                             round(b["box_px"][2]/W, 3), round(b["box_px"][3]/H, 3)]
            b["source"] = "manual_v2_4x_crop_measured"

        order = {"other": 0, "sign": 1, "plate": 2}
        for b in sorted(boxes, key=lambda b: order[b["category"]]):
            x0, y0, x1, y1 = b["box_px"]
            col = COLORS[b["category"]]
            draw.rectangle([x0, y0, x1, y1], outline=col + (255,), width=3, fill=col + (55,))
            tag = f"{LABELS[b['category']]}:{b['label'][:20]}"
            ty = max(0, y0 - 24)
            tw = draw.textlength(tag, font=font)
            draw.rectangle([x0, ty, x0 + tw + 8, ty + 22], fill=(0, 0, 0, 170))
            draw.text((x0 + 4, ty), tag, fill=col + (255,), font=font)

        out_png = os.path.join(OUT_DIR, name.replace(".png", "_preview_v2.png"))
        out_json = os.path.join(OUT_DIR, name.replace(".png", "_boxes_v2.json"))
        im.save(out_png)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"file": name, "width": W, "height": H, "boxes": boxes},
                      f, ensure_ascii=False, indent=2)
        print(f"  {name}: 번호판{sum(1 for b in boxes if b['category']=='plate')} "
              f"상호{sum(1 for b in boxes if b['category']=='sign')} "
              f"기타{sum(1 for b in boxes if b['category']=='other')} → {out_png}")


if __name__ == "__main__":
    main()
