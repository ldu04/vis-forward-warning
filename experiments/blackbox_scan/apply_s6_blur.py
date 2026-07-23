"""
S6 t17/t25 실제 블러 적용 (v2 좌표 기준).

블러 방식은 프로젝트 기존 관례(build_selected.py 의 blur_box)와 동일:
  극단 다운스케일(1/22) → NEAREST 업스케일 → 가우시안(radius=6)
  → 픽셀 정보 자체를 파괴한 뒤 경계를 흐려 원복 불가능하게 만든다.

대상: v2 박스 전체(번호판·상호·OSD) — OSD도 프로젝트 전체 관례상 항상 마스킹 대상이었다.
strength 인자로 반복 시 강도를 올릴 수 있게 한다(4단계 재검증 실패 시 대비).

출력: candidates/selected/S6_A_t17_blurred_final.png
      candidates/selected/S6_C_t25_blurred_final.png
      (원본 및 v2 미리보기는 절대 덮어쓰지 않음)
"""
from __future__ import annotations
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "candidates", "final_review")
PREVIEW_DIR = os.path.join(SRC_DIR, "preview")
DST_DIR = os.path.join(HERE, "candidates", "selected")
os.makedirs(DST_DIR, exist_ok=True)

JOBS = [
    ("S6_A_stop_t17.0_area0.152.png", "S6_A_stop_t17.0_area0.152_boxes_v2.json",
     "S6_A_t17_blurred_final.png"),
    ("S6_C_userpick_t25.0_area0.030.png", "S6_C_userpick_t25.0_area0.030_boxes_v2.json",
     "S6_C_t25_blurred_final.png"),
]


def blur_box(im: Image.Image, box, downscale=22, radius=6):
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(im.width, x1), min(im.height, y1)
    if x1 <= x0 or y1 <= y0:
        return
    region = im.crop((x0, y0, x1, y1))
    w, h = region.size
    small = region.resize((max(1, w // downscale), max(1, h // downscale)), Image.BILINEAR)
    region = small.resize((w, h), Image.NEAREST).filter(ImageFilter.GaussianBlur(radius))
    im.paste(region, (x0, y0))


def run(downscale=22, radius=6, suffix=""):
    results = []
    for src_name, json_name, dst_name in JOBS:
        src = os.path.join(SRC_DIR, src_name)
        boxes = json.load(open(os.path.join(PREVIEW_DIR, json_name), encoding="utf-8"))["boxes"]
        im = Image.open(src).convert("RGB")
        for b in boxes:
            blur_box(im, b["box_px"], downscale=downscale, radius=radius)
        out_name = dst_name if not suffix else dst_name.replace(".png", f"_{suffix}.png")
        out = os.path.join(DST_DIR, out_name)
        im.save(out)
        print(f"  [저장] {out}  (박스 {len(boxes)}개 블러, downscale={downscale} radius={radius})")
        results.append(out)
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--downscale", type=int, default=22)
    ap.add_argument("--radius", type=int, default=6)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    run(args.downscale, args.radius, args.suffix)
