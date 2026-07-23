"""
저장소 커밋 이미지 전체 OCR 재스캔 — 번호판/전화번호 패턴 검출만.
발견돼도 삭제·수정하지 않는다. 목록만 만든다.
"""
import re, sys, os, json, subprocess
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from PIL import Image

PLATE_RE = re.compile(r"\d{2,3}\s?[가-힣]\s?\d{4}")
PHONE_RE = re.compile(r"(01[016-9]|0\d{1,2})[\s\-.]?\d{3,4}[\s\-.]?\d{4}")

REPO = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True,
                      text=True, encoding="utf-8", errors="replace").stdout.strip()

def tracked_images():
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", cwd=REPO).stdout.splitlines()
    return [f for f in out if re.search(r"\.(png|jpe?g)$", f, re.I)]

def main():
    import easyocr
    reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    files = tracked_images()
    print(f"대상 {len(files)}개 커밋 이미지 전수 스캔")
    findings = []
    for f in files:
        path = os.path.join(REPO, f)
        try:
            img = np.array(Image.open(path).convert("RGB"))
        except Exception as e:
            print(f"  [경고] 열기 실패 {f}: {e}")
            continue
        hits = reader.readtext(img, low_text=0.3, text_threshold=0.4)
        plate = [t for _, t, c in hits if c >= 0.05 and PLATE_RE.search(t)]
        phone = [t for _, t, c in hits if c >= 0.05 and PHONE_RE.search(t.replace(" ", ""))]
        status = "위험" if (plate or phone) else "안전"
        print(f"  [{status}] {f}  plate={plate} phone={phone}")
        if plate or phone:
            findings.append({"file": f, "plate": plate, "phone": phone})
    with open(os.path.join(REPO, "experiments/blackbox_scan/candidates/final_review/_repo_ocr_scan.json"),
             "w", encoding="utf-8") as fp:
        json.dump({"scanned": len(files), "findings": findings}, fp, ensure_ascii=False, indent=2)
    print(f"\n스캔 {len(files)}개, 위험 {len(findings)}건")

if __name__ == "__main__":
    main()
