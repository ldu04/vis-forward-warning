"""
S6 블러 결과 OCR 재검증. 번호판 패턴/상호 텍스트가 다시 읽히면 실패.
반환 코드: 0=통과, 1=재검증 실패(재블러 필요)
"""
from __future__ import annotations
import re, sys, os, json
sys.stdout.reconfigure(encoding="utf-8")

PLATE_RE = re.compile(r"\d{2,3}\s?[가-힣]\s?\d{4}")
TARGET_TEXT = ("YONG", "NAM", "용남")   # 상호 재검출 감시 대상

def main(paths):
    import easyocr
    import numpy as np
    from PIL import Image
    reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    all_ok = True
    report = []
    for p in paths:
        img = np.array(Image.open(p).convert("RGB"))
        hits = reader.readtext(img, low_text=0.3, text_threshold=0.4)
        plate_hits = [t for _, t, c in hits if c >= 0.05 and PLATE_RE.search(t)]
        sign_hits = [t for _, t, c in hits if c >= 0.05 and any(k in t.upper() for k in TARGET_TEXT)]
        ok = not plate_hits and not sign_hits
        all_ok = all_ok and ok
        report.append({"file": os.path.basename(p), "ok": ok,
                       "plate_hits": plate_hits, "sign_hits": sign_hits})
        print(f"  {os.path.basename(p)}: {'PASS' if ok else 'FAIL'} "
              f"plate={plate_hits} sign={sign_hits}")
    return all_ok, report

if __name__ == "__main__":
    ok, report = main(sys.argv[1:])
    with open("candidates/selected/_s6_verify_last.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    sys.exit(0 if ok else 1)
