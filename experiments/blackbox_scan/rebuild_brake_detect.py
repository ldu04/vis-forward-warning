"""
급정거(브레이크등) 다중 단서 재추출 — S11(급정거 예시)·S18(검출 성공/한계 대조)용.

규칙 기반 '다중 단서' 검출기다(딥러닝 SOTA 아님). 문안에는 '검증'이 아니라
'동작 확인/개선'으로 쓴다. looming·점등은 '위험 단서 검출'이지 '사고 발생'이 아니며,
실사 결론(급감속 CRITICAL 0건)과 모순되지 않는다.

기존 자산 재사용
----------------
- eval_brake_lights_v2.py 의 leading_vehicle_box(중앙±15%·면적최대 선행차 선정),
  region_red_ratio, imwrite_unicode, 상수(CENTER_TOL/MIN_AREA_RATIO/ANY_VEHICLE)를 import.
- 기존 eval_brake_lights.py / brake_rank.py 는 건드리지 않는다(덮어쓰기 금지).

파이프라인
----------
1. 선행차 = YOLO car/bus/truck 중 cx 중앙 ±15% & 면적 최대 1대. centroid+IoU 간이추적으로
   프레임 간 동일 선행차 연결(dLamp·looming 의 시간 단서가 같은 차량을 보게 함).
2. 브레이크등 다중 단서 (ROI = 선행차 bbox 후미 하단 45%):
   - naive_red% : ROI 붉은픽셀 비율(기존 단순 방식, 대조용으로만)
   - bright mask: H∈[0,10]∪[170,180] & S≥S_CUT & V≥V_CUT → 점등 램프 후보(차체 빨강 제외)
   - lamp%      : bright mask 면적 / ROI 면적
   - sym        : bright 성분 중 ROI 수평중심 기준 좌우 대칭 2개(+CHMSL 가점)
   - dLamp      : 추적된 선행차 lamp% 를 직전 baseline 대비 배수 급증 & 지속 → onset
   - brake_on = (lamp% ≥ THR_LAMP) AND (sym OR onset)
3. 운동학: 추적 선행차 bbox 면적 증가율 → 접근속도 근사 → TTC. 면적 급증(≥AREA_SURGE)
   또는 TTC ≤ TTC_CRIT → looming.
4. risk = brake_on OR looming (둘 다면 confidence=high).
5. 버킷: A(점등)=brake_on / B(오검출대조)=naive 높은데 bright·sym 낮음(빨간차체) /
   C(근접 비점등)=선행차 근접인데 A 아님. 버킷당 상위 N장 저장(클립당 최대 MAX_PER_CLIP).
6. 오버레이: 선행차 bbox + 램프쌍 작은 박스 + 텍스트(naive/lamp/dLamp/sym/TTC + naive판정
   vs 개선판정 둘 다). 출력: candidates/rebuild_<date>/{A,B,C}/ (새 폴더).
7. 개인정보: 이 스크립트는 오버레이 '원본'까지만. 블러는 selected 이동 시 별도 파이프라인.
   미처리본은 gitignore 유지.

임계값은 영상마다 튜닝 필요 → 상단 상수로 분리. 초기값:
  lamp%≥4%, S≥110, V≥150, dLamp≥2x, 면적급증≥20%.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import deque

import cv2
import numpy as np

from eval_brake_lights_v2 import (
    ANY_VEHICLE, CENTER_TOL, MIN_AREA_RATIO,
    leading_vehicle_box, region_red_ratio, imwrite_unicode,
)

# ─────────────────── 튜닝 상수 (영상별 조정 지점) ───────────────────
THR_LAMP = 0.04       # lamp% 하한 임계 (ROI 대비 밝은적색 비율) — 4%
LAMP_MAX = 0.20       # lamp% 상한: 이보다 크면 램프가 아니라 붉은 차체/패널로 간주 → A 제외
MAX_LAMP_COMP_FRAC = 0.06  # 대칭쌍 각 성분 면적이 ROI 의 이 비율 미만이어야 '램프'(작고 컴팩트)
S_CUT, V_CUT = 110, 150   # 밝은적색 HSV 채도·명도 하한 (차체 빨강 배제)
DLAMP_MULT = 2.0      # onset: lamp% ≥ baseline × 이 배수
AREA_SURGE = 0.20     # looming: 면적 증가율 ≥ 20% (짧은 창)
TTC_CRIT = 2.0        # looming: TTC ≤ 2.0s (S12 CRITICAL 등급 재사용)
NAIVE_THR = 0.08      # B 버킷: naive_red% 가 이 이상인데 점등 아님
NEAR_AREA = 0.06      # C 버킷: 선행차 면적비 ≥ 이 값이면 '근접'
ROI_Y_FRAC = 0.45     # 선행차 bbox 후미 하단 비율
BASE_N = 4            # dLamp baseline = 직전 N 샘플 평균
BASE_FLOOR = 0.8      # baseline 이 이보다 작으면(%) 배수 비교 대신 절대증가로 대체
IOU_TRACK = 0.3       # 프레임 간 동일 선행차로 볼 IoU 하한
STEP_NATIVE = 3       # 원본 프레임 샘플링 간격
PREROLL_S = 1.5       # 후보 t_start 이전 baseline 확보용 선행 구간(초)
MIN_COMP_AREA = 6     # 램프 성분 최소 픽셀
SYM_TOL = 0.18        # 좌우 대칭 허용오차(ROI 폭 비율)
TOPN_PER_BUCKET = 8   # 버킷당 저장 상위 장수
MAX_PER_CLIP = 2      # 한 클립에서 한 버킷에 담을 최대 장수


def bright_red_mask(bgr_region):
    if bgr_region is None or bgr_region.size == 0:
        return None
    hsv = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, S_CUT, V_CUT), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, S_CUT, V_CUT), (180, 255, 255))
    return cv2.bitwise_or(m1, m2)


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def lamp_pair_boxes(mask):
    """좌우 대칭 램프쌍이 있으면 (True, [(x,y,w,h)*2]) 반환. CHMSL 별도.
    실제 램프는 작고 컴팩트 → ROI 대비 너무 큰 성분(붉은 차체 패널)은 램프로 보지 않는다."""
    if mask is None:
        return False, [], False
    h, w = mask.shape[:2]
    roi_area = float(h * w)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < MIN_COMP_AREA:
            continue
        if a > MAX_LAMP_COMP_FRAC * roi_area:
            continue  # 너무 큰 성분 = 붉은 차체 패널, 램프 아님
        comps.append((cent[i][0], cent[i][1],
                      stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], a))
    cx_mid = w / 2.0
    # CHMSL: 상단 30% 중앙 ±15%
    chmsl = any(cyc < 0.30 * h and abs(cxc - cx_mid) <= 0.15 * w
                for cxc, cyc, *_ in comps)
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            x1c, y1c, l1, t1, w1, h1, a1 = comps[i]
            x2c, y2c, l2, t2, w2, h2, a2 = comps[j]
            if not ((x1c < cx_mid <= x2c) or (x2c < cx_mid <= x1c)):
                continue
            d1, d2 = abs(x1c - cx_mid), abs(x2c - cx_mid)
            if abs(d1 - d2) > SYM_TOL * w:
                continue
            if abs(y1c - y2c) > 0.30 * h:
                continue
            amax = max(a1, a2)
            if amax > 0 and min(a1, a2) / amax < 0.35:
                continue  # 면적이 너무 다르면 대칭쌍 아님
            return True, [(l1, t1, w1, h1), (l2, t2, w2, h2)], chmsl
    return False, [], chmsl


def estimate_ttc(area_now, area_prev, dt):
    """sqrt(area) 선형확대 기반 단안 TTC 근사. 접근중 아니면 None."""
    if area_prev is None or dt is None or dt <= 0:
        return None
    s_now, s_prev = np.sqrt(area_now), np.sqrt(area_prev)
    ds = s_now - s_prev
    if ds <= 1e-9:
        return None
    return float(s_now * dt / ds)


def process_window(model, vid, t0, t1):
    """후보 창을 훑어 프레임별 단서 레코드 리스트 반환(선행차 있는 프레임만)."""
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        return [], "open_failed"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_f = max(0, int((t0 - PREROLL_S) * fps))
    end_f = min(total - 1, int(t1 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    lamp_hist = deque(maxlen=BASE_N)
    elevated_prev = False
    prev_box, prev_area, prev_t = None, None, None
    recs = []
    fidx = start_f
    while fidx <= end_f:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if (fidx - start_f) % STEP_NATIVE != 0:
            fidx += 1
            continue
        H, W = frame.shape[:2]
        t_now = fidx / fps
        res = model.predict(frame, imgsz=640, conf=0.30, verbose=False,
                            classes=list(ANY_VEHICLE))[0]
        box = leading_vehicle_box(res, W, H)
        if box is None:
            lamp_hist.clear(); elevated_prev = False
            prev_box, prev_area, prev_t = None, None, None
            fidx += 1
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        area_ratio = (x2 - x1) * (y2 - y1) / (W * H)

        # 추적 연속성: 이전 선행차와 IoU 낮으면 시간단서 리셋
        same_track = iou((x1, y1, x2, y2), prev_box) >= IOU_TRACK
        if not same_track:
            lamp_hist.clear(); elevated_prev = False
            prev_area, prev_t = None, None

        naive = region_red_ratio(frame, x1, y1, x2, y2) * 100

        h_box = y2 - y1
        ry1 = y1 + h_box * (1 - ROI_Y_FRAC)
        rx1i, ry1i, rx2i, ry2i = (max(0, int(x1)), max(0, int(ry1)),
                                  min(W, int(x2)), min(H, int(y2)))
        roi = frame[ry1i:ry2i, rx1i:rx2i]
        mask = bright_red_mask(roi)
        lamp = (float(np.count_nonzero(mask)) / float(mask.size + 1e-6) * 100
                if mask is not None else 0.0)
        sym, pair, chmsl = lamp_pair_boxes(mask)

        base = float(np.mean(lamp_hist)) if len(lamp_hist) >= 2 else None
        if base is None:
            onset, dlamp_ratio = False, 0.0
        elif base < BASE_FLOOR:
            onset = lamp >= max(THR_LAMP * 100, base + 2.0)  # baseline≈0 이면 절대증가로
            dlamp_ratio = lamp / base if base > 1e-6 else float("inf")
        else:
            dlamp_ratio = lamp / base
            onset = dlamp_ratio >= DLAMP_MULT
        sustained = onset and elevated_prev
        lamp_hist.append(lamp)
        elevated_prev = onset

        # 점등 판정. 색·대칭만으로는 '붉은 차체'와 '점등 램프'가 구분되지 않음이 확인됨
        # (붉은 버스 하부 패널이 대칭 컴팩트 blob 으로 잡혀 sym=Y 오검출). 유일하게
        # 물리적으로 유효한 판별자는 '시간적 급증(onset)' — 실제 제동은 붉은량이 상승하나
        # 붉은 차체는 정상. 따라서 onset 을 필수조건으로 두고, 공간 단서(sym/chmsl)는 가점.
        brake_on = ((THR_LAMP * 100 <= lamp < LAMP_MAX * 100)
                    and onset and (sym or chmsl))

        dt = (t_now - prev_t) if prev_t is not None else None
        ttc = estimate_ttc(area_ratio, prev_area, dt)
        area_grow = (area_ratio / prev_area - 1.0) if (prev_area and prev_area > 0) else None
        looming = ((area_grow is not None and area_grow >= AREA_SURGE) or
                   (ttc is not None and ttc <= TTC_CRIT))
        prev_box, prev_area, prev_t = (x1, y1, x2, y2), area_ratio, t_now

        recs.append(dict(
            t=t_now, box=(x1, y1, x2, y2), roi_box=(rx1i, ry1i, rx2i, ry2i),
            pair_boxes=pair, roi_origin=(rx1i, ry1i),
            naive=naive, lamp=lamp, dlamp=dlamp_ratio, sym=sym, chmsl=chmsl,
            onset=onset, sustained=sustained, brake_on=brake_on,
            ttc=ttc, area_grow=area_grow, area_ratio=area_ratio, looming=looming,
            frame=frame,
        ))
        fidx += 1
    cap.release()
    return recs, "ok"


def bucket_of(r):
    # A = 개선 방식이 '점등'으로 확정(컴팩트 대칭 램프 or onset, 붉은패널 제외)
    if r["brake_on"]:
        return "A"
    # B = naive(단순 붉은비율)는 detected 로 보는데 개선 방식은 점등 아님 = 붉은 차체 오검출
    #     (어두운 빨간차 lamp≈0 도, 밝은 빨간버스 lamp 큼[패널]도 여기 — 둘 다 naive 한계 전시)
    if r["naive"] >= NAIVE_THR * 100 and not r["brake_on"] and not r["sym"]:
        return "B"
    # C = 선행차 근접(면적 큼)인데 점등 아님 = 비점등 대비
    if r["area_ratio"] >= NEAR_AREA:
        return "C"
    return None


def score(r, bucket):
    if bucket == "A":
        # 진짜 램프에 가까운 것 우선: 대칭·onset·looming 가점, 과도한 lamp% 는 감점
        return ((8 if r["sym"] else 0) + (5 if r["onset"] else 0)
                + (3 if r["looming"] else 0) + min(r["lamp"], 12))
    if bucket == "B":
        return r["naive"]
    return r["area_ratio"] * 100  # C


def naive_verdict(r):
    return "detected" if r["naive"] >= NAIVE_THR * 100 else \
           "near" if r["naive"] >= 5.0 else "miss"


def draw(r, out_path, file_label):
    vis = r["frame"].copy()
    x1, y1, x2, y2 = [int(v) for v in r["box"]]
    rx1, ry1, rx2, ry2 = [int(v) for v in r["roi_box"]]
    b = bucket_of(r)
    col = {"A": (0, 0, 255), "B": (0, 165, 255), "C": (160, 160, 160)}[b]
    cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
    ox, oy = r["roi_origin"]
    for (lx, ly, lw, lh) in r["pair_boxes"]:
        cv2.rectangle(vis, (ox + lx, oy + ly), (ox + lx + lw, oy + ly + lh),
                      (0, 255, 0), 2)
    ttc_txt = f"{r['ttc']:.2f}s" if r["ttc"] is not None else "-"
    dl = "inf" if r["dlamp"] == float("inf") else f"{r['dlamp']:.1f}x"
    line1 = (f"naive={r['naive']:.1f}%({naive_verdict(r)})  lamp={r['lamp']:.1f}%  "
             f"dLamp={dl}  sym={'Y' if r['sym'] else 'N'}  TTC={ttc_txt}")
    line2 = (f"IMPROVED verdict={b}  brake_on={'Y' if r['brake_on'] else 'N'}  "
             f"looming={'Y' if r['looming'] else 'N'}  t={r['t']:.1f}s")
    cv2.rectangle(vis, (x1, max(0, y1 - 56)), (min(vis.shape[1], x1 + 720), y1), col, -1)
    cv2.putText(vis, line1, (x1 + 6, max(20, y1 - 34)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, line2, (x1 + 6, max(42, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, file_label, (12, vis.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    imwrite_unicode(out_path, vis)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True, help="label=csv=srcdir")
    ap.add_argument("--outdir", required=True, help="rebuild_<date> 폴더")
    args = ap.parse_args()

    for b in ("A", "B", "C"):
        os.makedirs(os.path.join(args.outdir, b), exist_ok=True)
    from ultralytics import YOLO
    model = YOLO(os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt"))

    # 버킷별 후보 수집: 클립당 버킷당 최대 MAX_PER_CLIP (클립 내 상위)
    pool = {"A": [], "B": [], "C": []}
    n_clip, n_skip = 0, 0
    for spec in args.sources:
        label, path, srcdir = spec.split("=", 2)
        if not os.path.exists(path):
            print(f"[skip] {path} 없음"); continue
        with open(path, newline="", encoding="utf-8-sig") as f:
            cands = [r for r in csv.DictReader(f) if r.get("type", "").startswith("c_brake")]
        print(f"[{label}] c_brake 후보 {len(cands)}건", flush=True)
        for c in cands:
            n_clip += 1
            vid = os.path.join(srcdir, c["file"])
            if not os.path.exists(vid):
                n_skip += 1; print(f"  · {c['file']} 원본없음"); continue
            recs, status = process_window(model, vid, float(c["t_start"]), float(c["t_end"]))
            if status != "ok" or not recs:
                print(f"  · {c['file']} {status or 'no-rec'}"); continue
            per_clip = {"A": [], "B": [], "C": []}
            for r in recs:
                b = bucket_of(r)
                if b:
                    r["_src"] = c["file"]; r["_label"] = label
                    per_clip[b].append(r)
            for b in ("A", "B", "C"):
                top = sorted(per_clip[b], key=lambda r: -score(r, b))[:MAX_PER_CLIP]
                pool[b].extend(top)
            summ = {b: len(per_clip[b]) for b in ("A", "B", "C")}
            print(f"  [{n_clip}] {c['file']} A{summ['A']} B{summ['B']} C{summ['C']}", flush=True)

    # 전역 상위 N 저장
    saved = {"A": [], "B": [], "C": []}
    for b in ("A", "B", "C"):
        ranked = sorted(pool[b], key=lambda r: -score(r, b))[:TOPN_PER_BUCKET]
        for i, r in enumerate(ranked, 1):
            name = (f"{b}{i:02d}_{r['_label']}_{os.path.splitext(r['_src'])[0]}"
                    f"_t{r['t']:.1f}s_lamp{r['lamp']:04.1f}_naive{r['naive']:04.1f}.png")
            draw(r, os.path.join(args.outdir, b, name), r["_src"])
            saved[b].append(name)

    # 리포트
    lines = ["# 급정거 다중단서 재추출 리포트", "",
             f"생성 대상 클립 {n_clip}건 (원본없음 skip {n_skip}건)", "",
             "## 사용 임계값",
             f"- lamp% ≥ {THR_LAMP*100:.0f}% · S≥{S_CUT} · V≥{V_CUT}",
             f"- dLamp onset ≥ {DLAMP_MULT}x (baseline<{BASE_FLOOR}%%면 절대증가)",
             f"- looming: 면적급증 ≥ {AREA_SURGE*100:.0f}% OR TTC ≤ {TTC_CRIT}s",
             f"- B(오검출): naive ≥ {NAIVE_THR*100:.0f}% & lamp < {THR_LAMP*100:.0f}% & sym=N",
             f"- C(근접): 선행차 면적비 ≥ {NEAR_AREA}", "",
             "## 버킷별 결과 (전역 상위 저장)"]
    for b, desc in (("A", "점등 → S11 급정거 예시 + S18 검출 성공"),
                    ("B", "naive 오검출(빨간 차체) → S18 단순 방식 한계 대조"),
                    ("C", "근접 비점등 → S18 비점등 대비")):
        lines.append(f"\n### {b} — {desc}  ({len(saved[b])}장)")
        for nm in saved[b]:
            lines.append(f"- `{b}/{nm}`")
    with open(os.path.join(args.outdir, "REBUILD_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n=== 저장 완료: A{len(saved['A'])} B{len(saved['B'])} C{len(saved['C'])} ===")
    print(f"[리포트] {os.path.join(args.outdir, 'REBUILD_REPORT.md')}")


if __name__ == "__main__":
    main()
