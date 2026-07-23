"""
공식 다발지역 10개 지점의 교차로 유형 분류 → intersection_types.csv.

지점명 괄호 안 랜드마크에 '사거리/삼거리/오거리/육거리/교차로'가 있으면 교차로형으로
본다. 서청주IC삼거리처럼 IC 진출입 삼거리는 별도로 표시해, '지표면 교차로'만 셀지
'IC 포함'으로 셀지 판단할 수 있게 남긴다.

이 분류 결과가 제안서 3-5 캡션 문구("10곳 중 N곳이 교차로")의 근거다.
"""
from __future__ import annotations

import csv
import os
import sys

# 콘솔이 cp949 여도 유니코드 대시가 깨지지 않게 UTF-8 로 강제한다
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "cheongju_truck_hotspots.csv")
OUT = os.path.join(HERE, "intersection_types.csv")

INTERSECTION_KW = ("사거리", "삼거리", "오거리", "육거리", "교차로")


def landmark(spot_nm: str) -> str:
    if "(" in spot_nm and ")" in spot_nm:
        return spot_nm[spot_nm.index("(") + 1:spot_nm.rindex(")")].replace(" 부근", "").strip()
    return spot_nm.split()[-1]


def classify(landmark_nm: str) -> tuple[str, bool, bool]:
    """(유형, 교차로여부, IC진출입여부) 반환."""
    hit = next((k for k in INTERSECTION_KW if k in landmark_nm), None)
    is_ic = "IC" in landmark_nm
    return (hit or "비교차로", hit is not None, is_ic)


def main() -> None:
    rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
    out = []
    for r in rows:
        lm = landmark(r["spot_nm"])
        kind, is_inter, is_ic = classify(lm)
        out.append({
            "year": r["year"], "district": r["district"], "landmark": lm,
            "occrrnc_cnt": r["occrrnc_cnt"], "caslt_cnt": r["caslt_cnt"],
            "dth_dnv_cnt": r["dth_dnv_cnt"], "se_dnv_cnt": r["se_dnv_cnt"],
            "intersection_type": kind,
            "is_intersection": int(is_inter),
            "is_ic_junction": int(is_ic),
            "lat": r["lat"], "lon": r["lon"],
        })
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    n = len(out)
    n_inter = sum(r["is_intersection"] for r in out)
    n_ic = sum(1 for r in out if r["is_intersection"] and r["is_ic_junction"])
    n_surface = n_inter - n_ic
    n_death = sum(int(r["dth_dnv_cnt"]) for r in out)
    print(f"[저장] {OUT}")
    print(f"  전체 {n}개 지점")
    print(f"  교차로형(사거리·삼거리·육거리 등): {n_inter}/{n} "
          f"— 지표면 교차로 {n_surface}, IC 진출입 {n_ic}")
    print(f"  비교차로(랜드마크 인접): {n - n_inter}")
    print(f"  사망자 총합: {n_death}명 (전 지점 중상 중심)")


if __name__ == "__main__":
    main()
