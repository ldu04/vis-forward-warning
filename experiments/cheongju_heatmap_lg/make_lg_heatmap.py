"""
지자체별 교통사고 다발지역 히트맵 생성.

색상·타이포는 experiments/cheongju_heatmap/make_heatmap.py 와 동일한 팔레트를 쓴다
(dataviz 레퍼런스 팔레트: sequential blue / series red / ink 토큰).
두 그림이 제안서에 나란히 실리므로 한 벌로 보여야 한다.

★ 아직 실데이터가 없다 (해당 API 401, collect_lg_hotspots.py 주석 참조).
  그래서 --dummy 로 합성 데이터를 넣어 렌더링 경로를 미리 검증해 둔다.
  키가 열리면 collect_lg_hotspots.py 를 돌린 뒤 인자 없이 실행하면
  같은 코드가 실데이터로 그대로 그린다.

  더미로 그린 그림에는 지울 수 없는 'DUMMY' 워터마크가 박힌다.
  실데이터 그림과 섞여서 제안서에 잘못 실리는 사고를 막기 위함이다.

사용법:
    python make_lg_heatmap.py --dummy     # 합성 데이터로 렌더링 테스트
    python make_lg_heatmap.py             # 실데이터 (CSV 필요)
"""
from __future__ import annotations

import argparse
import csv
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "cheongju_lg_hotspots.csv")

# --- 팔레트 (cheongju_heatmap/make_heatmap.py 와 동일) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
RED = "#e34948"
VIOLET = "#4a3aa7"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf",
       "#184f95", "#0d366b"]

REFERENCE_POINTS = [   # 오송 제외 — 실증지는 오창 단독
    ("오창IC", 36.7220, 127.4300),
    ("오창과학산업단지", 36.7100, 127.4400),
    ("청주국제공항", 36.7166, 127.4990),
]

LAT0, LAT1 = 36.595, 36.765
LON0, LON1 = 127.295, 127.525


def setup_font() -> None:
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for c in ("Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR"):
        if c in have:
            plt.rcParams["font.family"] = c
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_spots() -> list[dict]:
    rows = []
    if not os.path.exists(CSV):
        return rows
    with open(CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                lat, lon = float(r["lat"]), float(r["lon"])
            except (TypeError, ValueError, KeyError):
                continue
            rows.append({
                "lat": lat, "lon": lon,
                "name": r.get("spot_nm") or "",
                "occ": int(float(r.get("occrrnc_cnt") or 0)),
                "cas": int(float(r.get("caslt_cnt") or 0)),
                "year": r.get("year") or "",
            })
    return rows


def make_dummy(n: int = 24, seed: int = 20260721) -> list[dict]:
    """렌더링 경로 검증용 합성 지점. 실제 사고 위치와 무관하다."""
    rnd = random.Random(seed)
    # 기준점 주변에 몰리게 만들어 KDE 가 실제와 비슷한 모양을 내도록 한다
    rows = []
    for i in range(n):
        base = REFERENCE_POINTS[i % len(REFERENCE_POINTS)]
        lat = base[1] + rnd.gauss(0, 0.018)
        lon = base[2] + rnd.gauss(0, 0.022)
        lat = min(max(lat, LAT0 + 0.005), LAT1 - 0.005)
        lon = min(max(lon, LON0 + 0.005), LON1 - 0.005)
        rows.append({"lat": lat, "lon": lon,
                     "name": f"더미지점{i+1:02d}",
                     "occ": rnd.randint(4, 12), "cas": rnd.randint(4, 20),
                     "year": rnd.choice([2021, 2022, 2023, 2024])})
    return rows


def draw(spots: list[dict], out: str, dummy: bool) -> None:
    setup_font()
    fig, ax = plt.subplots(figsize=(10.5, 9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    lats = np.array([s["lat"] for s in spots])
    lons = np.array([s["lon"] for s in spots])
    weights = np.array([max(s["cas"], 1) for s in spots], dtype=float)

    # KDE 밀도면
    gx, gy = np.mgrid[LON0:LON1:220j, LAT0:LAT1:220j]
    if len(spots) >= 3:
        try:
            kde = gaussian_kde(np.vstack([lons, lats]), weights=weights)
            z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
            z = z / z.max()
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ)
            ax.contourf(gx, gy, z, levels=np.linspace(0.06, 1.0, 12),
                        cmap=cmap, alpha=0.85, zorder=2)
            sm = matplotlib.cm.ScalarMappable(
                cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0, vmax=1))
            cb = fig.colorbar(sm, ax=ax, fraction=0.036, pad=0.02)
            cb.set_label("상대 위험 밀도", color=INK2, fontsize=9)
            cb.ax.tick_params(colors=MUTED, labelsize=8)
            cb.outline.set_edgecolor(AXIS)
        except Exception as exc:      # 표본이 축퇴하면 KDE 가 실패할 수 있다
            print(f"  [경고] KDE 생략: {exc}")

    # 사고 지점
    sizes = [28 + 9 * s["occ"] for s in spots]
    ax.scatter(lons, lats, s=sizes, c=RED, alpha=0.82, zorder=6,
               edgecolors="white", linewidths=0.9)

    # 물류 거점 기준점
    for name, rlat, rlon in REFERENCE_POINTS:
        ax.scatter([rlon], [rlat], marker="^", s=115, c=VIOLET, zorder=7,
                   edgecolors="white", linewidths=0.9)
        ax.annotate(name, (rlon, rlat), xytext=(7, 5),
                    textcoords="offset points", fontsize=7.6, color=INK)

    ax.set_xlim(LON0, LON1)
    ax.set_ylim(LAT0, LAT1)
    ax.set_xlabel("경도 (°E)", color=MUTED, fontsize=9)
    ax.set_ylabel("위도 (°N)", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(AXIS)
    ax.grid(color=GRID, linewidth=0.6, zorder=1)

    title = "청주권 지자체별 교통사고 다발지역"
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", pad=28, loc="left")
    sub = ("합성(더미) 데이터 — 렌더링 검증용, 실제 사고 위치 아님"
           if dummy else "도로교통공단 지자체별 교통사고 다발지역, 공공데이터포털")
    ax.text(0, 1.015, f"{len(spots)}개 지점 · {sub}", transform=ax.transAxes,
            fontsize=9.5, va="bottom", ha="left", color=INK2)

    ax.legend(handles=[
        Line2D([], [], marker="o", color="none", markerfacecolor=RED,
               markersize=9, label="사고 다발지점 (크기=발생건수)"),
        Line2D([], [], marker="^", color="none", markerfacecolor=VIOLET,
               markersize=10, label="물류 거점 (기준점)"),
    ], loc="upper left", fontsize=8.5, framealpha=0.95, edgecolor=GRID)

    if dummy:
        ax.text(0.5, 0.5, "DUMMY", transform=ax.transAxes, ha="center", va="center",
                fontsize=78, fontweight="bold", color="#d03b3b", alpha=0.22, zorder=11)
        ax.text(0.985, 0.985, "합성 데이터 — 제안서에 사용 금지",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color="#d03b3b", fontweight="bold", zorder=11)

    fig.tight_layout()
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    print(f"[저장] {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true",
                    help="합성 데이터로 렌더링 경로만 검증")
    args = ap.parse_args()

    spots = [] if args.dummy else load_spots()
    dummy = args.dummy
    if not spots and not args.dummy:
        print("실데이터 CSV 가 없습니다 (cheongju_lg_hotspots.csv).")
        print("→ 인증이 열린 뒤 collect_lg_hotspots.py 를 먼저 실행하세요.")
        print("→ 지금은 --dummy 로 렌더링만 검증할 수 있습니다.")
        raise SystemExit(1)
    if dummy:
        spots = make_dummy()
        print(f"합성 데이터 {len(spots)}개 지점으로 렌더링합니다.")

    out = os.path.join(HERE, "lg_heatmap_DUMMY.png" if dummy else "cheongju_lg_heatmap.png")
    draw(spots, out, dummy)


if __name__ == "__main__":
    main()
