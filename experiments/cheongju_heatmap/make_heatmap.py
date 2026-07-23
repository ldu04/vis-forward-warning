"""
제안서 3-5용 대비 그림 생성.

왼쪽: 기존 공공데이터(도로교통공단 화물차 사고 다발지역) — 교차로 중심·사망 0
오른쪽: 본 서비스의 궤적 기반 수집 개념도 — ★실데이터 아님★

이 그림의 논지
- 공식 데이터 10곳은 '반경 100m 내 4건' 기준이라, 사고가 한 점에 수렴하는 교차로만
  잡힌다(10곳 중 8곳이 교차로형, 그중 7곳이 지표면 교차로). 선형으로 분산되는
  고속도로·간선도로의 사망사고 구간은 원리적으로 못 잡는다.
- 게다가 10곳 전부 사망 0명(중상 중심)이다. 명칭이 '사망·중상 다발지역'이라
  사망사고 다발로 읽히기 쉬우나 실제는 중상 중심이다.
- 본 서비스는 차량 궤적을 따라 위험 이벤트를 수집하므로 선형 구간을 커버한다.

색상은 dataviz 레퍼런스 팔레트를 따른다 (sequential blue / series red / ink tokens).
교차로 분류는 classify_spots.py → intersection_types.csv 참조.
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "cheongju_truck_hotspots.csv")
TYPES_CSV = os.path.join(HERE, "intersection_types.csv")

# --- 팔레트 (references/palette.md) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
RED = "#e34948"      # 실데이터 사고지점
AMBER = "#e0a33a"    # 비교차로(랜드마크형) 구분용
VIOLET = "#4a3aa7"   # 물류 거점 참조점
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf",
       "#184f95", "#0d366b"]

# 오송생명과학단지는 제외한다 — 최근접 다발지점이 8.7km 로 근거가 없어
# 실증지 서술에서 오창 단독으로 간다.
REFERENCE_POINTS = [
    ("오창IC", 36.7220, 127.4300),
    ("오창과학산업단지", 36.7100, 127.4400),
    ("청주국제공항", 36.7166, 127.4990),
]

INTERSECTION_KW = ("사거리", "삼거리", "오거리", "육거리", "교차로")

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


def load_spots():
    rows = []
    with open(CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                nm = r["spot_nm"]
                lm = (nm[nm.index("(") + 1:nm.rindex(")")].replace(" 부근", "").strip()
                      if "(" in nm else nm.split()[-1])
                rows.append({
                    "year": int(r["year"]),
                    "landmark": lm,
                    "occ": int(r["occrrnc_cnt"]),
                    "cas": int(r["caslt_cnt"]),
                    "death": int(r["dth_dnv_cnt"]),
                    "lat": float(r["lat"]),
                    "lon": float(r["lon"]),
                    "is_inter": any(k in lm for k in INTERSECTION_KW),
                    "is_ic": "IC" in lm,
                })
            except (ValueError, KeyError):
                continue
    return rows


def stats(spots):
    n = len(spots)
    inter = sum(1 for s in spots if s["is_inter"])
    ic = sum(1 for s in spots if s["is_inter"] and s["is_ic"])
    return n, inter, inter - ic, ic, sum(s["death"] for s in spots)


def style_axes(ax, title, subtitle):
    ax.set_xlim(LON0, LON1)
    ax.set_ylim(LAT0, LAT1)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(AXIS)
        s.set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("경도 (°E)", color=MUTED, fontsize=9)
    ax.set_ylabel("위도 (°N)", color=MUTED, fontsize=9)
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", pad=30, loc="left")
    ax.text(0, 1.008, subtitle, transform=ax.transAxes, color=INK2,
            fontsize=9.5, va="bottom", ha="left")


def draw_refs(ax, label=True):
    for name, la, lo in REFERENCE_POINTS:
        ax.plot(lo, la, marker="D", markersize=8, color=VIOLET,
                markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=6)
        if label:
            ax.annotate(name, (lo, la), xytext=(8, -12), textcoords="offset points",
                        fontsize=8.5, color=VIOLET, fontweight="bold", zorder=7)


def panel_official(ax, spots):
    n, inter, surface, ic, death = stats(spots)
    style_axes(
        ax,
        "기존 공공데이터 — 화물차 사고 다발지역",
        "도로교통공단 · 2021~2024 누적 · 청주 4개구 + 진천 + 음성",
    )
    draw_refs(ax)

    # 교차로형 / 비교차로를 마커로 구분
    for s in spots:
        c = RED if s["is_inter"] else AMBER
        mk = "o" if s["is_inter"] else "s"
        ax.scatter([s["lon"]], [s["lat"]], s=230, facecolor=c, alpha=0.55,
                   edgecolor=SURFACE, linewidth=1.6, marker=mk, zorder=5)

    # 라벨: 위치 중복(창리사거리 2년) 제거, 발생건수·사망자수 표기
    seen, uniq = set(), []
    for s in spots:
        if s["landmark"] in seen:
            continue
        seen.add(s["landmark"])
        uniq.append(s)
    offsets = [(11, 10), (13, -30), (-14, 26), (12, 12), (-16, -34), (14, -14),
               (-18, 16), (12, 30), (-20, -20)]
    for i, s in enumerate(uniq):
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(f"{s['landmark']}\n{s['occ']}건 / 사망 {s['death']}",
                    (s["lon"], s["lat"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.4, color=INK,
                    zorder=8, ha="left" if dx > 0 else "right",
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7,
                                    shrinkA=0, shrinkB=3),
                    bbox=dict(boxstyle="round,pad=0.28", facecolor=SURFACE,
                              edgecolor=GRID, linewidth=0.6, alpha=0.95))

    ax.text(0.5, 0.05,
            f"{n}곳 중 {inter}곳이 교차로형(지표면 {surface} · IC삼거리 {ic}) — "
            f"사고가 한 점에 수렴하는 지점만 포착\n"
            f"전 지점 사망 0명(중상 중심) · 간선·고속도로의 선형 사망구간은 기준상 미포착",
            transform=ax.transAxes, ha="center", fontsize=9, color=INK2,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0efec",
                      edgecolor=GRID, linewidth=0.8))

    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", markersize=9, color=RED,
               alpha=0.6, label=f"교차로형 다발지점 ({inter}곳)"),
        Line2D([], [], marker="s", linestyle="", markersize=8, color=AMBER,
               alpha=0.6, label=f"비교차로(랜드마크 인접, {n - inter}곳)"),
        Line2D([], [], marker="D", linestyle="", markersize=8, color=VIOLET,
               label="물류 거점 (참조)"),
    ], loc="upper left", fontsize=8.5, framealpha=0.95, edgecolor=GRID)


def panel_concept(ax, spots, rng):
    style_axes(
        ax,
        "본 서비스의 궤적 기반 수집 (개념도)",
        "차량 단말이 주행 궤적을 따라 상시 수집하는 위험 이벤트 — 선형 구간 커버",
    )
    # 물류 통행 축(오창 거점 중심)을 따라 선형으로 가상 이벤트 생성
    corridors = [
        ((36.7220, 127.4300), (36.7100, 127.4400), 190),   # 오창IC ~ 오창산단
        ((36.7100, 127.4400), (36.7166, 127.4990), 150),   # 오창 ~ 공항
        ((36.7380, 127.4585), (36.6450, 127.4160), 170),   # 창리사거리 ~ 서청주IC
        ((36.6450, 127.4160), (36.6300, 127.4900), 150),   # 서청주 ~ 시내
        ((36.7220, 127.4300), (36.7380, 127.4585), 120),   # 오창IC ~ 창리
    ]
    xs, ys = [], []
    for (a_la, a_lo), (b_la, b_lo), n in corridors:
        t = rng.random(n)
        ys.append(a_la + (b_la - a_la) * t + rng.normal(0, 0.0055, n))
        xs.append(a_lo + (b_lo - a_lo) * t + rng.normal(0, 0.0055, n))
    for s in spots:      # 실제 다발지점 주변에도 밀도 부여(개념적 정합)
        ys.append(rng.normal(s["lat"], 0.0045, 26))
        xs.append(rng.normal(s["lon"], 0.0045, 26))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    m = (x > LON0) & (x < LON1) & (y > LAT0) & (y < LAT1)
    x, y = x[m], y[m]

    kde = gaussian_kde(np.vstack([x, y]), bw_method=0.16)
    gx, gy = np.mgrid[LON0:LON1:260j, LAT0:LAT1:260j]
    z = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    z /= z.max()

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ)
    ax.contourf(gx, gy, z, levels=np.linspace(0.06, 1.0, 12), cmap=cmap,
                alpha=0.88, zorder=2)
    ax.scatter(x, y, s=1.6, color="#0d366b", alpha=0.20, zorder=3)
    draw_refs(ax)

    sm = matplotlib.cm.ScalarMappable(cmap=cmap,
                                      norm=matplotlib.colors.Normalize(0, 1))
    cb = plt.colorbar(sm, ax=ax, fraction=0.036, pad=0.02)
    cb.set_label("상대 위험 밀도", color=INK2, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_edgecolor(AXIS)

    ax.text(0.5, 0.05,
            f"가상 이벤트 {len(x):,}건 · 도로 구간 단위 해상도\n"
            "교차로 '점'이 아니라 주행 '선'을 따라 분포 — 선형 위험구간 포착",
            transform=ax.transAxes, ha="center", fontsize=9, color=INK2,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0efec",
                      edgecolor=GRID, linewidth=0.8))
    # 실데이터 아님 명시 (요구사항)
    ax.text(0.5, 0.55, "개념도 — 실데이터 아님",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=26, fontweight="bold", color="#d03b3b", alpha=0.32,
            rotation=22, zorder=10)
    ax.text(0.985, 0.975, "※ 개념도: 가상 데이터로 생성 — 실데이터 아님",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color="#d03b3b", fontweight="bold", zorder=11,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE,
                      edgecolor="#d03b3b", linewidth=1.1))


FOOT = ("출처: 도로교통공단, 화물차 교통사고 다발지역 정보, 공공데이터포털 "
        "(조회일 2026-07-20) · 선정기준: 최근 3년 중상 중심 화물차 사고, "
        "반경 100m 내 4건 이상 · 오른쪽은 개념도(실데이터 아님)")


def main() -> None:
    setup_font()
    spots = load_spots()

    fig, axes = plt.subplots(1, 2, figsize=(19, 9), facecolor=SURFACE)
    panel_official(axes[0], spots)
    panel_concept(axes[1], spots, np.random.default_rng(42))
    fig.suptitle("청주권 화물차 사고 위험 정보 — 기존 공공데이터 vs 본 서비스",
                 fontsize=16, fontweight="bold", color=INK, y=0.975)
    fig.text(0.5, 0.012, FOOT, ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.028, 1, 0.955])
    out = os.path.join(HERE, "cheongju_heatmap_comparison.png")
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    print(f"[저장] {out}")

    for fname, drawer, kind in (
        ("cheongju_official_hotspots.png", panel_official, "official"),
        ("cheongju_concept_heatmap.png", panel_concept, "concept"),
    ):
        fig, ax = plt.subplots(figsize=(10.5, 9), facecolor=SURFACE)
        if kind == "official":
            drawer(ax, spots)
        else:
            drawer(ax, spots, np.random.default_rng(42))
        fig.text(0.5, 0.012, FOOT, ha="center", fontsize=7.6, color=MUTED)
        fig.tight_layout(rect=[0, 0.03, 1, 1])
        p = os.path.join(HERE, fname)
        fig.savefig(p, dpi=180, facecolor=SURFACE)
        plt.close(fig)
        print(f"[저장] {p}")


if __name__ == "__main__":
    main()
