"""
S1 / S3 최종 차트 제작 (make_s1_s3_drafts.py 시안 검토 후 확정본).

데이터 출처: docs/트럭너머_인계문서.md "확정된 핵심 데이터" — 신규 수치 생성 없음.
  S1: 고속도로 화물차 사고 사망자 2023=71 / 2024=89 / 2025=93명 (경찰청)
  S3: 충북 고속도로 사망자 중 화물차 관련 54%(37명/2023~2025, 한국도로공사 충북본부)
      보조 수치(2026년 38%, 경남권 48.4%)는 인계문서 미확정이라 이번 차트에서 제외.

A안(도넛)을 채택 — 이전 시안 대비 여백을 줄이고 도넛/배지 크기를 키워 재배치.
B안(대형타이포)은 참고용으로 함께 저장.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, FancyBboxPatch

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "candidates", "final_review", "S1_S3_final")
os.makedirs(HERE, exist_ok=True)

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; AXIS = "#c3c2b7"; RED = "#e34948"; VIOLET = "#4a3aa7"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]


def setup_font():
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for c in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        if c in have:
            plt.rcParams["font.family"] = c
            break
    plt.rcParams["axes.unicode_minus"] = False


def footnote(fig, text):
    fig.text(0.02, 0.02, text, fontsize=9, color=MUTED, ha="left")


# ============================================================ S1
def s1_bar():
    setup_font()
    fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    years = ["2023", "2024", "2025"]
    vals = [71, 89, 93]
    colors = [AXIS, AXIS, RED]

    bars = ax.bar(years, vals, width=0.5, color=colors, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 2.5, f"{v}명",
                ha="center", fontsize=20, fontweight="bold",
                color=RED if v == 93 else INK)

    ax.set_ylim(0, 108)
    ax.set_ylabel("고속도로 화물차 사고 사망자 수 (명)", fontsize=13, color=INK2)
    ax.set_title("고속도로 화물차 사고 사망자는 줄지 않고 있다", fontsize=18,
                fontweight="bold", color=INK, pad=18)
    ax.tick_params(axis="x", labelsize=15, colors=INK)
    ax.tick_params(axis="y", labelsize=10, colors=MUTED)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)

    footnote(fig, "출처: 경찰청 (2026년 진행 중 통계는 별도 슬라이드)")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(HERE, "S1_bar_deaths.png")
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return out


# ============================================================ S3 공통: 소형 충북 위치 배지
def _chungbuk_locator(ax, cx, cy, r):
    """실측 지도 아님 — 위치 강조용 소형 배지(대한민국 윤곽 안에 충북 점 표시)."""
    ax.add_patch(FancyBboxPatch((cx - r, cy - r*1.15), r*2, r*2.3,
                                boxstyle="round,pad=0.01,rounding_size=0.05",
                                fc="#f3f2ee", ec=AXIS, lw=1.2, zorder=2))
    ax.text(cx, cy + r*1.35, "대한민국", fontsize=9, color=MUTED, ha="center")
    ax.add_patch(Circle((cx, cy + r*0.15), r*0.16, fc=RED, ec="white", lw=1.2, zorder=4))
    ax.text(cx, cy - r*0.55, "충북", fontsize=11, color=INK, fontweight="bold",
           ha="center", zorder=4)


# ============================================================ S3-A 도넛 (채택본 — 재배치)
def s3_a_donut():
    setup_font()
    fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_aspect("equal")

    ax.text(0.6, 9.35, "충북 고속도로 사망자의 절반 이상이 화물차 관련이다",
           fontsize=17, fontweight="bold", color=INK, ha="left")

    cx, cy, rad = 3.7, 4.9, 2.75
    ax.add_patch(Wedge((cx, cy), rad, 90, 90 - 0.54*360, width=0.95,
                       fc=RED, ec=SURFACE, lw=2, zorder=3))
    ax.add_patch(Wedge((cx, cy), rad, 90 - 0.54*360, 90, width=0.95,
                       fc=GRID, ec=SURFACE, lw=2, zorder=3))
    ax.text(cx, cy + 0.2, "54%", fontsize=58, fontweight="bold", color=RED,
           ha="center", va="center", zorder=4)
    ax.text(cx, cy - 1.05, "화물차 관련", fontsize=13.5, color=INK2, ha="center", zorder=4)

    ax.text(cx, cy - rad - 0.75,
           "충북 고속도로 사망자 중 화물차 관련 37명\n(2023~2025년, 주요원인: 졸음운전·전방주시 태만)",
           fontsize=12.5, color=INK, ha="center", linespacing=1.5)

    _chungbuk_locator(ax, 8.35, 5.6, 1.15)

    footnote(fig, "출처: 한국도로공사 충북본부")
    out = os.path.join(HERE, "S3_A_donut.png")
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return out


# ============================================================ S3-B 대형 타이포 (참고용)
def s3_b_typo():
    setup_font()
    fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_aspect("equal")

    ax.text(1.0, 9.2, "충북 고속도로 사망자의 절반 이상이 화물차 관련이다",
           fontsize=17, fontweight="bold", color=INK, ha="left")

    ax.text(3.9, 5.1, "54%", fontsize=110, fontweight="bold", color=RED,
           ha="center", va="center")
    ax.text(3.9, 3.1,
           "충북 고속도로 사망자 중 화물차 관련 37명\n(2023~2025년, 주요원인: 졸음운전·전방주시 태만)",
           fontsize=13, color=INK, ha="center", linespacing=1.6)

    _chungbuk_locator(ax, 8.3, 5.6, 1.05)

    footnote(fig, "출처: 한국도로공사 충북본부 (참고용 — A안 채택됨)")
    out = os.path.join(HERE, "S3_B_typo.png")
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for fn in (s1_bar, s3_a_donut, s3_b_typo):
        path = fn()
        print(f"  [saved] {path}")
