"""
S10(서비스 개요) / S24(점 vs 선형 개념도) 시안 제작 — 각 3개씩.

★ 스타일 참조 불가 안내
  S06_시야차단_다이어그램.svg / S13_경고화면_3단계.svg 는 이 환경에 파일이
  존재하지 않아(0단계에서 확인) 아이콘 스타일·정확한 폰트·레이아웃 톤을
  참조할 수 없다. 색상만 이 프로젝트에서 이미 검증된 dataviz 팔레트
  (make_heatmap.py, make_ochang_map.py 와 동일)를 그대로 써서 최소한의
  일관성을 확보했다. 실제 SVG 확보 후 폰트·아이콘 스타일 재조정 필요.

문안 디자인 규칙 반영: 밝은 배경, 16:9, 각주 1줄 좌하단, 색 3개 이내,
강조색은 위험·핵심에만, 글머리 최대 4개.

세부 스펙(정확한 아이콘 형태, 레이아웃 비중 등)이 불명확한 지점은 확정하지
않고 시안 A/B/C 로 나눠 남긴다 — 선택은 사용자 복귀 후.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import numpy as np

# 스크립트 자체는 blackbox_scan/ 루트(추적 대상)에 두고, 산출물(PNG)만
# candidates/final_review/S10_S24_drafts/ 에 저장한다(.gitignore 로 로컬 전용).
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "candidates", "final_review", "S10_S24_drafts")
os.makedirs(HERE, exist_ok=True)

# 프로젝트 기존 팔레트 (make_heatmap.py 와 동일)
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
    fig.text(0.02, 0.02, text, fontsize=8, color=MUTED, ha="left")


def box(ax, xy, w, h, text, fc=SURFACE, ec=AXIS, tc=INK, fs=12, bold=False):
    b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                       fc=fc, ec=ec, lw=1.4, zorder=3)
    ax.add_patch(b)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal",
            zorder=4, linespacing=1.3)


def arrow(ax, p0, p1, color=VIOLET, lw=2.2, style="-|>"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=18,
                        color=color, lw=lw, zorder=2)
    ax.add_patch(a)


# ============================================================ S10 시안
def s10_common_axes():
    fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.6)
    ax.axis("off")
    return fig, ax


def s10_A():
    """시안A — 좌우 대칭 2열, 중앙 세로 흐름 + 우측 브로드캐스트 화살표."""
    setup_font()
    fig, ax = s10_common_axes()
    ax.text(0.3, 5.3, "트럭이 본 것을, 뒤차가 함께 본다", fontsize=18,
            fontweight="bold", color=INK, ha="left")

    # 좌: 대형차량
    box(ax, (0.5, 3.9), 3.2, 0.7, "대형차량 스마트폰", fc=VIOLET, tc="white", fs=13, bold=True)
    box(ax, (0.5, 3.0), 3.2, 0.7, "전방 카메라 촬영", fs=12)
    arrow(ax, (2.1, 3.0), (2.1, 2.5))
    box(ax, (0.5, 1.8), 3.2, 0.7, "온디바이스 AI\n위험 감지", fs=12)
    arrow(ax, (2.1, 1.8), (2.1, 1.3))
    box(ax, (0.5, 0.6), 3.2, 0.7, "TTC 계산 →\n위험등급 분류", fs=12)

    # 중앙: 브로드캐스트
    arrow(ax, (3.7, 0.95), (6.3, 0.95), color=RED, lw=3)
    ax.text(5.0, 1.25, "근거리 브로드캐스트", fontsize=11, color=RED,
            ha="center", fontweight="bold")

    # 우: 후방차량
    box(ax, (6.3, 3.9), 3.2, 0.7, "후방차량 스마트폰", fc=VIOLET, tc="white", fs=13, bold=True)
    box(ax, (6.3, 3.0), 3.2, 0.7, "알림 앱 실행 중", fs=12)
    box(ax, (6.3, 0.6), 3.2, 0.7, "색상·소리\n경고 표시", fs=12, fc="#fff3f2", ec=RED)
    arrow(ax, (7.9, 3.0), (7.9, 1.3), color=AXIS, style="-")

    footnote(fig, "시안 A — 좌우 2열 세로흐름. 스타일참조 SVG 없음(색상만 프로젝트 팔레트 준용)")
    fig.savefig(os.path.join(HERE, "S10_A_leftright.png"), dpi=170, facecolor=SURFACE)
    plt.close(fig)


def s10_B():
    """시안B — 상단 좌우 배치 + 하단 5단계 흐름 바 (문안 원문 순서 그대로)."""
    setup_font()
    fig, ax = s10_common_axes()
    ax.text(0.3, 5.3, "트럭이 본 것을, 뒤차가 함께 본다", fontsize=18,
            fontweight="bold", color=INK, ha="left")

    box(ax, (0.5, 4.0), 4.2, 0.9, "대형차량 스마트폰\n전방 카메라 촬영",
        fc=VIOLET, tc="white", fs=13, bold=True)
    box(ax, (5.3, 4.0), 4.2, 0.9, "후방차량 스마트폰\n알림 앱 실행 중",
        fc=VIOLET, tc="white", fs=13, bold=True)
    arrow(ax, (4.7, 4.45), (5.3, 4.45), color=RED, lw=2.5)

    steps = ["온디바이스 AI\n위험 감지", "TTC 계산\n→ 위험등급", "근거리\n브로드캐스트",
             "색상·소리\n경고 표시"]
    x0 = 0.4; w = 2.15; gap = 0.2
    for i, s in enumerate(steps):
        x = x0 + i * (w + gap)
        fc = "#fff3f2" if i == 3 else SEQ[1]
        box(ax, (x, 1.6), w, 1.0, s, fc=fc, fs=11.5)
        if i < 3:
            arrow(ax, (x + w, 2.1), (x + w + gap, 2.1))
    ax.annotate("", xy=(2.65, 3.9), xytext=(2.65, 2.6),
               arrowprops=dict(arrowstyle="-|>", color=AXIS, lw=1.8))

    footnote(fig, "시안 B — 상단 좌우 요약 + 하단 4단계 흐름 바")
    fig.savefig(os.path.join(HERE, "S10_B_flowbar.png"), dpi=170, facecolor=SURFACE)
    plt.close(fig)


def s10_C():
    """시안C — 원형 아이콘 자리표시 + 중앙 대형 화살표(가장 단순/여백 중심)."""
    setup_font()
    fig, ax = s10_common_axes()
    ax.text(0.3, 5.3, "트럭이 본 것을, 뒤차가 함께 본다", fontsize=18,
            fontweight="bold", color=INK, ha="left")

    cx1, cx2, cy = 2.3, 7.7, 2.6
    ax.add_patch(Circle((cx1, cy), 1.35, fc=VIOLET, ec="none", zorder=2))
    ax.text(cx1, cy, "대형\n차량", fontsize=20, ha="center", va="center", zorder=3,
           color="white", fontweight="bold", linespacing=1.2)
    ax.text(cx1, cy - 1.9, "대형차량\n전방 카메라 → 온디바이스 AI\n→ TTC·위험등급", fontsize=11,
           ha="center", color=INK2, linespacing=1.4)

    ax.add_patch(Circle((cx2, cy), 1.35, fc=VIOLET, ec="none", zorder=2))
    ax.text(cx2, cy, "후방\n차량", fontsize=20, ha="center", va="center", zorder=3,
           color="white", fontweight="bold", linespacing=1.2)
    ax.text(cx2, cy - 1.9, "후방차량\n알림 앱 실행 중\n→ 색상·소리 경고", fontsize=11,
           ha="center", color=INK2, linespacing=1.4)

    arrow(ax, (cx1 + 1.5, cy), (cx2 - 1.5, cy), color=RED, lw=4)
    ax.text((cx1 + cx2) / 2, cy + 0.55, "근거리 브로드캐스트", fontsize=13,
           color=RED, ha="center", fontweight="bold")

    footnote(fig, "시안 C — 원형 라벨 + 대형 화살표, 가장 단순/여백 중심 (실제 아이콘은 SVG 확보 후 교체, 이모지는 폰트 미지원으로 텍스트 대체)")
    fig.savefig(os.path.join(HERE, "S10_C_iconminimal.png"), dpi=170, facecolor=SURFACE)
    plt.close(fig)


# ============================================================ S24 시안
def s24_A():
    """시안A — 순수 추상 개념도(지도 없이 점 vs 선)."""
    setup_font()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4), facecolor=SURFACE)
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        ax.set_facecolor(SURFACE)

    ax = axes[0]
    ax.set_title("공식 다발지역 — 점(교차로) 포착", fontsize=13, color=INK, fontweight="bold")
    rng = np.random.default_rng(3)
    for _ in range(8):
        x, y = rng.uniform(2, 8), rng.uniform(2, 8)
        ax.add_patch(Circle((x, y), 0.35, fc=RED, ec="white", lw=1.2, zorder=3))
    ax.plot([1, 9], [5, 5], color=GRID, lw=14, zorder=1, solid_capstyle="butt")
    ax.text(5, 1, "반경 100m·4건 기준\n→ 점에 수렴하는 곳만 포착", fontsize=10,
           ha="center", color=INK2, linespacing=1.4)

    ax = axes[1]
    ax.set_title("실제 사망사고 — 선(간선·고속도로) 분산", fontsize=13, color=INK, fontweight="bold")
    t = np.linspace(0, 1, 200)
    xline = 1 + 8 * t
    yline = 5 + 1.2 * np.sin(t * 6)
    ax.plot(xline, yline, color=VIOLET, lw=6, zorder=2, alpha=0.85, solid_capstyle="round")
    idx = rng.choice(len(t), 10, replace=False)
    ax.scatter(xline[idx], yline[idx], s=40, color=RED, zorder=3)
    ax.text(5, 1, "선형으로 분산 → 어느 한 점도\n임계(4건)를 넘지 못해 미포착", fontsize=10,
           ha="center", color=INK2, linespacing=1.4)

    fig.suptitle("정작 사망사고가 발생하는 구간은 이 데이터에 잡히지 않는다",
                fontsize=15, fontweight="bold", color=INK, y=0.98)
    footnote(fig, "시안 A — 추상 개념도(지도 없음)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(os.path.join(HERE, "S24_A_abstract.png"), dpi=170, facecolor=SURFACE)
    plt.close(fig)


def s24_B():
    """시안B — 청주 실제 지리 윤곽 위에 점(8곳) vs 가상 궤적(선) 오버레이."""
    setup_font()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4), facecolor=SURFACE)

    # 왼쪽: 실제 화물차 다발지역 8곳(교차로형) 좌표 재사용
    import csv
    csv_path = os.path.join(HERE, "..", "..", "..", "..", "cheongju_heatmap",
                            "intersection_types.csv")
    pts = []
    if os.path.exists(csv_path):
        for r in csv.DictReader(open(csv_path, encoding="utf-8-sig")):
            if r["is_intersection"] == "1":
                pts.append((float(r["lon"]), float(r["lat"])))
    LON0, LON1, LAT0, LAT1 = 127.30, 127.52, 36.60, 36.76

    ax = axes[0]
    ax.set_facecolor(SURFACE)
    ax.set_xlim(LON0, LON1); ax.set_ylim(LAT0, LAT1)
    ax.set_title(f"공식 다발지역 {len(pts)}곳(교차로형) — 점 포착", fontsize=12.5,
                color=INK, fontweight="bold")
    for lo, la in pts:
        ax.scatter([lo], [la], s=180, color=RED, ec="white", lw=1.3, zorder=4)
    ax.grid(color=GRID, lw=0.6)
    for s in ax.spines.values(): s.set_edgecolor(AXIS)
    ax.tick_params(colors=MUTED, labelsize=7)

    ax = axes[1]
    ax.set_facecolor(SURFACE)
    ax.set_xlim(LON0, LON1); ax.set_ylim(LAT0, LAT1)
    ax.set_title("실제 위험구간 — 선(간선·고속도로) 분산 [개념도]", fontsize=12.5,
                color=INK, fontweight="bold")
    rng = np.random.default_rng(7)
    for _ in range(4):
        x0, y0 = rng.uniform(LON0+0.02, LON1-0.02), rng.uniform(LAT0+0.02, LAT1-0.02)
        x1, y1 = rng.uniform(LON0+0.02, LON1-0.02), rng.uniform(LAT0+0.02, LAT1-0.02)
        ax.plot([x0, x1], [y0, y1], color=VIOLET, lw=5, alpha=0.75, zorder=3,
               solid_capstyle="round")
    ax.text(0.5, 0.04, "※ 개념도 — 실데이터 아님", transform=ax.transAxes, ha="center",
           fontsize=9, color=RED, fontweight="bold")
    ax.grid(color=GRID, lw=0.6)
    for s in ax.spines.values(): s.set_edgecolor(AXIS)
    ax.tick_params(colors=MUTED, labelsize=7)

    fig.suptitle("정작 사망사고가 발생하는 구간은 이 데이터에 잡히지 않는다",
                fontsize=15, fontweight="bold", color=INK, y=0.99)
    footnote(fig, "시안 B — 청주 실지도 좌표 기반(좌: 실데이터 8곳 / 우: 개념도)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(os.path.join(HERE, "S24_B_realmap.png"), dpi=170, facecolor=SURFACE)
    plt.close(fig)


def s24_C():
    """시안C — 도로망 스케치(직선 간선 + 교차점) 위에 점/선 병치, 중간 형태."""
    setup_font()
    fig, ax = plt.subplots(figsize=(12.8, 6.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    # 배경 도로망(회색 격자형 간선)
    for y in (3, 5, 7):
        ax.plot([0.5, 9.5], [y, y], color=GRID, lw=10, zorder=1, solid_capstyle="round")
    for x in (2, 5, 8):
        ax.plot([x, x], [0.5, 9.5], color=GRID, lw=10, zorder=1, solid_capstyle="round")

    # 점(교차로) 8개 — 교차 지점에 배치
    inter_pts = [(2, 3), (5, 3), (8, 5), (2, 7), (5, 7), (8, 3), (5, 5), (2, 5)]
    for x, y in inter_pts:
        ax.add_patch(Circle((x, y), 0.28, fc=RED, ec="white", lw=1.2, zorder=4))
    ax.text(1.4, 1.2, "● 공식 다발지역 8곳\n(교차점에서만 포착)", fontsize=10.5,
           color=RED, fontweight="bold", linespacing=1.4)

    # 선(고속도로) — 대각선으로 관통, 임계 미달로 미포착 강조
    ax.plot([0.7, 9.3], [1.3, 8.7], color=VIOLET, lw=6, alpha=0.85, zorder=2,
           solid_capstyle="round", linestyle=(0, (1, 0)))
    ax.text(7.3, 8.9, "간선·고속도로\n(선형 분산 → 미포착)", fontsize=10.5,
           color=VIOLET, fontweight="bold", ha="center", linespacing=1.4)

    ax.text(5, 9.6, "정작 사망사고가 발생하는 구간은 이 데이터에 잡히지 않는다",
           fontsize=15, fontweight="bold", color=INK, ha="center")

    footnote(fig, "시안 C — 도로망 스케치 위 점/선 병치(중간 형태, 지도·순수개념도 사이)")
    fig.savefig(os.path.join(HERE, "S24_C_roadsketch.png"), dpi=170, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    for fn in (s10_A, s10_B, s10_C, s24_A, s24_B, s24_C):
        fn()
        print(f"  [저장] {fn.__name__}")
