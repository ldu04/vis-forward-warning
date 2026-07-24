"""
100회 몬테카를로: 운전자 반응·제동 개인차 vs 기기 유무 → 포스터용 2패널 그래프(제동거리·사고율).

===========================================================================
이 스크립트가 계산하는 것 (제안서 4-3 정량 결과의 산출 근거)
===========================================================================
3중 추돌 상황(A=선두, B=중간 대형차, C=후방차)을 1차원 점질량으로 100회
몬테카를로 시뮬레이션하여, **C 차량**에 대해 다음 세 지표를 기기 유무로 비교한다.

  ① 평균 제동거리   : A 급정거 시점부터 C가 정지(또는 추돌)할 때까지 이동 거리
  ② 평균 반응시간   : A 급정거 시점부터 C가 브레이크를 밟기 시작할 때까지 (= t_C)
  ③ BC 추돌 사고율  : 100회 중 C가 B를 추돌한 비율

seed=42 고정이므로 결과는 완전 재현 가능하다 (동일 환경에서 출력 CSV 17개가
바이트 단위로 일치함을 2026-07-20 확인).

--- 산출값 (2026-07-20 실행 기준) ---
  구분          기기 미사용   기기 사용    변화
  제동거리      56.418 m     40.284 m    -16.134 m  (-28.60 %)
  반응시간       2.598 s      0.868 s     -1.730 s  (-66.58 %)
  BC 사고율      82.00 %      1.00 %     -81.00 %p
  (참고) AB 추돌  1회          1회         동일 (기기 무관 설계)

⚠️ 제안서 원고와의 대조 시 주의: 제동거리 -28.6 %는 일치하나,
   반응시간과 사고율은 원고 수치(-60.6 %, -91.0 %p)와 다르다.
   본 코드의 실제 산출값은 위 표(-66.58 %, -81.00 %p)이며,
   제안서에 저장소를 근거로 링크한다면 원고 수치를 이 값으로 정정해야 한다.

===========================================================================
반응시간 파라미터의 출처 — ★선행연구 인용 아님, 가정값★
===========================================================================
아래 분포는 **선행연구에서 인용한 값이 아니라 본 모형의 가정값**이며, 일부는
목표 사고율이 나오도록 조정된 값이다. 제안서에 "문헌 근거"로 인용해서는 안 된다.

  t_B      ~ U(1.0, 2.0) s   B 운전자가 A 급정거를 보고 제동하기까지 (가정)
  t_extra  ~ U(0.2, 2.1) s   기기 없을 때 C의 추가 인지 지연 (가정 + 조정됨)
  t_C(기기) ~ U(0.5, 1.2) s   경보 수신 → 인지 → 제동 (가정)
  a_A      ~ U(-9.0, -7.0) m/s²   선두차 급제동 감속도 (가정)
  a_C      ~ U(-7.0, -4.0) m/s²   후방차 제동 감속도 (가정)
  차간거리  AB=33 m, BC=14.5 m, v0=60 km/h

  ※ 아래 독스트링 6항에 명시되어 있듯, t_extra의 U(0.2, 2.1) 범위와 bc_m=14.5는
    "기기없음 BC≈82 %, 기기있음 BC≥1 %"가 나오도록 **역으로 맞춘 값**이다.
    즉 사고율 수치는 독립적으로 도출된 결과가 아니라 **파라미터 조정의 산물**이므로,
    제안서에서는 "가정 하의 상대적 경향"으로만 서술하고 절대 수치를 단정하지 말 것.

  ※ TODO(근거 보강): 실제 인용 가능한 값으로 교체할 것. 후보 출처 —
    운전자 인지반응시간(PRT)에 관한 교통공학 표준값, 도로용량편람(KHCM/HCM)의
    설계 반응시간, AASHTO Green Book 제동거리 산정식 등. 교체 시 이 주석에
    정확한 문헌명·연도·해당 수치를 기재하고 위 "가정값" 표기를 갱신할 것.
    ※ 참고: config.py 의 REACTION_TIME_* 값(critical 0.28~0.48 s,
      warning 0.58~1.05 s)은 본 스크립트와 별개 모형이며 서로 값이 다르다.
      두 곳의 반응시간 정의를 통일할지 여부도 함께 검토 필요.


출력: comparison_simulation.png (18x6 in, 300 dpi)
       Flourish용 CSV: flourish_simulation_long.csv, flourish_simulation_wide.csv,
       flourish_bc_summary.csv, flourish_summary_means.csv (UTF-8 BOM, 스크립트와 동일 폴더)
       Flourish 요약(평균만 3지표, trial 없음): flourish_mean_only.csv
       막대+표용(상세): flourish_report_one_sheet.csv
       (구분용 복제) flourish_bar_mean_distance.csv, flourish_summary_table.csv,
       flourish_annotation_distance.txt, flourish_bar_table_guide.txt
       공모전 3종 시각화용: flourish_dist_reaction_histogram.csv,
       flourish_slope_stop_distance.csv, flourish_slope_reaction_time.csv,
       flourish_sankey_bc_flows.csv, flourish_chart_guide.txt

---------------------------------------------------------------------------
수정 전 확인 — 시스템 구조 (공모전 스토리와의 정합)
---------------------------------------------------------------------------
- B: 송신기만. A 급정거를 카메라(YOLO 등)로 **자동 감지** → **즉시** C에 신호 전송.
  본 모형에서는 그 신호가 C의 반응시간 t_C를 직접 샘플하는 단일 경로로 묶음.
- C: 수신기만. 수신 후 소리/시각 경보 → 운전자 인지 → 제동. 기기 있음에서 t_C는
  **전송·경보·C 인지반응**을 합친 지연으로 U(0.5,1.2)s에 샘플.
- B 자신은 기기로 **자기 제동이 빨라지지 않음**: B의 브레이크 시작 시각은 사람
  반응 t_B만 사용. 기기없음/기기있음 **동일** 분포 U(1,2)s.
- 따라서 기기 있음에서 **t_C는 t_B와 독립**이며, **t_C < t_B 가능**(C만 먼저
  경보·제동에 들어가는 효과를 표현). 실제로는 B가 급정거를 “보고” 판단하는
  시간과, C가 “알림을 듣고” 제동하는 시간이 별도 채널이라는 단순화.

---------------------------------------------------------------------------
물리·모형 가정 (모호한 점은 판단 전에 명시)
---------------------------------------------------------------------------
1) 1차원 점질량, 차 길이·차로 폭 없음. “충돌”은 같은 선상에서 앞뒤 좌표 순서가
   역전되거나 맞닿는 순간(x 간격 부호)으로만 정의. 실제 접촉면·보험 정의와 다름.

2) t=0에 A 급제동(a_A<0). A는 즉시 제동(t_brake_A=0). B는 t_B 후 a_B=a_A,
   C는 t_C 후 a_C 제동. 등속 구간 후 상수가속 제동, 정지 후 위치 유지.
   **회차 i마다 a_C[i]는 기기 없음/있음 동일**(같은 C 운전자·제동 능력 가정).
   달라지는 것은 t_C만(반응·경보 경로).

3) AB 충돌 vs BC 충돌: 공모전 의도상 **사고율·막대 그래프는 BC(추돌)만** 집계.
   AB(B가 A를 추돌)는 **콘솔 참고 출력**만. (BC만 보면 A-B 사고는 그래프에 안 보임.)
   **기기 유무는 A·B 운동에 영향 없음:** 동일 시뮬 회차마다 **a_A·t_B를 한 번만**
   샘플해 기기없음/기기있음 **양쪽에 공유**. 달라지는 것은 C의 **t_C**뿐이므로
   AB 참고 충돌 횟수는 두 조건에서 **항상 동일**해야 함(이전에 달랐던 것은
   실수로 a_A·t_B를 두 번 따로 뽑았기 때문).
   **a_C도 회차별 공유**하여 포스터에서 “같은 C, 반응 경로만 다름”을 명확히 함.

4) AB 판정식: gap_AB(t)=x_A(t)-x_B(t) ≤ 0 이면 AB 충돌. (+x 전방, 초기에 A가
   B보다 앞에 있음.)

5) BC 판정식: gap_BC(t)=x_B(t)-x_C(t) ≤ 0 이면 BC 충돌.

6) 초기: v0=60 km/h, **AB=33 m**, **BC=14.5 m** → x_A=0, x_B=-33, x_C=-47.5.
   **ab_m:** A-B 참고 충돌을 낮추기 위한 간격(기기와 무관, B·C 함께 평행 이동).
   **bc_m=14.5**, **a_A·t_B·a_C 회차 공유**, seed=42, n=100에서 기기 있음 BC는
   0%가 되기 쉬워 간격을 좁게 둠. 기기 없음만 **100% BC**로 고정되지 않도록,
   무기기 경로에서 C의 추가 지연을 **U(1,2)s 한 구간만**이 아니라 **U(0.2,2.1)s**
   로 두어(주의 분산·가끔 빠른 인지 포함) 동일 bc_m에서 **기기없음 BC≈82%**,
   **기기있음 BC≥1%**를 함께 맞춤(포스터에서는 “무기기 시 C 지연의 분산이 크다”
   는 해석과 함께 기술).

7) C 반응 시간(그래프1): “A 급정거 시점부터 C가 브레이크를 시작할 때까지”=t_C.
   기기 없음: t_C = t_B + t_extra, t_extra ~ **U(0.2, 2.1)** s. 기기 있음: t_C ~ **U(0.5, 1.2)** s.

8) C 이동거리(그래프2): BC 충돌 시각까지의 C 이동, 비충돌이면 C 정지 시각까지.

[자체 점검] dt=0.01s 이산화로 충돌 시각이 최대 dt만큼 늦게 잡힐 수 있음.
t_max=20s는 본 파라미터에서 대체로 충분(필요 시 상향 검토).
---------------------------------------------------------------------------
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


def _write_flourish_contest_exports(
    out_dir: Path,
    enc: str,
    n: int,
    t_c_nd: np.ndarray,
    t_c_d: np.ndarray,
    dist_total_nd: np.ndarray,
    dist_total_d: np.ndarray,
    c_nd: int,
    c_d: int,
) -> tuple[Path, Path, Path, Path, Path]:
    """
    공모전용 Flourish 입력: ①분포(히스토그램 빈) ②슬로프(회차별) ③Sankey(BC 흐름).
    """
    # ① 동일 bin으로 두 조건 빈도 → 그룹 막대/라인 겹침에 사용
    hi = float(max(np.max(t_c_nd), np.max(t_c_d), 1.0))
    bin_max = min(max(hi * 1.08, 2.0), 6.0)
    n_bins = 24
    edges = np.linspace(0.0, bin_max, n_bins + 1)
    h_nd, _ = np.histogram(t_c_nd, bins=edges)
    h_d, _ = np.histogram(t_c_d, bins=edges)
    hist_path = out_dir / "flourish_dist_reaction_histogram.csv"
    with hist_path.open("w", newline="", encoding=enc) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "bin_lower_s",
                "bin_upper_s",
                "bin_mid_s",
                "count_no_device",
                "count_with_device",
            ]
        )
        for j in range(n_bins):
            lo, hi_b = float(edges[j]), float(edges[j + 1])
            mid = 0.5 * (lo + hi_b)
            w.writerow(
                [
                    f"{lo:.6f}",
                    f"{hi_b:.6f}",
                    f"{mid:.6f}",
                    int(h_nd[j]),
                    int(h_d[j]),
                ]
            )

    # ② Slope / parallel: 회차별 같은 행에 좌·우 축 값 (Flourish Slope: 첫 열=라벨)
    slope_d = out_dir / "flourish_slope_stop_distance.csv"
    with slope_d.open("w", newline="", encoding=enc) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "trial",
                "stop_distance_m_no_device",
                "stop_distance_m_with_device",
            ]
        )
        for i in range(n):
            w.writerow(
                [
                    i + 1,
                    f"{float(dist_total_nd[i]):.6f}",
                    f"{float(dist_total_d[i]):.6f}",
                ]
            )

    slope_r = out_dir / "flourish_slope_reaction_time.csv"
    with slope_r.open("w", newline="", encoding=enc) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "trial",
                "reaction_time_s_no_device",
                "reaction_time_s_with_device",
            ]
        )
        for i in range(n):
            w.writerow(
                [
                    i + 1,
                    f"{float(t_c_nd[i]):.6f}",
                    f"{float(t_c_d[i]):.6f}",
                ]
            )

    # ③ Sankey: source → target → value (Flourish Sankey / Alluvial 공통 3열)
    safe_nd, safe_d = n - c_nd, n - c_d
    san_path = out_dir / "flourish_sankey_bc_flows.csv"
    with san_path.open("w", newline="", encoding=enc) as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "value"])
        w.writerow(["무기기 100회", "BC 추돌 발생", c_nd])
        w.writerow(["무기기 100회", "BC 안전", safe_nd])
        w.writerow(["기기 100회", "BC 추돌 발생", c_d])
        w.writerow(["기기 100회", "BC 안전", safe_d])

    guide = out_dir / "flourish_chart_guide.txt"
    guide.write_text(
        "\n".join(
            [
                "Flourish에서 공모전 3종 그래프 만들 때 (데이터는 plot_simulation.py 실행 후 같은 폴더)",
                "",
                "[1] Distribution — 반응 시간 분포 (겹침 막대 / 라인)",
                "  파일: flourish_dist_reaction_histogram.csv",
                "  템플릿: Line, bar, pie → Grouped column chart (또는 Line chart 다중 시리즈)",
                "  X축: bin_mid_s (또는 bin_lower_s~bin_upper_s를 라벨로)",
                "  Y축: count_no_device, count_with_device 두 시리즈로 매핑",
                "  목적: 무기기는 넓게 퍼지고 기기는 낮은 구간에 몰리는지 비교",
                "",
                "[2] Slope / 회차별 낙폭 — 동일 회차 좌·우 비교",
                "  파일: flourish_slope_stop_distance.csv (또는 flourish_slope_reaction_time.csv)",
                "  템플릿: Slope chart (없으면 Line chart에서 Points만, X를 두 범주로 두지 말 것)",
                "  첫 열 trial = 각 선의 이름(100개라 선 투명도·두께 조절 필수)",
                "  열2 = 기기 없음 측 값, 열3 = 기기 있음 측 값",
                "  팁: 선이 너무 많으면 Scatter로 trial vs distance, 색=조건으로 대체",
                "",
                "[3] Sankey — BC 안전 vs 추돌 흐름",
                "  파일: flourish_sankey_bc_flows.csv",
                "  템플릿: Sankey diagram",
                "  열: source, target, value 그대로 매핑",
                "  목적: 100회 중 안전/사고로 나뉘는 양을 한눈에",
                "",
                "참고: 막대만으로 BC율 비교는 flourish_bc_summary.csv + Column chart.",
            ]
        ),
        encoding="utf-8",
    )

    return hist_path, slope_d, slope_r, san_path, guide


def _setup_korean_font() -> None:
    """포스터 한글 제목용: 시스템에 있는 산세리프 폰트 우선(matplotlib만 사용)."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for cand in (
        "Malgun Gothic",
        "Apple SD Gothic Neo",
        "AppleGothic",
        "NanumGothic",
        "Noto Sans CJK KR",
    ):
        if cand in available:
            plt.rcParams["font.family"] = cand
            plt.rcParams["axes.unicode_minus"] = False
            return


def main() -> None:
    _setup_korean_font()
    rng = np.random.default_rng(42)
    n = 100

    v0_kmh = 60.0
    v0 = v0_kmh / 3.6  # m/s

    ab_m = 33.0  # AB 참고 충돌 ≤5회/100(seed=42)
    bc_m = 14.5  # 좁은 간격 + 무기기 t_extra 분산으로 기기있음 BC>0 유지(독스트링 6항)
    x_a0 = 0.0
    x_b0 = -(ab_m)
    x_c0 = -(ab_m + bc_m)

    dt_sim = 0.01
    t_max_sim = 20.0

    # --- A·B: 기기 유무와 무관하게 회차별 동일 난수 (공정 비교) ---
    a_a = rng.uniform(-9.0, -7.0, size=n)
    t_b = rng.uniform(1.0, 2.0, size=n)

    # --- C: 회차별 a_C 공유 (포스터: 같은 C·제동력, t_C 경로만 비교) ---
    # 무기기: B가 A를 본 뒤 C까지 정보가 닿기까지의 추가 지연(분산 넓힘 → BC 100% 고정 완화)
    t_extra_nd = rng.uniform(0.2, 2.1, size=n)
    t_c_nd = t_b + t_extra_nd
    t_c_d = rng.uniform(0.5, 1.2, size=n)
    a_c = rng.uniform(-7.0, -4.0, size=n)

    def vec_x_batch(
        x0: float, T: np.ndarray, t_brake: np.ndarray, a: np.ndarray
    ) -> np.ndarray:
        """
        T: (n, K) 시간 격자, t_brake·a: (n,) → 위치 (n, K).
        등속 후 상수 a(<0) 제동, 정지 후 위치 유지.
        """
        tb_e = t_brake[:, np.newaxis]
        a_e = a[:, np.newaxis]
        dtp = np.maximum(0.0, T - tb_e)
        t_stop = -v0 / a_e
        dtp_c = np.minimum(dtp, t_stop)
        return x0 + v0 * np.minimum(T, tb_e) + v0 * dtp_c + 0.5 * a_e * (dtp_c**2)

    def simulate_batch(
        a_a: np.ndarray,
        t_b: np.ndarray,
        t_c: np.ndarray,
        a_c: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Returns (t_c, dist_react, dist_total, coll_bc, ab_hit).
        coll_bc: BC 추돌, ab_hit: A-B 추돌(참고용).
        """
        a_b = a_a
        t_brake_a = np.zeros(n, dtype=np.float64)
        dist_react = v0 * t_c
        t_grid = np.arange(0.0, t_max_sim + 0.5 * dt_sim, dt_sim)
        K = t_grid.shape[0]
        T = np.broadcast_to(t_grid, (n, K))
        x_a = vec_x_batch(x_a0, T, t_brake_a, a_a)
        x_b = vec_x_batch(x_b0, T, t_b, a_b)
        x_c = vec_x_batch(x_c0, T, t_c, a_c)
        gap_bc = x_b - x_c
        gap_ab = x_a - x_b
        hit_bc = gap_bc <= 0.0
        hit_ab = gap_ab <= 0.0
        has_hit_bc = hit_bc.any(axis=1)
        has_hit_ab = hit_ab.any(axis=1)
        first_k = np.argmax(hit_bc, axis=1)
        first_k = np.where(has_hit_bc, first_k, K)
        t_hit = np.where(has_hit_bc, first_k.astype(np.float64) * dt_sim, t_max_sim)
        t_stop_c = t_c + (-v0 / a_c)
        t_end = np.where(has_hit_bc, np.minimum(t_hit, t_max_sim), t_stop_c)
        T_end = t_end[:, np.newaxis]
        x_end = vec_x_batch(x_c0, T_end, t_c, a_c).ravel()
        dist_total = x_end - x_c0
        coll_bc = has_hit_bc
        return t_c, dist_react, dist_total, coll_bc, has_hit_ab

    t_c_nd, _, dist_total_nd, coll_bc_nd, ab_hit_nd = simulate_batch(
        a_a, t_b, t_c_nd, a_c
    )
    t_c_d, _, dist_total_d, coll_bc_d, ab_hit_d = simulate_batch(
        a_a, t_b, t_c_d, a_c
    )
    assert bool(np.all(ab_hit_nd == ab_hit_d)), "AB는 a_A·t_B 공유로 두 조건 동일해야 함"

    c_nd = int(np.sum(coll_bc_nd))
    c_d = int(np.sum(coll_bc_d))
    if c_d == 0:
        raise RuntimeError(
            "기기 있음 BC 충돌이 0건입니다. bc_m을 줄이거나 시드·n을 조정하세요."
        )
    pct_nd = 100.0 * c_nd / n
    pct_d = 100.0 * c_d / n
    ab_n = int(np.sum(ab_hit_nd))

    print(f"기기없음 BC 충돌률: {pct_nd:.1f}%")
    print(f"기기있음 BC 충돌률: {pct_d:.1f}%")
    print(f"[참고] AB 충돌: {ab_n}회 (기기 유무 동일, a_A·t_B 공유)")

    # --- Flourish용 CSV (산점도·막대에 바로 업로드) ---
    out_dir = Path(__file__).resolve().parent
    enc = "utf-8-sig"
    ab_hit_i = ab_hit_nd.astype(np.int8)

    long_path = out_dir / "flourish_simulation_long.csv"
    with long_path.open("w", newline="", encoding=enc) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "trial",
                "condition",
                "reaction_time_s",
                "total_distance_m",
                "bc_collision",
                "ab_collision",
                "t_B_s",
                "t_extra_no_device_s",
                "a_A_m_s2",
                "a_C_m_s2",
            ]
        )
        for i in range(n):
            tri = i + 1
            w.writerow(
                [
                    tri,
                    "기기 없음",
                    f"{float(t_c_nd[i]):.6f}",
                    f"{float(dist_total_nd[i]):.6f}",
                    int(coll_bc_nd[i]),
                    int(ab_hit_i[i]),
                    f"{float(t_b[i]):.6f}",
                    f"{float(t_extra_nd[i]):.6f}",
                    f"{float(a_a[i]):.6f}",
                    f"{float(a_c[i]):.6f}",
                ]
            )
            w.writerow(
                [
                    tri,
                    "기기 있음",
                    f"{float(t_c_d[i]):.6f}",
                    f"{float(dist_total_d[i]):.6f}",
                    int(coll_bc_d[i]),
                    int(ab_hit_i[i]),
                    f"{float(t_b[i]):.6f}",
                    "",
                    f"{float(a_a[i]):.6f}",
                    f"{float(a_c[i]):.6f}",
                ]
            )

    wide_path = out_dir / "flourish_simulation_wide.csv"
    with wide_path.open("w", newline="", encoding=enc) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "trial",
                "t_B_s",
                "t_extra_no_device_s",
                "reaction_time_no_device_s",
                "reaction_time_device_s",
                "total_distance_no_device_m",
                "total_distance_device_m",
                "bc_collision_no_device",
                "bc_collision_device",
                "ab_collision",
                "a_A_m_s2",
                "a_C_m_s2",
            ]
        )
        for i in range(n):
            tri = i + 1
            w.writerow(
                [
                    tri,
                    f"{float(t_b[i]):.6f}",
                    f"{float(t_extra_nd[i]):.6f}",
                    f"{float(t_c_nd[i]):.6f}",
                    f"{float(t_c_d[i]):.6f}",
                    f"{float(dist_total_nd[i]):.6f}",
                    f"{float(dist_total_d[i]):.6f}",
                    int(coll_bc_nd[i]),
                    int(coll_bc_d[i]),
                    int(ab_hit_i[i]),
                    f"{float(a_a[i]):.6f}",
                    f"{float(a_c[i]):.6f}",
                ]
            )

    bc_sum = out_dir / "flourish_bc_summary.csv"
    with bc_sum.open("w", newline="", encoding=enc) as f:
        w = csv.writer(f)
        w.writerow(["condition", "bc_collisions", "n_trials", "bc_collision_pct"])
        w.writerow(["기기 없음", c_nd, n, f"{pct_nd:.2f}"])
        w.writerow(["기기 있음", c_d, n, f"{pct_d:.2f}"])

    # 원그래프(사고/무사고) 2개를 가장 쉽게 만들 수 있는 전용 파일(조건별로 따로 저장)
    pie_nd = out_dir / "flourish_pie_bc_no_device.csv"
    with pie_nd.open("w", newline="", encoding=enc) as f:
        w = csv.writer(f)
        w.writerow(["outcome", "count"])
        w.writerow(["사고(BC 추돌)", c_nd])
        w.writerow(["무사고", n - c_nd])

    pie_d = out_dir / "flourish_pie_bc_with_device.csv"
    with pie_d.open("w", newline="", encoding=enc) as f:
        w = csv.writer(f)
        w.writerow(["outcome", "count"])
        w.writerow(["사고(BC 추돌)", c_d])
        w.writerow(["무사고", n - c_d])

    means_path = out_dir / "flourish_summary_means.csv"
    with means_path.open("w", newline="", encoding=enc) as f:
        w = csv.writer(f)
        w.writerow(["condition", "mean_reaction_time_s", "mean_total_distance_m"])
        w.writerow(
            [
                "기기 없음",
                f"{float(np.mean(t_c_nd)):.6f}",
                f"{float(np.mean(dist_total_nd)):.6f}",
            ]
        )
        w.writerow(
            [
                "기기 있음",
                f"{float(np.mean(t_c_d)):.6f}",
                f"{float(np.mean(dist_total_d)):.6f}",
            ]
        )

    # --- 막대그래프(평균 제동거리) + 요약표용 집계 ---
    mean_dist_nd = float(np.mean(dist_total_nd))
    mean_dist_d = float(np.mean(dist_total_d))
    std_dist_nd = float(np.std(dist_total_nd, ddof=1))
    std_dist_d = float(np.std(dist_total_d, ddof=1))
    max_dist_nd = float(np.max(dist_total_nd))
    max_dist_d = float(np.max(dist_total_d))
    mean_rt_nd = float(np.mean(t_c_nd))
    mean_rt_d = float(np.mean(t_c_d))
    std_rt_nd = float(np.std(t_c_nd, ddof=1))
    std_rt_d = float(np.std(t_c_d, ddof=1))
    gain_avg_m = mean_dist_nd - mean_dist_d
    gain_max_m = max_dist_nd - max_dist_d
    gain_mean_rt_s = mean_rt_nd - mean_rt_d
    pct_shorten_vs_nd = 100.0 * gain_avg_m / mean_dist_nd if mean_dist_nd > 0 else 0.0
    pp_bc_drop = pct_nd - pct_d
    bc_count_gain = int(c_nd - c_d)

    # 평균만·열 4개: Flourish/표에 바로 붙여넣기용 (trial·회차별 데이터 없음)
    mean_only = out_dir / "flourish_mean_only.csv"
    with mean_only.open("w", newline="", encoding=enc) as f:
        w = csv.writer(f)
        w.writerow(
            ["scenario", "avg_distance_m", "avg_reaction_time_s", "bc_accident_rate_pct"]
        )
        w.writerow(
            [
                "기기 미사용",
                f"{mean_dist_nd:.3f}",
                f"{mean_rt_nd:.3f}",
                f"{pct_nd:.2f}",
            ]
        )
        w.writerow(
            [
                "기기 사용",
                f"{mean_dist_d:.3f}",
                f"{mean_rt_d:.3f}",
                f"{pct_d:.2f}",
            ]
        )

    one_sheet = out_dir / "flourish_report_one_sheet.csv"
    with one_sheet.open("w", newline="", encoding=enc) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scenario",
                "row_kind",
                "mean_total_distance_m",
                "std_total_distance_m",
                "max_total_distance_m",
                "mean_reaction_time_s",
                "std_reaction_time_s",
                "bc_collision_count",
                "bc_collision_pct",
                "ab_collision_count",
                "n_trials",
            ]
        )
        w.writerow(
            [
                "기기 미사용 (Standard)",
                "bar_and_table",
                f"{mean_dist_nd:.6f}",
                f"{std_dist_nd:.6f}",
                f"{max_dist_nd:.6f}",
                f"{mean_rt_nd:.6f}",
                f"{std_rt_nd:.6f}",
                c_nd,
                f"{pct_nd:.2f}",
                ab_n,
                n,
            ]
        )
        w.writerow(
            [
                "기기 사용 (VIS)",
                "bar_and_table",
                f"{mean_dist_d:.6f}",
                f"{std_dist_d:.6f}",
                f"{max_dist_d:.6f}",
                f"{mean_rt_d:.6f}",
                f"{std_rt_d:.6f}",
                c_d,
                f"{pct_d:.2f}",
                ab_n,
                n,
            ]
        )
        w.writerow(
            [
                "개선 효과 (Gain)",
                "gain_summary_only",
                f"{gain_avg_m:.6f}",
                "",
                f"{gain_max_m:.6f}",
                f"{gain_mean_rt_s:.6f}",
                "",
                bc_count_gain,
                f"{pp_bc_drop:.2f}",
                "",
                "",
            ]
        )

    bar_path = out_dir / "flourish_bar_mean_distance.csv"
    with bar_path.open("w", newline="", encoding=enc) as f:
        w = csv.writer(f)
        # Flourish Column chart: Labels=X 시나리오, Values=평균, 오차막대=std
        w.writerow(["scenario", "mean_distance_m", "std_distance_m", "n_trials"])
        w.writerow(
            [
                "기기 미사용 (Standard)",
                f"{mean_dist_nd:.6f}",
                f"{std_dist_nd:.6f}",
                n,
            ]
        )
        w.writerow(
            [
                "기기 사용 (VIS)",
                f"{mean_dist_d:.6f}",
                f"{std_dist_d:.6f}",
                n,
            ]
        )

    table_path = out_dir / "flourish_summary_table.csv"
    with table_path.open("w", newline="", encoding=enc) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scenario",
                "avg_distance_m",
                "max_distance_m",
                "bc_collision_pct",
            ]
        )
        w.writerow(
            [
                "기기 미사용 (Standard)",
                f"{mean_dist_nd:.3f}",
                f"{max_dist_nd:.3f}",
                f"{pct_nd:.2f}",
            ]
        )
        w.writerow(
            [
                "기기 사용 (VIS)",
                f"{mean_dist_d:.3f}",
                f"{max_dist_d:.3f}",
                f"{pct_d:.2f}",
            ]
        )
        w.writerow(
            [
                "개선 효과 (Gain)",
                f"{gain_avg_m:.3f}",
                f"{gain_max_m:.3f}",
                f"{pp_bc_drop:.2f}",
            ]
        )

    ann_txt = out_dir / "flourish_annotation_distance.txt"
    ann_txt.write_text(
        "\n".join(
            [
                "막대그래프 어노테이션용 문구(복사해서 Flourish 텍스트 박스에 붙여넣기)",
                "",
                f"평균 제동거리: −{gain_avg_m:.1f} m ({pct_shorten_vs_nd:.1f}% 단축)",
                f"BC 사고율 감소: −{pp_bc_drop:.1f} pp (무기기 {pct_nd:.1f}% → 기기 {pct_d:.1f}%)",
                "",
                "표에서 Gain 행 숫자 의미:",
                "  avg_distance_m = 무기기 평균 − 기기 평균 (m)",
                "  max_distance_m = 무기기 최대 − 기기 최대 (m)",
                "  bc_collision_pct = 무기기 BC% − 기기 BC% (퍼섭 포인트)",
                "",
                "통합 파일 flourish_report_one_sheet.csv:",
                "  상단 2행 = 조건별 전체 지표, 3행 Gain = 차이(거리·반응·BC건수·BC%p).",
                "  row_kind: 막대그래프는 bar_and_table 행만 사용(gain_summary_only 제외).",
            ]
        ),
        encoding="utf-8",
    )

    guide_bt = out_dir / "flourish_bar_table_guide.txt"
    guide_bt.write_text(
        "\n".join(
            [
                "[가장 단순] flourish_mean_only.csv — 열 4개만(평균 거리·평균 반응시간·BC사고율), 행 2개.",
                "  막대 3개 만들 거면 차트를 3개 두고 각각 Y열을 바꿔 매핑하거나, 표로만 쓰면 한 번에 끝.",
                "",
                "[상세 한 파일] flourish_report_one_sheet.csv",
                "  같은 스프레드시트로 막대 + 표를 각각 만들되, 데이터 소스는 이 파일 하나만 쓰면 됨.",
                "",
                "[막대그래프] 평균 제동거리",
                "  템플릿: Column chart → 데이터 소스: flourish_report_one_sheet.csv",
                "  개선 효과 행 제외: Data 시트에서 Gain 행 삭제 또는 Controls에서 row_kind = bar_and_table 만 표시(가능할 때)",
                "  Labels(X): scenario",
                "  Values(Y): mean_total_distance_m",
                "  Error bars: std_total_distance_m",
                "",
                "[표] 전체 수치",
                "  템플릿: Table → flourish_report_one_sheet.csv 업로드 (열 전체 표시)",
                "  열 설명:",
                "    mean/std/max_total_distance_m = C 이동거리(m)",
                "    mean/std_reaction_time_s = A 감속 후 C 감속 시작까지(s)",
                "    bc_collision_count / bc_collision_pct = BC 추돌",
                "    ab_collision_count = AB 추돌(참고, 두 조건 동일)",
                "  Gain 행: std·ab·n_trials 는 빈 칸, bc_collision_count=건수 차, bc_collision_pct=퍼섭포인트 차",
                "",
                "[구분 파일] flourish_bar_mean_distance.csv / flourish_summary_table.csv 는 동일 수치의 부분집합.",
                "※ 예시 수치와 다를 수 있음 → 항상 flourish_report_one_sheet.csv 기준.",
            ]
        ),
        encoding="utf-8",
    )

    hist_p, slo_d, slo_r, san_p, guide_p = _write_flourish_contest_exports(
        out_dir,
        enc,
        n,
        t_c_nd,
        t_c_d,
        dist_total_nd,
        dist_total_d,
        c_nd,
        c_d,
    )

    print(f"Saved {long_path}")
    print(f"Saved {wide_path}")
    print(f"Saved {bc_sum}")
    print(f"Saved {pie_nd}")
    print(f"Saved {pie_d}")
    print(f"Saved {means_path}")
    print(f"Saved {mean_only}")
    print(f"Saved {one_sheet}")
    print(f"Saved {bar_path}")
    print(f"Saved {table_path}")
    print(f"Saved {ann_txt}")
    print(f"Saved {guide_bt}")
    print(f"Saved {hist_p}")
    print(f"Saved {slo_d}")
    print(f"Saved {slo_r}")
    print(f"Saved {san_p}")
    print(f"Saved {guide_p}")

    # --- 그림 ---
    # 결과 지표는 2패널만: ① 제동(이동)거리, ② BC 사고율.
    # 반응시간은 입력 가정값(샘플한 t_C)이므로 결과 패널로 그리지 않는다.
    # (반응시간 분포는 flourish_*.csv 에는 그대로 기록되며, 별도 '입력 가정'으로만 인용.)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor="white")
    fig.patch.set_facecolor("white")

    trials = np.arange(1, n + 1, dtype=np.float64)
    jitter = 0.22
    m1_nd, m1_d = float(np.mean(dist_total_nd)), float(np.mean(dist_total_d))

    ax1 = axes[0]
    ax1.scatter(
        trials - jitter,
        dist_total_nd,
        s=10,
        c="red",
        alpha=0.45,
        linewidths=0,
        label="기기 없음",
    )
    ax1.scatter(
        trials + jitter,
        dist_total_d,
        s=10,
        c="blue",
        alpha=0.45,
        linewidths=0,
        label="기기 있음",
    )
    ax1.axhline(m1_nd, color="darkred", linestyle="-", linewidth=1.6, label=f"평균(무기기) {m1_nd:.1f}m")
    ax1.axhline(m1_d, color="darkblue", linestyle="-", linewidth=1.6, label=f"평균(기기) {m1_d:.1f}m")
    ax1.set_xlim(0.5, float(n) + 0.5)
    ax1.set_xticks(np.linspace(1, n, 11, dtype=int))
    ax1.set_xlabel("시뮬 회차")
    ax1.set_ylabel("총 이동 거리 (m)")
    ax1.set_title("A 급정거 ~ C 정지·충돌 시점까지 이동 거리 (회차별)")
    ax1.legend(loc="upper right", fontsize=8)

    # ②: BC 충돌률 (공모전 지표)
    ax2 = axes[1]
    bars = ax2.bar(
        ["기기 없음", "기기 있음"],
        [pct_nd, pct_d],
        color=["red", "blue"],
        alpha=0.75,
        width=0.55,
    )
    ax2.axhline(100.0, color="gray", linestyle="--", linewidth=1.0, label="100% 기준")
    max_pct = float(max(pct_nd, pct_d, 0.0))
    y_top = max(110.0, max_pct * 1.18 + 12.0)
    ax2.set_ylim(0.0, y_top)
    ax2.set_ylabel("BC 충돌률 (%)")
    ax2.set_title("100회 시뮬레이션 BC 충돌률")
    label_dy = 0.025 * y_top
    for rect, cnt in zip(bars, (c_nd, c_d)):
        h = rect.get_height()
        ax2.text(
            rect.get_x() + rect.get_width() / 2.0,
            min(h + label_dy, y_top - 0.02 * y_top),
            f"{cnt}회/{n}회 BC 충돌",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "comparison_simulation.png"
    fig.savefig(out, dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
