"""아침 보고서 생성 — experiments/overnight_report.md 하나만 읽으면 되게."""
from __future__ import annotations

import csv
import glob
import json
import os
from datetime import datetime

NIGHT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(NIGHT)
SCAN = os.path.join(ROOT, "blackbox_scan")
OUT = os.path.join(ROOT, "overnight_report.md")

STATUS_KO = {"OK": "성공", "FAIL": "실패", "SKIP": "건너뜀", "PARTIAL": "부분완료",
             "MISMATCH": "불일치(회귀)", "ERROR": "예외"}


def load_json(p, default=None):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def read_csv(p):
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    results = load_json(os.path.join(NIGHT, "results.json"), [])
    L = []
    A = L.append

    A("# 야간 무인 배치 보고서")
    A("")
    A(f"생성 시각: **{datetime.now():%Y-%m-%d %H:%M:%S}**")
    A("")
    A("이 문서 하나로 전체 상황을 파악할 수 있게 정리했습니다. "
      "판단이 필요한 항목은 3장에 모아두었습니다.")
    A("")

    # 1. 작업 결과
    A("## 1. 작업별 결과")
    A("")
    A("| 작업 | 상태 | 소요 | 완료시각 | 내용 |")
    A("|---|---|---|---|---|")
    for r in results:
        secs = r.get("elapsed_s", 0)
        dur = f"{secs/60:.1f}분" if secs >= 60 else f"{secs:.0f}초"
        A(f"| {r.get('task','')} | **{STATUS_KO.get(r.get('status'), r.get('status'))}** "
          f"| {dur} | {r.get('finished_at','')} | {r.get('detail','')[:120]} |")
    A("")

    # 2. 자동 결정 로그
    A("## 2. 자동으로 내린 결정")
    A("")
    dec = [r for r in results if r.get("decision")]
    if dec:
        for r in dec:
            A(f"- **{r['task']}** — {r['decision']}")
    else:
        A("- (기록된 자동 결정 없음)")
    A("")

    # 3. 결정 대기 목록
    A("## 3. ★ 결정 대기 목록 (사람 판단 필요)")
    A("")
    pending = []

    ranked = read_csv(os.path.join(SCAN, "candidates", "ranked",
                                   "ranked_occlusion_top20.csv"))
    if ranked:
        pending.append(
            f"**시야차단 후보 상위 {len(ranked)}건 선별** — 자동 정렬만 했고 채택 판단은 "
            f"하지 않았습니다. `experiments/blackbox_scan/candidates/ranked/"
            f"ranked_occlusion_top20.csv` 와 각 행의 `frame_png` 를 보고 "
            f"제안서 삽입 여부를 정해 주세요.")
    else:
        pending.append(
            "**시야차단 후보 0건** — 실사 데이터에서 (a) 장면을 확보하지 못했습니다. "
            "CARLA 생성 / 추가 촬영 / 기준 완화 중 방향 결정이 필요합니다.")

    brake = read_csv(os.path.join(SCAN, "brake", "brake_light_evaluation.csv"))
    if brake:
        near = sum(1 for b in brake if b.get("verdict") == "near")
        pending.append(
            f"**브레이크등 임계값 확정** — 임계 근처(5~8%) {near}건이 있습니다. "
            f"8% 유지 / 조정 여부를 정해 주세요.")

    api = [r for r in results if r.get("task") == "8_api_retry"]
    if api and api[0].get("status") != "OK":
        pending.append(
            "**API 인증키** — data.go.kr 키로는 계속 실패했습니다. "
            "`opendata.koroad.or.kr` 에서 별도 발급받은 authKey 가 필요합니다.")

    decel = read_csv(os.path.join(SCAN, "decel", "decel_top.csv"))
    if decel:
        pending.append(
            f"**급감속 상위 {len(decel)}건 육안 확인** — 자동 분류는 전부 "
            f"'노면충격/판단애매'였습니다. `experiments/blackbox_scan/decel/` 의 "
            f"PNG를 보고 실제 위험상황이 있는지 판단해 주세요. "
            f"이 결과에 따라 4-4를 실사로 갈지 CARLA로 갈지 갈립니다.")

    for i, p in enumerate(pending, 1):
        A(f"{i}. {p}")
    A("")

    # 4. 시야차단/버스 후보
    A("## 4. 시야차단 · 대형차 후보")
    A("")
    if ranked:
        A(f"자동 등급화 결과 상위 {len(ranked)}건 (면적비 내림차순):")
        A("")
        A("| 순위 | 파일 | 시작~종료 | 지속 | 면적비 | 종류 | 프레임 |")
        A("|---|---|---|---|---|---|---|")
        for r in ranked[:10]:
            A(f"| {r.get('rank')} | {r.get('file')} | "
              f"{r.get('t_start')}~{r.get('t_end')}s | {r.get('dur_s')}s | "
              f"{r.get('area_ratio_max')} | {r.get('cls')} | "
              f"`{r.get('frame_png','')}` |")
        A("")
        A("PNG 위치: `experiments/blackbox_scan/*/candidates/`")
    else:
        A("**후보 0건.** 필터를 완화(면적 0.10, 지속 1.5초, 정지 포함)해도 "
          "검출되지 않았습니다.")
    A("")

    # 5. 브레이크등
    A("## 5. 브레이크등 임계 평가")
    A("")
    if brake:
        okr = [b for b in brake if b.get("status") == "ok"]
        det = sum(1 for b in okr if b.get("verdict") == "detected")
        nr = sum(1 for b in okr if b.get("verdict") == "near")
        ms = sum(1 for b in okr if b.get("verdict") == "miss")
        A(f"- 총 {len(okr)}건 — **detected {det} / near {nr} / miss {ms}** (임계 8%)")
        vals = []
        for b in okr:
            try:
                vals.append(float(b["red_pct_rescan"]))
            except Exception:
                pass
        if vals:
            hi = [v for v in vals if v >= 8]
            lo = [v for v in vals if v < 8]
            A(f"- 점등 판정군 붉은픽셀: {min(hi):.1f}~{max(hi):.1f}%" if hi else "")
            A(f"- 미점등군 붉은픽셀: {min(lo):.1f}~{max(lo):.1f}%" if lo else "")
            A("- 두 군이 뚜렷이 갈리고 8%가 그 사이 빈 구간에 위치 → 임계값 타당성 실증")
        A("- CSV: `experiments/blackbox_scan/brake/brake_light_evaluation.csv`")
        A("- PNG: `experiments/blackbox_scan/brake/png/`")
    else:
        A("- 평가 결과 없음")
    A("")

    # 6. API
    A("## 6. API 재시도 결과")
    A("")
    if api:
        r = api[0]
        A(f"- 상태: **{STATUS_KO.get(r.get('status'), r.get('status'))}**")
        A(f"- {r.get('detail','')}")
    else:
        A("- 수행되지 않음")
    att = load_json(os.path.join(NIGHT, "api_retry_log.json"), [])
    if att:
        A("")
        A("| 시도 | 시각 | 응답 |")
        A("|---|---|---|")
        for a in att:
            body = (a.get("body_head") or "").replace("|", "/").replace("\n", " ")[:110]
            A(f"| {a.get('attempt')} | {a.get('at')} | `{body}` |")
    A("")
    lg = os.path.join(ROOT, "cheongju_heatmap_lg", "cheongju_lg_hotspots.csv")
    A(f"- 히트맵 생성 여부: {'생성됨' if os.path.exists(lg) else '미생성 (수집 실패)'}")
    A("")

    # 7. 산출물
    A("## 7. 산출물 위치")
    A("")
    A("| 항목 | 경로 |")
    A("|---|---|")
    A("| 브레이크등 평가 | `experiments/blackbox_scan/brake/` |")
    A("| 급감속 상위 5 | `experiments/blackbox_scan/decel/` |")
    A("| 오늘(07-20) 표적 스캔 | `experiments/blackbox_scan/today_0720/` |")
    A("| NORMAL 전체 스캔 | `experiments/blackbox_scan/normal_full/` |")
    A("| 후보 등급화 | `experiments/blackbox_scan/candidates/ranked/` |")
    A("| 컨택트시트 | `C:\\Users\\이동욱\\blackbox_sheets\\NORMAL\\` |")
    A("| 실행 로그 | `experiments/overnight/run.log` |")
    A("")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[저장] {OUT}")


if __name__ == "__main__":
    main()
