"""
야간 무인 배치 오케스트레이터.

이 프로세스는 대화 세션과 독립적으로 끝까지 실행된다.
작업 4~8을 순서대로 수행하고, 각 단계의 성공/실패/소요시간을 기록한 뒤
마지막에 overnight_report.md 를 생성한다.

원칙 (사용자 지시)
- 한 작업이 실패해도 멈추지 않고 다음으로 넘어간다. 실패는 기록만 한다.
- 파일 삭제·덮어쓰기 금지. 산출물은 항상 새 이름으로 쓴다.
- 판단이 필요한 지점은 정해진 규칙대로 자동 결정하고 그 결정을 로그에 남긴다.
- 선별·채택 판단은 하지 않는다 (작업 7은 정렬까지만).
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # experiments/
REPO = os.path.dirname(ROOT)
SCAN = os.path.join(ROOT, "blackbox_scan")
NIGHT = os.path.join(ROOT, "overnight")
NORMAL_SRC = r"C:\Users\이동욱\blackbox_full\NORMAL"
LOG = os.path.join(NIGHT, "run.log")
RESULTS = os.path.join(NIGHT, "results.json")

os.makedirs(NIGHT, exist_ok=True)
results = []
if os.path.exists(RESULTS):
    try:
        results = json.load(open(RESULTS, encoding="utf-8"))
    except Exception:
        results = []


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def record(task: str, status: str, secs: float, detail: str = "",
           decision: str = "") -> None:
    results.append({"task": task, "status": status, "elapsed_s": round(secs, 1),
                    "detail": detail, "decision": decision,
                    "finished_at": f"{datetime.now():%Y-%m-%d %H:%M:%S}"})
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def run(cmd: list, cwd: str, timeout: int = 36000):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def count_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        return sum(1 for _ in csv.DictReader(f))


# ---------------------------------------------------------------- 4) ffmpeg 검증
def task4_validate_ffmpeg():
    t = time.time()
    log("TASK4 시작 — ffmpeg 파이프 방식 검증 (파일럿 3개 재스캔 후 비교)")
    pilot_csv = os.path.join(SCAN, "pilot", "NORMAL_PILOT_candidates.csv")
    if not os.path.exists(pilot_csv):
        record("4_ffmpeg_validate", "SKIP", time.time() - t, "파일럿 CSV 없음")
        return "cv2"
    with open(pilot_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    files = sorted({r["file"] for r in rows})[:3]
    if not files:
        record("4_ffmpeg_validate", "SKIP", time.time() - t, "파일럿 후보 없음")
        return "cv2"
    log(f"  검증 대상: {files}")

    out = os.path.join(SCAN, "validate_ffmpeg")
    rc, o = run([sys.executable, "scan_v3.py", "--src", NORMAL_SRC, "--out", out,
                 "--label", "VAL_FF", "--decode", "ffmpeg", "--only", *files], SCAN)
    if rc != 0:
        log(f"  ffmpeg 스캔 실패(rc={rc}) → cv2 방식으로 회귀")
        record("4_ffmpeg_validate", "FAIL", time.time() - t, o[-800:],
               "ffmpeg 실행 실패 → decode=cv2 채택")
        return "cv2"

    # 같은 파일을 기존(cv2) 방식으로도 스캔해 비교
    rc2, o2 = run([sys.executable, "scan_v3.py", "--src", NORMAL_SRC, "--out", out,
                   "--label", "VAL_CV", "--decode", "cv2", "--only", *files], SCAN)
    ff = count_rows(os.path.join(out, "VAL_FF_candidates.csv"))
    cv = count_rows(os.path.join(out, "VAL_CV_candidates.csv"))

    def per_type(lbl):
        p = os.path.join(out, f"{lbl}_candidates.csv")
        d = {}
        if os.path.exists(p):
            with open(p, newline="", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    d[r["type"]] = d.get(r["type"], 0) + 1
        return d

    tf, tc = per_type("VAL_FF"), per_type("VAL_CV")
    same = tf == tc
    try:
        sf = json.load(open(os.path.join(out, "VAL_FF_summary.json"), encoding="utf-8"))
        sc = json.load(open(os.path.join(out, "VAL_CV_summary.json"), encoding="utf-8"))
        speed = f"ffmpeg {sf.get('sec_per_file')}s/file vs cv2 {sc.get('sec_per_file')}s/file"
    except Exception:
        speed = ""
    detail = f"ffmpeg {ff}건 {tf} / cv2 {cv}건 {tc}. {speed}"
    if same:
        log(f"  일치 → ffmpeg 채택. {detail}")
        record("4_ffmpeg_validate", "OK", time.time() - t, detail,
               "검출 결과 일치 → decode=ffmpeg 채택")
        return "ffmpeg"
    log(f"  불일치 → cv2 회귀. {detail}")
    record("4_ffmpeg_validate", "MISMATCH", time.time() - t, detail,
           "검출 결과 불일치 → 기존 decode=cv2 로 회귀")
    return "cv2"


# ------------------------------------------------- 5) 오늘(07-20) NORMAL 표적 스캔
def task5_today(decode: str):
    t = time.time()
    log(f"TASK5 시작 — 07-20 NORMAL 표적 스캔 (정지 포함 필터, decode={decode})")
    files = sorted(f for f in os.listdir(NORMAL_SRC)
                   if f.startswith("20260720") and f.lower().endswith(".avi"))
    if not files:
        record("5_today_scan", "SKIP", time.time() - t, "07-20 파일 없음")
        return
    out = os.path.join(SCAN, "today_0720")
    # 정지 상태 포함: cy 하한을 낮추고(정지 시 앞차가 더 위로 보임) 면적 임계도 낮춤
    rc, o = run([sys.executable, "scan_v3.py", "--src", NORMAL_SRC, "--out", out,
                 "--label", "TODAY", "--decode", decode,
                 "--occl-area", "0.10", "--occl-min-sec", "1.5",
                 "--cx-min", "0.05", "--cx-max", "0.95", "--cy-min", "0.30",
                 "--only", *files], SCAN)
    n = count_rows(os.path.join(out, "TODAY_candidates.csv"))
    na = 0
    p = os.path.join(out, "TODAY_candidates.csv")
    if os.path.exists(p):
        with open(p, newline="", encoding="utf-8-sig") as f:
            na = sum(1 for r in csv.DictReader(f) if r["type"] == "a_occlusion")
    st = "OK" if rc == 0 else "FAIL"
    log(f"  {st} — {len(files)}개 대상, 후보 {n}건 (a_occlusion {na}건)")
    record("5_today_scan", st, time.time() - t,
           f"{len(files)}파일, 후보 {n}건, a_occlusion {na}건",
           "정지 포함 위해 occl_area 0.10 / min_sec 1.5 / cy_min 0.30 으로 완화")


# --------------------------------------------------------- 6) NORMAL 전체 스캔
def task6_full(decode: str):
    t = time.time()
    log(f"TASK6 시작 — NORMAL 전체 스캔 (체크포인트, decode={decode})")
    out = os.path.join(SCAN, "normal_full")
    rc, o = run([sys.executable, "scan_v3.py", "--src", NORMAL_SRC, "--out", out,
                 "--label", "NORMAL_FULL", "--decode", decode,
                 "--occl-area", "0.10", "--occl-min-sec", "1.5",
                 "--cx-min", "0.05", "--cx-max", "0.95", "--cy-min", "0.30"], SCAN)
    p = os.path.join(out, "NORMAL_FULL_candidates.csv")
    n = count_rows(p)
    by = {}
    if os.path.exists(p):
        with open(p, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                by[r["type"]] = by.get(r["type"], 0) + 1
    st = "OK" if rc == 0 else "PARTIAL"
    log(f"  {st} — 후보 {n}건 {by}")
    record("6_normal_full", st, time.time() - t, f"후보 {n}건 {by}",
           "동일 필터(정지 포함) 적용, 파일 단위 체크포인트로 재실행 시 이어받음")


# ---------------------------------------------------------- 7) 후보 자동 등급화
def task7_rank():
    t = time.time()
    log("TASK7 시작 — 후보 자동 등급화 (정렬까지만, 선별 판단 없음)")
    srcs = [os.path.join(SCAN, "normal_full", "NORMAL_FULL_candidates.csv"),
            os.path.join(SCAN, "today_0720", "TODAY_candidates.csv"),
            os.path.join(SCAN, "pilot", "NORMAL_PILOT_candidates.csv"),
            os.path.join(SCAN, "EVENT_candidates.csv")]
    rows = []
    for s in srcs:
        if not os.path.exists(s):
            continue
        setname = os.path.basename(os.path.dirname(s)) or "EVENT"
        with open(s, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                r["_set"] = setname
                rows.append(r)
    occl = [r for r in rows if r.get("type") == "a_occlusion"]
    # 면적비 우선, 동률이면 지속시간
    def key(r):
        try:
            return (-float(r.get("area_ratio_max") or 0), -float(r.get("dur_s") or 0))
        except ValueError:
            return (0.0, 0.0)
    occl.sort(key=key)
    top = occl[:20]
    outdir = os.path.join(SCAN, "candidates", "ranked")
    os.makedirs(outdir, exist_ok=True)
    outcsv = os.path.join(outdir, "ranked_occlusion_top20.csv")
    if top:
        cols = ["rank"] + [c for c in top[0].keys() if c != "_set"] + ["_set"]
        with open(outcsv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for i, r in enumerate(top, 1):
                r2 = dict(r)
                r2["rank"] = i
                w.writerow(r2)
    log(f"  a_occlusion 후보 총 {len(occl)}건 → 상위 {len(top)}건 저장")
    record("7_rank", "OK", time.time() - t,
           f"a_occlusion 총 {len(occl)}건, 상위 {len(top)}건 → {outcsv}",
           "면적비 내림차순, 동률 시 지속시간. 채택 여부는 판단하지 않음")


# ------------------------------------------------------------- 8) API 재시도
def task8_api():
    t = time.time()
    log("TASK8 시작 — data.go.kr frequentzoneLg 재시도 (1시간 간격 최대 5회)")
    keyfile = os.path.join(os.path.expanduser("~"), ".config", "datagokr.key")
    if not os.path.exists(keyfile):
        record("8_api_retry", "SKIP", time.time() - t, "인증키 파일 없음")
        return
    key = open(keyfile, encoding="ascii").read().strip()
    districts = [("111", "상당구"), ("112", "서원구"), ("113", "흥덕구"),
                 ("114", "청원구"), ("750", "진천군"), ("770", "음성군")]
    years = [2021, 2022, 2023, 2024]
    attempts = []
    ok = False
    for attempt in range(1, 6):
        url = ("http://apis.data.go.kr/B552061/frequentzoneLg/getRestFrequentzoneLg"
               f"?serviceKey={key}&searchYearCd=2023&siDo=43&guGun=111"
               "&type=json&numOfRows=5&pageNo=1")
        body, err = "", ""
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                body = r.read().decode("utf-8", "replace")
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            try:
                body = exc.read().decode("utf-8", "replace")  # type: ignore
            except Exception:
                pass
        good = ('"resultCode":"00"' in body) or ("NORMAL" in body and "<items>" in body)
        attempts.append({"attempt": attempt, "at": f"{datetime.now():%H:%M:%S}",
                         "error": err, "body_head": body[:300]})
        log(f"  시도 {attempt}/5 — {'성공' if good else '실패'} {err} {body[:120]}")
        if good:
            ok = True
            break
        if attempt < 5:
            log("  1시간 대기 후 재시도")
            time.sleep(3600)

    with open(os.path.join(NIGHT, "api_retry_log.json"), "w", encoding="utf-8") as f:
        json.dump(attempts, f, ensure_ascii=False, indent=2)

    if not ok:
        record("8_api_retry", "FAIL", time.time() - t,
               f"5회 모두 실패. 마지막 응답: {attempts[-1]['body_head'][:200]}",
               "오류 원문만 기록하고 종료 (지시대로 임의 대체 수집 안 함)")
        return

    # 성공 시 전 연도·전 구 수집
    outdir = os.path.join(ROOT, "cheongju_heatmap_lg")
    raw = os.path.join(outdir, "raw")
    os.makedirs(raw, exist_ok=True)
    allrows = []
    for y in years:
        for gg, nm in districts:
            u = ("http://apis.data.go.kr/B552061/frequentzoneLg/getRestFrequentzoneLg"
                 f"?serviceKey={key}&searchYearCd={y}&siDo=43&guGun={gg}"
                 "&type=json&numOfRows=100&pageNo=1")
            try:
                with urllib.request.urlopen(u, timeout=40) as r:
                    data = json.loads(r.read().decode("utf-8", "replace"))
            except Exception as exc:
                log(f"  {y} {nm} 실패: {exc}")
                continue
            with open(os.path.join(raw, f"lg_{y}_43{gg}.json"), "w",
                      encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            node = (data.get("items") or {})
            it = node.get("item") if isinstance(node, dict) else None
            items = [it] if isinstance(it, dict) else (it or [])
            for x in items:
                x["_year"] = y
                x["_district"] = nm
                allrows.append(x)
            time.sleep(0.25)
    if allrows:
        cols = sorted({k for r in allrows for k in r.keys()})
        with open(os.path.join(outdir, "cheongju_lg_hotspots.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(allrows)
    record("8_api_retry", "OK", time.time() - t,
           f"수집 {len(allrows)}건 → {outdir}", "재시도 성공 후 전 연도·전 구 수집")


def main():
    log("=" * 70)
    log("야간 무인 배치 시작")
    decode = "cv2"
    for fn, name in ((task4_validate_ffmpeg, "4"),):
        try:
            decode = fn() or "cv2"
        except Exception as exc:
            log(f"TASK{name} 예외: {exc}")
            record(f"{name}_exception", "ERROR", 0, str(exc))
    for fn, name in ((lambda: task5_today(decode), "5"),
                     (lambda: task6_full(decode), "6"),
                     (task7_rank, "7"),
                     (task8_api, "8")):
        try:
            fn()
        except Exception as exc:
            log(f"TASK{name} 예외: {exc}")
            record(f"{name}_exception", "ERROR", 0, str(exc))
    log("야간 배치 종료 — 보고서 생성")
    try:
        subprocess.run([sys.executable, os.path.join(NIGHT, "make_report.py")],
                       cwd=NIGHT, timeout=600)
    except Exception as exc:
        log(f"보고서 생성 실패: {exc}")


if __name__ == "__main__":
    main()
