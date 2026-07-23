"""
청주 관내 '지자체별 교통사고 다발지역' 수집 — 공공데이터포털(data.go.kr) OpenAPI.

출처: 도로교통공단, 지자체별 교통사고 다발지역, 공공데이터포털
엔드포인트: http://apis.data.go.kr/B552061/frequentzoneLg/getRestFrequentzoneLg

────────────────────────────────────────────────────────────────────────────
★ 현재 이 API 는 401 Unauthorized 로 막혀 있다 (2026-07-21 확인, 5회 시도 전부 실패).
  응답 본문이 XML 오류코드조차 없는 `Unauthorized` 단일 문자열이라 게이트웨이
  단계에서 거부된 것이고, 이는 '해당 API 활용신청이 승인되지 않은 상태'의 전형이다.
  키 문자열 자체는 정상임을 확인했다 — 같은 키로 opendata.koroad.or.kr 의
  화물차 다발지역 API 는 정상 동작한다(experiments/cheongju_heatmap 참조).

  → 따라서 이 스크립트는 **키만 유효해지면 그대로 돌아가도록** 완성해 둔 것이다.
     data.go.kr 마이페이지 > 활용신청 현황에서 승인이 확인되면 인자 없이 실행하면 된다.
────────────────────────────────────────────────────────────────────────────

★ 인증키는 코드에 하드코딩하지 않는다.
   환경변수 DATA_GO_KR_KEY 또는 저장소 밖 파일(~/.config/datagokr.key)에서 읽는다.
   ※ 반드시 "인코딩된(Encoding) 키"를 그대로 쓸 것. URL 디코딩해서 넣으면
     서명이 깨져 SERVICE_KEY_IS_NOT_REGISTERED_ERROR 가 난다.

사용법:
    python collect_lg_hotspots.py                 # 실수집
    python collect_lg_hotspots.py --probe         # 1회만 호출해 인증 상태만 확인
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional

ENDPOINT = "http://apis.data.go.kr/B552061/frequentzoneLg/getRestFrequentzoneLg"
KEY_FILE_DEFAULT = os.path.join(os.path.expanduser("~"), ".config", "datagokr.key")

SIDO = "43"  # 충청북도
DISTRICTS = [
    ("111", "청주시 상당구"),
    ("112", "청주시 서원구"),
    ("113", "청주시 흥덕구"),
    ("114", "청주시 청원구"),
    ("750", "진천군"),
    ("770", "음성군"),
]
YEARS = [2021, 2022, 2023, 2024]

# 제안서 5-1 실증 후보지 대조용 기준점 (근사 좌표) — truck 수집과 동일 기준
# 오송생명과학단지 제외 (실증지는 오창 단독). collect_truck_hotspots.py 와 일치시킨다.
REFERENCE_POINTS = [
    ("오창IC",           36.7220, 127.4300),
    ("오창과학산업단지",  36.7100, 127.4400),
    ("청주국제공항",     36.7166, 127.4990),
]


def load_key(path: Optional[str]) -> str:
    k = os.environ.get("DATA_GO_KR_KEY", "").strip()
    if k:
        return k
    p = path or KEY_FILE_DEFAULT
    if os.path.exists(p):
        with open(p, "r", encoding="ascii") as f:
            return f.read().strip()
    raise SystemExit(
        "인증키를 찾을 수 없습니다. 환경변수 DATA_GO_KR_KEY 를 설정하거나 "
        f"{p} 에 인코딩된 키를 저장하세요."
    )


def fetch(key: str, year: int, gugun: str, rows: int = 100, page: int = 1):
    """(data, error_string) 반환. 401 등 HTTP 오류도 예외 대신 값으로 돌려준다."""
    url = (f"{ENDPOINT}?serviceKey={key}&searchYearCd={year}&siDo={SIDO}"
           f"&guGun={gugun}&type=json&numOfRows={rows}&pageNo={page}")
    try:
        with urllib.request.urlopen(url, timeout=40) as r:
            body = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:200]
        except Exception:
            pass
        return None, f"HTTP {e.code}: {detail or e.reason}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    try:
        return json.loads(body), ""
    except json.JSONDecodeError:
        # 인증 실패 시 JSON 이 아니라 평문/XML 이 온다
        return None, f"NON_JSON: {body[:200]}"


def polygon_centroid(geom_json: str):
    try:
        g = json.loads(geom_json)
        coords = g["coordinates"][0]
        return (sum(c[1] for c in coords) / len(coords),
                sum(c[0] for c in coords) / len(coords))
    except Exception:
        return None


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import radians, sin, cos, asin, sqrt
    R = 6371000.0
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def probe(key: str) -> int:
    """인증 상태만 1회 확인. 승인되면 0, 아니면 1 을 반환한다."""
    data, err = fetch(key, 2023, "111", rows=5)
    if err:
        print(f"[실패] {err}")
        print("\n→ data.go.kr 마이페이지 > 활용신청 현황에서 해당 API 승인 여부를 확인하세요.")
        print("   승인 상태인데도 401 이면 키 재발급이 필요합니다.")
        return 1
    code = str((data or {}).get("resultCode", ""))
    print(f"[성공] resultCode={code} {(data or {}).get('resultMsg','')}")
    print("\n→ 인증이 열렸습니다. 인자 없이 다시 실행하면 전 연도·전 구를 수집합니다:")
    print("   python collect_lg_hotspots.py")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--probe", action="store_true", help="1회만 호출해 인증 상태 확인")
    args = ap.parse_args()

    key = load_key(args.key_file)
    if args.probe:
        raise SystemExit(probe(key))

    raw_dir = os.path.join(args.out, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    queried_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    all_rows: List[Dict[str, Any]] = []
    log: List[Dict[str, Any]] = []
    auth_failed = 0

    for year in YEARS:
        for gugun, name in DISTRICTS:
            data, err = fetch(key, year, gugun)
            if err:
                print(f"  {year} {name:<14} 실패 {err}", flush=True)
                log.append({"year": year, "gugun": gugun, "name": name,
                            "result": f"ERROR:{err}", "n": 0})
                if "401" in err or "Unauthorized" in err:
                    auth_failed += 1
                    # 인증 문제면 24회를 다 돌 이유가 없다. 바로 중단한다.
                    if auth_failed >= 2:
                        print("\n★ 인증 오류가 반복됩니다. 수집을 중단합니다.")
                        print("  data.go.kr 마이페이지 > 활용신청 현황에서 승인 여부를 확인하세요.")
                        break
                continue

            code = str(data.get("resultCode", ""))
            msg = data.get("resultMsg", "")
            with open(os.path.join(raw_dir, f"lg_{year}_{SIDO}{gugun}.json"),
                      "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            items = []
            if code == "00":
                node = data.get("items") or {}
                it = node.get("item") if isinstance(node, dict) else None
                if isinstance(it, dict):
                    items = [it]
                elif isinstance(it, list):
                    items = it
            print(f"  {year} {name:<14} {code} {msg:<28} {len(items)}건", flush=True)
            log.append({"year": year, "gugun": gugun, "name": name,
                        "result": f"{code}:{msg}", "n": len(items)})

            for x in items:
                lat, lon = x.get("la_crd"), x.get("lo_crd")
                if lat is None or lon is None:
                    c = polygon_centroid(x.get("geom_json", ""))
                    if c:
                        lat, lon = c
                row = {
                    "year": year, "sido_cd": SIDO, "gugun_cd": gugun, "district": name,
                    "afos_fid": x.get("afos_fid"), "afos_id": x.get("afos_id"),
                    "bjd_cd": x.get("bjd_cd"), "spot_cd": x.get("spot_cd"),
                    "sido_sgg_nm": x.get("sido_sgg_nm"), "spot_nm": x.get("spot_nm"),
                    "occrrnc_cnt": x.get("occrrnc_cnt"), "caslt_cnt": x.get("caslt_cnt"),
                    "dth_dnv_cnt": x.get("dth_dnv_cnt"), "se_dnv_cnt": x.get("se_dnv_cnt"),
                    "sl_dnv_cnt": x.get("sl_dnv_cnt"), "wnd_dnv_cnt": x.get("wnd_dnv_cnt"),
                    "lat": lat, "lon": lon,
                }
                if lat is not None and lon is not None:
                    for rname, rlat, rlon in REFERENCE_POINTS:
                        row[f"dist_{rname}_m"] = round(
                            haversine_m(float(lat), float(lon), rlat, rlon))
                all_rows.append(row)
            time.sleep(0.25)
        else:
            continue
        break  # 안쪽 for 가 break 로 끝난 경우 바깥도 중단

    csv_path = os.path.join(args.out, "cheongju_lg_hotspots.csv")
    if all_rows:
        cols: List[str] = list(all_rows[0].keys())
        for r in all_rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n[저장] {csv_path}  ({len(all_rows)}행)")
    else:
        print("\n수집된 지점이 없습니다. (인증 미승인 상태면 정상적인 결과입니다)")

    meta = {
        "source": "도로교통공단, 지자체별 교통사고 다발지역, 공공데이터포털",
        "endpoint": ENDPOINT,
        "queried_at": queried_at,
        "sido": SIDO,
        "districts": [{"code": c, "name": n} for c, n in DISTRICTS],
        "years": YEARS,
        "total_spots": len(all_rows),
        "auth_failed_requests": auth_failed,
        "request_log": log,
    }
    with open(os.path.join(args.out, "collection_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[저장] {os.path.join(args.out, 'collection_meta.json')}")


if __name__ == "__main__":
    main()
