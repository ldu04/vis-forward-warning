"""
청주 관내 화물차 교통사고 다발지역 수집 — 한국도로교통공단 OpenAPI.

출처: 한국도로교통공단, 화물차 교통사고 다발지역 정보, 공공데이터포털
엔드포인트: https://opendata.koroad.or.kr/data/rest/frequentzone/truck
선정기준: 최근 3년 중상 중심 화물차 사고, 반경 100m 내 4건 이상
         (원 데이터 명칭은 '사망·중상'이나 수집된 10개 지점은 전부 사망 0명)

★ 인증키는 코드에 하드코딩하지 않는다.
   환경변수 DATA_GO_KR_KEY 또는 저장소 밖 파일에서 읽는다.
   ※ 반드시 "인코딩된(Encoding) 키"를 그대로 쓸 것. URL 디코딩하면
     SERVICE_KEY_IS_NOT_REGISTERED_ERROR 가 난다 (2026-07-20 확인).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional

ENDPOINT = "https://opendata.koroad.or.kr/data/rest/frequentzone/truck"
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

# 제안서 5-1 실증 후보지 대조용 기준점 (근사 좌표)
# 오송생명과학단지는 제외 — 최근접 다발지점이 8km 이상이라 실증 근거가 없어
# 실증지는 오창 단독으로 간다.
REFERENCE_POINTS = [
    ("오창IC",            36.7220, 127.4300),
    ("오창과학산업단지",   36.7100, 127.4400),
    ("청주국제공항",      36.7166, 127.4990),
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


def fetch(key: str, year: int, gugun: str, rows: int = 100, page: int = 1) -> Dict[str, Any]:
    url = (
        f"{ENDPOINT}?authKey={key}&searchYearCd={year}&siDo={SIDO}&guGun={gugun}"
        f"&numOfRows={rows}&pageNo={page}&type=json"
    )
    with urllib.request.urlopen(url, timeout=40) as r:
        return json.loads(r.read().decode("utf-8"))


def polygon_centroid(geom_json: str) -> Optional[tuple]:
    """la_crd/lo_crd 가 없을 때 폴리곤 중심으로 대체."""
    try:
        g = json.loads(geom_json)
        coords = g["coordinates"][0]
        lon = sum(c[0] for c in coords) / len(coords)
        lat = sum(c[1] for c in coords) / len(coords)
        return lat, lon
    except Exception:
        return None


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import radians, sin, cos, asin, sqrt
    R = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".", help="출력 폴더")
    ap.add_argument("--key-file", default=None)
    args = ap.parse_args()

    key = load_key(args.key_file)
    raw_dir = os.path.join(args.out, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    queried_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    all_rows: List[Dict[str, Any]] = []
    log: List[Dict[str, Any]] = []

    for year in YEARS:
        for gugun, name in DISTRICTS:
            try:
                data = fetch(key, year, gugun)
            except Exception as exc:
                print(f"  {year} {name}: 요청 실패 {exc}", flush=True)
                log.append({"year": year, "gugun": gugun, "name": name,
                            "result": "REQUEST_FAILED", "n": 0})
                continue

            code = str(data.get("resultCode", ""))
            msg = data.get("resultMsg", "")
            with open(os.path.join(raw_dir, f"truck_{year}_{SIDO}{gugun}.json"),
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
            n = len(items)
            print(f"  {year} {name:<14} {code} {msg:<28} {n}건", flush=True)
            log.append({"year": year, "gugun": gugun, "name": name,
                        "result": f"{code}:{msg}", "n": n})

            for x in items:
                lat = x.get("la_crd")
                lon = x.get("lo_crd")
                if lat is None or lon is None:
                    c = polygon_centroid(x.get("geom_json", ""))
                    if c:
                        lat, lon = c
                row = {
                    "year": year,
                    "sido_cd": SIDO,
                    "gugun_cd": gugun,
                    "district": name,
                    "afos_fid": x.get("afos_fid"),
                    "afos_id": x.get("afos_id"),
                    "bjd_cd": x.get("bjd_cd"),
                    "spot_cd": x.get("spot_cd"),
                    "sido_sgg_nm": x.get("sido_sgg_nm"),
                    "spot_nm": x.get("spot_nm"),
                    "occrrnc_cnt": x.get("occrrnc_cnt"),
                    "caslt_cnt": x.get("caslt_cnt"),
                    "dth_dnv_cnt": x.get("dth_dnv_cnt"),
                    "se_dnv_cnt": x.get("se_dnv_cnt"),
                    "sl_dnv_cnt": x.get("sl_dnv_cnt"),
                    "wnd_dnv_cnt": x.get("wnd_dnv_cnt"),
                    "lat": lat,
                    "lon": lon,
                }
                # 기준점과의 거리
                if lat is not None and lon is not None:
                    for rname, rlat, rlon in REFERENCE_POINTS:
                        row[f"dist_{rname}_m"] = round(
                            haversine_m(float(lat), float(lon), rlat, rlon))
                all_rows.append(row)
            time.sleep(0.25)  # 서버 부하 배려

    # CSV 저장
    csv_path = os.path.join(args.out, "cheongju_truck_hotspots.csv")
    if all_rows:
        cols = list(all_rows[0].keys())
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
        print("\n수집된 지점이 없습니다.")

    meta = {
        "source": "한국도로교통공단, 화물차 교통사고 다발지역 정보, 공공데이터포털",
        "endpoint": ENDPOINT,
        "queried_at": queried_at,
        "sido": SIDO,
        "districts": [{"code": c, "name": n} for c, n in DISTRICTS],
        "years": YEARS,
        "selection_criteria": "최근 3년 중상 중심 화물차 교통사고, 반경 100m 내 4건 이상 "
                              "(원 데이터 명칭 '사망·중상' — 수집 10개 지점 전부 사망 0명)",
        "total_spots": len(all_rows),
        "request_log": log,
    }
    with open(os.path.join(args.out, "collection_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[저장] {os.path.join(args.out, 'collection_meta.json')}")


if __name__ == "__main__":
    main()
