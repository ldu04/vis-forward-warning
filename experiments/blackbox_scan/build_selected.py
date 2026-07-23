"""
제안서용 선별본 생성 (개인정보 처리 포함).

스캔이 뽑아낸 원본 해상도 프레임 PNG(수백 장, 수백 MB)에서 커밋할 것만 골라
개인식별정보를 제거한 뒤 긴 변 1280px로 축소해 candidates/selected/ 아래에 모으고,
근거 표를 담은 README.md 를 생성한다.

원칙
- 원본 PNG는 건드리지 않는다 (읽기만 한다). 처리본만 새로 쓴다.
- 브레이크등은 검출 성공만 넣지 않는다. 미검출(near) 사례를 같은 수만큼 넣어
  평가 자료로서 균형을 맞춘다. 성공 사례만 모으면 평가가 아니라 홍보가 된다.
- 선별 기준은 전부 CSV의 수치로 결정하고, 그 수치를 README 표에 그대로 남긴다.

개인정보 처리 (공개 저장소 커밋 전제)
1. OSD 크롭 — 블랙박스가 프레임 상단에 새기는 오버레이에 촬영 일시(초 단위)와
   속도가 들어간다. 도로표지판·상호와 조합하면 주행 경로와 시간대가 역추적되므로
   상단 띠를 여유 있게 잘라낸다.
2. 급감속 컨택트시트는 6분할이라 타일마다 OSD 가 따로 들어간다. 통짜로 자를 수
   없으므로 타일별 상단 띠를 마스킹하고, 헤더의 원본 파일명(일시 포함)도 가린다.
3. 보행자(d_person) 프레임은 아예 싣지 않는다. 얼굴이 식별되는 프레임을 공개
   저장소에 올릴 실익이 없고, 보행자 검출 근거는 CSV 수치로 충분하다.
4. 크롭 후에도 번호판·얼굴·연락처가 판독되는 프레임은 BLUR_REGIONS 에 좌표를
   등록해 가우시안 블러로 가린다.

사용법:  python build_selected.py
"""
from __future__ import annotations

import csv
import os
import shutil
from collections import OrderedDict

from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
SELECTED = os.path.join(HERE, "candidates", "selected")
LONG_EDGE = 1280

# 상단 OSD 띠를 잘라낼 비율. 실측 OSD 는 1080px 기준 약 21px(=2%)이고,
# 여유를 둬서 5% 를 잘라낸다. 하늘 영역이라 판정 근거는 손실되지 않는다.
OSD_CROP_FRAC = 0.05

# 컨택트시트(급감속) 기하 — 3열 2행, 상단 헤더 바
SHEET_COLS, SHEET_ROWS = 3, 2
SHEET_HEADER_FRAC = 0.062      # 헤더 바 높이 비율
SHEET_TILE_OSD_FRAC = 0.075    # 타일 상단 OSD 띠 비율 (타일 높이 기준)
# 1차 검수에서 타일 OSD 마스크가 1~2px 아래로 밀려 글자 최상단이 노출됐다.
# 원본 해상도 기준으로 위아래 여유를 준다.
SHEET_TILE_OSD_PAD = 6
SHEET_NAME_X0, SHEET_NAME_X1 = 0.015, 0.245   # 헤더 안에서 파일명이 있는 가로 구간

# 브레이크등 프레임은 eval_brake_lights.py 가 좌하단에 원본 파일명과 타임코드를
# 캡션으로 새긴다. 상단 OSD 를 잘라도 여기 일시가 남으므로 함께 가린다.
CAPTION_BOX = (0.0, 0.925, 0.34, 1.0)

# 크롭 후에도 남는 식별정보(번호판·얼굴·연락처·상호·지명)를 가릴 영역.
#   값 = [(x0, y0, x1, y1), ...]  — 크롭 완료된 이미지 기준 0~1 비율 좌표
# 키잉 규칙:
#   - 시야차단(a_occlusion) = 출력 PNG 파일명(면적비). 선별이 안 바뀌므로 안정적.
#   - 브레이크등(c_brake)   = 원본 프레임 idx("001"~"034"). 검출/미검출 교체로
#     출력 번호가 밀려도 좌표가 엉뚱한 파일로 가지 않게 원본 고유번호로 키잉한다.
# 상호·지명은 여러 개가 한 컷에 묶이면 지도검색으로 촬영지가 특정되므로 보수적으로 가린다.
BLUR_REGIONS: dict[str, list[tuple[float, float, float, float]]] = {
    # ============ 시야차단 (출력 파일명 키) ============
    # 좌상단 도로표지판(지명) + 좌상단 건물간판
    "a02_occlusion_ar0.318.png": [
        (0.700, 0.268, 0.792, 0.355),   # 도로표지판(안교로 지명)
        (0.000, 0.000, 0.158, 0.205),   # 좌상단 건물 간판
    ],
    # 중앙 선행 세단 후면 번호판 + 좌측 관광버스 노란 번호판 + 좌측 ◯속관광 상호
    "a03_occlusion_ar0.317.png": [
        (0.432, 0.500, 0.523, 0.568),
        (0.000, 0.465, 0.062, 0.532),
        (0.000, 0.303, 0.112, 0.418),   # 좌측 관광버스 상호(◯속관광)
    ],
    # 우측 소형 상가 간판
    "a04_occlusion_ar0.311.png": [
        (0.750, 0.350, 0.822, 0.452),
    ],
    # 크레인 트럭 전화번호/상호 + HD 주유소·도로표지판·레미콘 배너
    "a08_occlusion_ar0.229.png": [
        (0.000, 0.420, 0.120, 0.495),   # 적재함 전화번호
        (0.110, 0.060, 0.205, 0.135),   # 상단 배너 전화번호
        (0.560, 0.235, 0.628, 0.345),   # HD 주유소 로고(폴)
        (0.720, 0.455, 0.775, 0.515),   # HD (벽면)
        (0.440, 0.255, 0.535, 0.315),   # 초록 도로표지판(지명)
        (0.598, 0.365, 0.672, 0.420),   # 레미콘마켓 배너
        (0.014, 0.150, 0.078, 0.212),   # 크레인차 CCTV/상호
    ],
    # 은색 세단 번호판 + e마트·GOOD TIRE·VOLVO·대산관광·3+1 간판
    "a09_occlusion_ar0.174.png": [
        (0.575, 0.488, 0.640, 0.545),   # 은색 세단 번호판
        (0.706, 0.105, 0.834, 0.218),   # e-mart
        (0.450, 0.285, 0.540, 0.352),   # GOOD TIRE
        (0.008, 0.148, 0.098, 0.222),   # VOLVO
        (0.786, 0.365, 0.880, 0.454),   # 대산관광
        (0.478, 0.412, 0.542, 0.464),   # 3+1 / EFS
    ],
    # ============ 브레이크등 (원본 idx 키) ============
    "014": [   # c_detected 41.69% — KB손해보험(좌 건물)
        (0.128, 0.275, 0.308, 0.378),
    ],
    "012": [   # c_detected 30.24% — 번호판2 + KB(세로+우) + 도로표지판 + 소형인물2
        (0.210, 0.508, 0.285, 0.560),   # 흰 SUV 전면 번호판
        (0.882, 0.558, 0.952, 0.607),   # 적색 쿠페 후면 번호판(검출박스 안)
        (0.700, 0.128, 0.768, 0.375),   # KB손해보험(세로 간판, 중앙우측)
        (0.862, 0.118, 0.995, 0.255),   # KB손해보험(우 건물 상단)
        (0.415, 0.075, 0.505, 0.152),   # 초록 도로표지판(kumdae-gil)
        (0.344, 0.385, 0.386, 0.462),   # 횡단보도 부근 소형 인물 2명
    ],
    "010": [   # c_detected 27.16% — 부흥자동차공업사 상호밴드 + 도로표지판
        (0.355, 0.225, 0.720, 0.378),   # 공업사 상호 + 브랜드로고 밴드
        (0.120, 0.250, 0.288, 0.420),   # 좌측 건물 텍스트
        (0.875, 0.245, 0.968, 0.320),   # 우측 상단 도로표지판
    ],
    "004": [   # (교체입고) c_detected 16.95% — 충남고속 버스
        (0.395, 0.545, 0.455, 0.585),   # 버스 후면 번호판
        (0.283, 0.483, 0.347, 0.522),   # 전방 좌측 SUV 후면 번호판
        (0.358, 0.333, 0.480, 0.500),   # 충남고속/우등 버스 상호(텍스트 밴드)
        (0.393, 0.803, 0.474, 0.854),   # 와이퍼 카울 전화번호
    ],
    "013": [   # c_near 6.88% — 스타렉스 밴 2대 후면 번호판
        (0.686, 0.536, 0.750, 0.592),
        (0.772, 0.536, 0.840, 0.590),
    ],
    "008": [   # c_near 5.43% — 흰 SUV 번호판 + 요양병원 광고
        (0.448, 0.496, 0.510, 0.542),   # 흰 SUV 후면 번호판
        (0.478, 0.400, 0.582, 0.462),   # 요양병원 광고판
    ],
    "032": [   # c_near 5.35% — GS25 + 위층 이마트/학원 배너(전화)
        (0.700, 0.290, 0.782, 0.360),   # GS25
        (0.560, 0.222, 0.758, 0.308),   # 이마트24/학원 배너 + 전화
    ],
    "016": [   # (교체입고) c_near 5.21% — 교외 교차로, 앞차 SUV 번호판만
        (0.740, 0.482, 0.836, 0.553),   # 검정 SUV 후면 번호판(41◯1816)
    ],
}

# 교체 규칙 — 원본 idx 기준. 도심 간판 밀집 + 인물(초상권) 프레임을 배경 단순 프레임으로 뺀다.
BRAKE_DET_EXCLUDE = {"001", "002"}   # 안경점·약국 밀집 + 횡단보도 보행자 → 004 로
BRAKE_NEAR_EXCLUDE = {"020"}         # 다이소 등 간판 밀집 + 안전조끼 작업자 → 016 으로

# 원본 영상 보관 위치 (저장소에는 포함하지 않는다)
SRC_VIDEO_DIRS = [r"C:\Users\이동욱\blackbox_full\NORMAL",
                  r"C:\Users\이동욱\blackbox_full\EVENT"]


def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fnum(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def find_png(name: str) -> str | None:
    if not name:
        return None
    for d in (os.path.join(HERE, "normal_full", "candidates"),
              os.path.join(HERE, "today_0720", "candidates"),
              os.path.join(HERE, "candidates"),
              os.path.join(HERE, "brake", "png"),
              os.path.join(HERE, "decel"),
              os.path.join(HERE, "pilot", "candidates")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def blur_box(im: Image.Image, box: tuple[int, int, int, int]) -> None:
    """지정 영역을 되돌릴 수 없게 뭉갠다 (축소 후 확대 + 가우시안)."""
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(im.width, x1), min(im.height, y1)
    if x1 <= x0 or y1 <= y0:
        return
    region = im.crop((x0, y0, x1, y1))
    w, h = region.size
    # 먼저 극단적으로 축소했다가 되키워 정보 자체를 파괴한 뒤 블러로 경계를 없앤다
    small = region.resize((max(1, w // 22), max(1, h // 22)), Image.BILINEAR)
    region = small.resize((w, h), Image.NEAREST).filter(ImageFilter.GaussianBlur(6))
    im.paste(region, (x0, y0))


def strip_osd(im: Image.Image) -> Image.Image:
    """일반 프레임: 상단 OSD 띠를 잘라낸다."""
    cut = int(round(im.height * OSD_CROP_FRAC))
    return im.crop((0, cut, im.width, im.height))


def mask_sheet(im: Image.Image) -> Image.Image:
    """컨택트시트: 헤더의 파일명과 타일별 OSD 띠를 가린다."""
    W, H = im.size
    header_h = int(round(H * SHEET_HEADER_FRAC))
    # 헤더 안 파일명 구간
    blur_box(im, (int(W * SHEET_NAME_X0), 0, int(W * SHEET_NAME_X1), header_h))
    # 타일별 상단 OSD
    grid_h = H - header_h
    tile_h = grid_h / SHEET_ROWS
    tile_w = W / SHEET_COLS
    band = int(round(tile_h * SHEET_TILE_OSD_FRAC))
    pad = SHEET_TILE_OSD_PAD
    for r in range(SHEET_ROWS):
        for c in range(SHEET_COLS):
            x0 = int(round(c * tile_w))
            y0 = max(0, int(round(header_h + r * tile_h)) - pad)
            blur_box(im, (x0, y0, int(round(x0 + tile_w)), y0 + band + 2 * pad))
    return im


def emit(src: str, dst_dir: str, dst_name: str, sheet: bool = False,
         caption: bool = False, blur_key: str | None = None):
    """개인정보 처리 후 긴 변 LONG_EDGE 로 축소 저장.

    blur_key: BLUR_REGIONS 조회 키. 브레이크등은 원본 idx 를 넘겨 선별 순서가
    바뀌어도 좌표가 안정적으로 따라가게 한다. None 이면 출력 파일명으로 조회.
    """
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, dst_name)
    with Image.open(src) as im0:
        ow, oh = im0.size
        im = im0.convert("RGB")
        im = mask_sheet(im) if sheet else strip_osd(im)

        if caption:
            cx0, cy0, cx1, cy1 = CAPTION_BOX
            blur_box(im, (int(cx0 * im.width), int(cy0 * im.height),
                          int(cx1 * im.width), int(cy1 * im.height)))

        # 등록된 블러 영역 적용 (크롭 완료 기준 비율 좌표)
        for (bx0, by0, bx1, by1) in BLUR_REGIONS.get(blur_key or dst_name, []):
            blur_box(im, (int(bx0 * im.width), int(by0 * im.height),
                          int(bx1 * im.width), int(by1 * im.height)))

        scale = min(1.0, LONG_EDGE / max(im.size))
        if scale < 1.0:
            im = im.resize((round(im.width * scale), round(im.height * scale)),
                           Image.LANCZOS)
        im.save(dst, "PNG", optimize=True)
    return ow, oh, os.path.getsize(src), os.path.getsize(dst)


rows_out: dict[str, list[dict]] = OrderedDict()


# --------------------------------------------------------- a) 시야차단 상위 10
def pick_occlusion() -> None:
    rows = read_csv(os.path.join(HERE, "candidates", "ranked",
                                 "ranked_occlusion_top20.csv"))
    # ranked CSV 는 normal_full 과 today_0720 을 합쳐 만들었는데 두 스캔이 07-20
    # 파일을 공통으로 훑어서 같은 후보가 두 번 들어가 있다. frame_png 로 중복 제거.
    uniq: dict[str, dict] = OrderedDict()
    for r in rows:
        k = r.get("frame_png") or ""
        if k and k not in uniq:
            uniq[k] = r
    picked = list(uniq.values())[:10]

    out = []
    for i, r in enumerate(picked, 1):
        src = find_png(r["frame_png"])
        if not src:
            print(f"  [경고] 원본 PNG 없음: {r['frame_png']}")
            continue
        name = f"a{i:02d}_occlusion_ar{fnum(r['area_ratio_max']):.3f}.png"
        ow, oh, ob, nb = emit(src, os.path.join(SELECTED, "a_occlusion"), name)
        out.append({
            "png": name, "src_video": r["file"],
            "timecode": f"{r['t_start']}~{r['t_end']}s (지속 {r['dur_s']}s)",
            "metric": (f"면적비 최대 {fnum(r['area_ratio_max']):.4f} / "
                       f"평균 {fnum(r['area_ratio_mean']):.4f}"),
            "verdict": f"{r.get('cls','')} · conf {r.get('conf','')} · 임계 0.10 통과",
            "size": f"{ow}x{oh} → {ob/1e6:.1f}MB / {nb/1e6:.2f}MB",
        })
    rows_out["a_occlusion"] = out
    print(f"  시야차단 {len(out)}장 (중복 제거 전 {len(rows)}행 → 고유 {len(uniq)}건)")


# ------------------------------------------- c) 브레이크등 검출 5 + 미검출 5
def _idx(r) -> str:
    return (r.get("frame_png") or "")[:3]


def pick_brake() -> None:
    rows = read_csv(os.path.join(HERE, "brake", "brake_light_evaluation.csv"))
    # 도심 간판 밀집 + 인물(초상권) 프레임은 배경 단순 프레임으로 교체한다.
    # 001/002(안경점·약국·횡단보도 보행자) 제외 → 004(충남고속 버스)가 자동 승격.
    det = [r for r in sorted([x for x in rows if x["verdict"] == "detected"],
                             key=lambda r: -fnum(r["red_pct_rescan"]))
           if _idx(r) not in BRAKE_DET_EXCLUDE][:4]
    # 020(다이소 등 간판 밀집 + 안전조끼 작업자) 제외 → 016(교외 교차로)이 승격.
    near = [r for r in sorted([x for x in rows if x["verdict"] == "near"],
                              key=lambda r: -fnum(r["red_pct_rescan"]))
            if _idx(r) not in BRAKE_NEAR_EXCLUDE][:5]

    out = []
    for tag, group in (("detected", det), ("near", near)):
        for i, r in enumerate(group, 1):
            src = find_png(r["frame_png"])
            if not src:
                print(f"  [경고] 원본 PNG 없음: {r['frame_png']}")
                continue
            name = f"c_{tag}_{i:02d}_red{fnum(r['red_pct_rescan']):05.2f}pct.png"
            ow, oh, ob, nb = emit(src, os.path.join(SELECTED, "c_brake"), name,
                                  caption=True, blur_key=_idx(r))
            passed = fnum(r["red_ratio_rescan"]) >= fnum(r["threshold"])
            out.append({
                "png": name, "src_video": r["file"],
                "timecode": f"{r['t_start']}~{r['t_end']}s (피크 {r['t_peak_s']}s)",
                "metric": f"붉은픽셀 {fnum(r['red_pct_rescan']):.2f}%",
                "verdict": (f"**{tag}** · 임계 {fnum(r['threshold'])*100:.0f}% "
                            f"{'통과' if passed else '미달'} · {r.get('vehicle_cls','')}"),
                "size": f"{ow}x{oh} → {ob/1e6:.1f}MB / {nb/1e6:.2f}MB",
            })
    rows_out["c_brake"] = out
    print(f"  브레이크등 {len(out)}장 (detected {len(det)} + near {len(near)})")


# ------------------------------------------------------------ 급감속 상위 5
def pick_decel() -> None:
    rows = read_csv(os.path.join(HERE, "decel", "decel_top.csv"))[:5]
    out = []
    for i, r in enumerate(rows, 1):
        src = find_png(r["png"])
        if not src:
            print(f"  [경고] 원본 PNG 없음: {r['png']}")
            continue
        name = f"decel_{i:02d}_drop{fnum(r['drop_pct']):04.1f}pct.png"
        ow, oh, ob, nb = emit(src, os.path.join(SELECTED, "decel"), name,
                              sheet=True)
        out.append({
            "png": name, "src_video": r["file"],
            "timecode": "(6분할 컨택트시트 — 0/4/8/12/16/20초)",
            "metric": f"속도 강하 {fnum(r['drop_pct']):.1f}%",
            "verdict": f"자동분류 **{r['kind']}** — {r['why']}",
            "size": f"{ow}x{oh} → {ob/1e6:.1f}MB / {nb/1e6:.2f}MB",
        })
    rows_out["decel"] = out
    print(f"  급감속 {len(out)}장")


TITLES = {
    "a_occlusion": ("시야차단 (대형차)",
                    "면적비 내림차순 상위 10건. 앞차가 화면에서 차지하는 면적비가 "
                    "클수록 후방 차량의 시야가 막힌다."),
    "c_brake": ("브레이크등 검출",
                "검출 성공 4건 + **미검출(near) 5건**. 미검출을 반드시 함께 싣는다 — "
                "성공 사례만 모으면 평가가 아니라 홍보가 된다. "
                "(도심 간판 밀집·인물 노출 프레임 2종은 배경 단순 프레임으로 교체함)"),
    "decel": ("급감속",
              "속도 강하율 상위 5건. 자동분류 결과는 육안 확인 전이라 확정이 아니다."),
}


def write_readme() -> None:
    total_n = sum(len(v) for v in rows_out.values())
    total_b = 0
    for root, _, files in os.walk(SELECTED):
        for f in files:
            if f.lower().endswith(".png"):
                total_b += os.path.getsize(os.path.join(root, f))

    L = []
    L.append("# 제안서용 선별 프레임\n")
    L.append("스캔 파이프라인이 뽑아낸 후보 프레임 중 제안서에 실을 것만 골라 "
             "개인식별정보를 제거한 뒤 모았습니다.\n")
    L.append("")
    L.append("## ⚠ 원본 영상은 이 저장소에 포함하지 않습니다\n")
    L.append("아래 PNG들은 모두 **개인 소유 차량의 블랙박스 주행기록**에서 잘라낸 "
             "프레임입니다.")
    L.append("원본 영상 파일(`.avi`)은 개인 주행 이력·위치 정보에 해당하므로 저장소에 "
             "커밋하지 않고")
    L.append("로컬에만 보관합니다. 재현이 필요하면 아래 경로의 원본과 `scan_v3.py` 로 "
             "다시 생성할 수 있습니다.\n")
    for d in SRC_VIDEO_DIRS:
        L.append(f"- `{d}`")
    L.append("")
    L.append("같은 이유로 `.gitignore` 가 `**/candidates/**` 를 기본 제외하고, 이 "
             "`selected/` 폴더만")
    L.append("예외로 열어 둡니다. 선별하지 않은 원본 해상도 PNG가 실수로 커밋되는 것을 "
             "구조적으로 막기 위함입니다.\n")
    L.append("")
    L.append("## 개인정보 처리 내역\n")
    L.append(f"1. **상단 OSD 크롭** — 블랙박스가 프레임 상단에 새기는 촬영 일시(초 단위)와 "
             f"속도 오버레이를 잘라냈습니다 (상단 {OSD_CROP_FRAC*100:.0f}%). "
             f"도로표지판·상호와 조합하면 주행 경로와 시간대가 역추적되기 때문입니다.")
    L.append("2. **컨택트시트 마스킹** — 급감속 6분할 시트는 타일마다 OSD 가 따로 들어가 "
             "통짜로 자를 수 없어, 타일별 상단 띠와 헤더의 원본 파일명을 마스킹했습니다.")
    L.append("3. **보행자 프레임 제외** — 얼굴이 식별되는 프레임은 싣지 않습니다. "
             "보행자 검출 근거는 `EVENT_candidates.csv` 의 검출 신뢰도 수치로 대신합니다.")
    if BLUR_REGIONS:
        L.append(f"4. **번호판·연락처·상호·지명·인물 블러** — 크롭 후에도 판독되는 "
                 f"식별정보(차량번호판, 휴대전화번호, 사업장 상호, 도로표지판 지명, "
                 f"소형 인물)를 {len(BLUR_REGIONS)}개 프레임에 모자이크 처리했습니다. "
                 f"상호·지명은 여러 개가 한 컷에 묶이면 촬영지가 특정되므로 보수적으로 "
                 f"가렸습니다.")
    L.append("5. **간판 밀집·인물 프레임 교체** — 도심 광고 인물(초상권)·간판이 화면을 "
             "덮는 브레이크등 프레임 2종은 블러 대신 배경이 단순한 다른 검출 프레임으로 "
             "교체했습니다.")
    L.append("")
    L.append("> 처리는 `build_selected.py` 가 수행하며, 원본 PNG 는 수정하지 않고 "
             "처리본만 새로 씁니다.\n")
    L.append("")
    L.append("## 이미지 처리\n")
    L.append(f"- 커밋본은 OSD 제거 후 긴 변 **{LONG_EDGE}px** 로 축소 + PNG 최적화한 "
             f"것입니다.")
    L.append("- **원본 해상도 PNG는 로컬에 그대로 보존**되어 있습니다 "
             "(`normal_full/candidates/`, `brake/png/`, `decel/` 등).")
    L.append("- 아래 표의 `크기` 열은 `원본해상도 → 원본용량 / 커밋본용량` 입니다.\n")
    L.append("")
    L.append("> **원본 영상 파일명과 타임코드는 이 공개 문서에 싣지 않습니다.** 파일명·"
             "캡션·픽셀에서 촬영 일시를 제거한 것과 같은 이유입니다 — 일시가 도로표지판·"
             "상호와 조합되면 주행 시각·경로가 역추적됩니다. 각 PNG ↔ 원본 대응관계는 "
             "저장소에 커밋되지 않는 로컬 매핑 파일에만 둡니다.\n")
    L.append("")
    L.append(f"**총 {total_n}장 / {total_b/1e6:.1f}MB**\n")

    for key, (title, note) in TITLES.items():
        rows = rows_out.get(key) or []
        if not rows:
            continue
        L.append("")
        L.append(f"## {title} — {len(rows)}장\n")
        L.append(f"{note}\n")
        L.append("| PNG | 판정 수치 | 결과 | 크기 |")
        L.append("|---|---|---|---|")
        for r in rows:
            L.append(f"| `{r['png']}` | {r['metric']} | {r['verdict']} | {r['size']} |")
        L.append("")

    L.append("")
    L.append("## 재현\n")
    L.append("```bash")
    L.append("# 후보 스캔 (원본 영상 필요)")
    L.append("python scan_v3.py --src <NORMAL 원본 폴더> --out normal_full "
             "--label NORMAL_FULL \\")
    L.append("    --occl-area 0.10 --occl-min-sec 1.5 --cx-min 0.05 --cx-max 0.95 "
             "--cy-min 0.30")
    L.append("")
    L.append("# 선별 + 개인정보 처리 (이 폴더)")
    L.append("python build_selected.py")
    L.append("```")
    L.append("")

    os.makedirs(SELECTED, exist_ok=True)
    with open(os.path.join(SELECTED, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n[저장] {os.path.join(SELECTED, 'README.md')}")
    print(f"선별본 합계: {total_n}장 / {total_b/1e6:.1f}MB")


def write_mapping() -> None:
    """PNG ↔ 원본 영상·타임코드 대응표. 저장소에 커밋하지 않는 로컬 전용 파일.

    candidates/ 아래(그러나 selected/ 밖)에 두므로 .gitignore 의 `**/candidates/**`
    규칙에 걸려 자동으로 추적에서 제외된다. 재현·검증 시 로컬에서만 참조한다.
    """
    L = ["# [로컬 전용] 선별 PNG ↔ 원본 영상 매핑",
         "",
         "이 파일은 **저장소에 커밋되지 않습니다** (`.gitignore` 의 `**/candidates/**`).",
         "공개 문서(`selected/README.md`)에서 제거한 원본 파일명·타임코드를 로컬 재현용으로만 보관합니다.",
         "원본 영상 자체는 아래 폴더에 있습니다:"]
    for d in SRC_VIDEO_DIRS:
        L.append(f"- `{d}`")
    L.append("")
    for key, (title, _note) in TITLES.items():
        rows = rows_out.get(key) or []
        if not rows:
            continue
        L.append(f"## {title}")
        L.append("")
        L.append("| PNG | 원본 영상 | 타임코드 |")
        L.append("|---|---|---|")
        for r in rows:
            L.append(f"| `{r['png']}` | `{r['src_video']}` | {r['timecode']} |")
        L.append("")
    outdir = os.path.join(HERE, "candidates")     # selected/ 밖 → gitignore 대상
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "SOURCE_MAP.local.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[저장] {path}  (로컬 전용, 커밋 안 됨)")


def main() -> None:
    if os.path.isdir(SELECTED):
        shutil.rmtree(SELECTED)          # 처리본은 항상 재생성 (원본은 건드리지 않음)
    print("선별 시작 (보행자 프레임은 개인정보 사유로 제외)")
    pick_occlusion()
    pick_brake()
    pick_decel()
    write_readme()
    write_mapping()


if __name__ == "__main__":
    main()
