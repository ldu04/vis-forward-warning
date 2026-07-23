# 야간 무인 배치 보고서

생성 시각: **2026-07-21 08:04:31**

이 문서 하나로 전체 상황을 파악할 수 있게 정리했습니다. 판단이 필요한 항목은 3장에 모아두었습니다.

## 1. 작업별 결과

| 작업 | 상태 | 소요 | 완료시각 | 내용 |
|---|---|---|---|---|
| 4_ffmpeg_validate | **불일치(회귀)** | 1.1분 | 2026-07-20 23:33:03 | ffmpeg 0건 {} / cv2 7건 {'d_person': 6, 'c_brake': 1}. ffmpeg 0s/file vs cv2 19.08s/file |
| 5_today_scan | **성공** | 28.4분 | 2026-07-21 00:01:29 | 125파일, 후보 621건, a_occlusion 13건 |
| 6_normal_full | **성공** | 37.2분 | 2026-07-21 00:38:39 | 후보 846건 {'d_person': 618, 'c_brake': 75, 'c_brake_near': 129, 'a_occlusion': 13, 'b_reveal': 11} |
| 7_rank | **성공** | 0초 | 2026-07-21 00:38:39 | a_occlusion 총 26건, 상위 20건 → C:\Users\이동욱\Desktop\자소서\엔지니어링 산업 공모전\carla-project\experiments\blackbox_scan\candidates\ran |

## 2. 자동으로 내린 결정

- **4_ffmpeg_validate** — 검출 결과 불일치 → 기존 decode=cv2 로 회귀
- **5_today_scan** — 정지 포함 위해 occl_area 0.10 / min_sec 1.5 / cy_min 0.30 으로 완화
- **6_normal_full** — 동일 필터(정지 포함) 적용, 파일 단위 체크포인트로 재실행 시 이어받음
- **7_rank** — 면적비 내림차순, 동률 시 지속시간. 채택 여부는 판단하지 않음

## 3. ★ 결정 대기 목록 (사람 판단 필요)

1. **시야차단 후보 상위 20건 선별** — 자동 정렬만 했고 채택 판단은 하지 않았습니다. `experiments/blackbox_scan/candidates/ranked/ranked_occlusion_top20.csv` 와 각 행의 `frame_png` 를 보고 제안서 삽입 여부를 정해 주세요.
2. **브레이크등 임계값 확정** — 임계 근처(5~8%) 9건이 있습니다. 8% 유지 / 조정 여부를 정해 주세요.
3. **급감속 상위 5건 육안 확인** — 자동 분류는 전부 '노면충격/판단애매'였습니다. `experiments/blackbox_scan/decel/` 의 PNG를 보고 실제 위험상황이 있는지 판단해 주세요. 이 결과에 따라 4-4를 실사로 갈지 CARLA로 갈지 갈립니다.

## 4. 시야차단 · 대형차 후보

자동 등급화 결과 상위 20건 (면적비 내림차순):

| 순위 | 파일 | 시작~종료 | 지속 | 면적비 | 종류 | 프레임 |
|---|---|---|---|---|---|---|
| 1 | 20260720-17h25m47s_N.avi | 6.5~8.0s | 1.5s | 0.3442 | truck | `a_20260720-17h25m47s_N_t7.5s_ar0.344.png` |
| 2 | 20260720-17h25m47s_N.avi | 6.5~8.0s | 1.5s | 0.3442 | truck | `a_20260720-17h25m47s_N_t7.5s_ar0.344.png` |
| 3 | 20260720-12h02m15s_N.avi | 26.5~28.5s | 2.0s | 0.3177 | truck | `a_20260720-12h02m15s_N_t28.5s_ar0.318.png` |
| 4 | 20260720-12h02m15s_N.avi | 26.5~28.5s | 2.0s | 0.3177 | truck | `a_20260720-12h02m15s_N_t28.5s_ar0.318.png` |
| 5 | 20260720-11h49m11s_N.avi | 17.0~22.0s | 5.0s | 0.3172 | bus | `a_20260720-11h49m11s_N_t21.5s_ar0.317.png` |
| 6 | 20260720-11h49m11s_N.avi | 17.0~22.0s | 5.0s | 0.3172 | bus | `a_20260720-11h49m11s_N_t21.5s_ar0.317.png` |
| 7 | 20260720-17h36m47s_N.avi | 3.5~5.0s | 1.5s | 0.3109 | truck | `a_20260720-17h36m47s_N_t4.0s_ar0.311.png` |
| 8 | 20260720-17h36m47s_N.avi | 3.5~5.0s | 1.5s | 0.3109 | truck | `a_20260720-17h36m47s_N_t4.0s_ar0.311.png` |
| 9 | 20260720-17h38m43s_N.avi | 18.0~20.0s | 2.0s | 0.3078 | bus | `a_20260720-17h38m43s_N_t19.0s_ar0.308.png` |
| 10 | 20260720-17h38m43s_N.avi | 18.0~20.0s | 2.0s | 0.3078 | bus | `a_20260720-17h38m43s_N_t19.0s_ar0.308.png` |

PNG 위치: `experiments/blackbox_scan/*/candidates/`

## 5. 브레이크등 임계 평가

- 총 34건 — **detected 8 / near 9 / miss 17** (임계 8%)
- 점등 판정군 붉은픽셀: 9.2~41.7%
- 미점등군 붉은픽셀: 0.0~6.9%
- 두 군이 뚜렷이 갈리고 8%가 그 사이 빈 구간에 위치 → 임계값 타당성 실증
- CSV: `experiments/blackbox_scan/brake/brake_light_evaluation.csv`
- PNG: `experiments/blackbox_scan/brake/png/`

## 6. API 재시도 결과

- 수행되지 않음

- 히트맵 생성 여부: 미생성 (수집 실패)

## 7. 산출물 위치

| 항목 | 경로 |
|---|---|
| 브레이크등 평가 | `experiments/blackbox_scan/brake/` |
| 급감속 상위 5 | `experiments/blackbox_scan/decel/` |
| 오늘(07-20) 표적 스캔 | `experiments/blackbox_scan/today_0720/` |
| NORMAL 전체 스캔 | `experiments/blackbox_scan/normal_full/` |
| 후보 등급화 | `experiments/blackbox_scan/candidates/ranked/` |
| 컨택트시트 | `C:\Users\이동욱\blackbox_sheets\NORMAL\` |
| 실행 로그 | `experiments/overnight/run.log` |

