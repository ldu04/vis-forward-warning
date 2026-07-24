# 선별 프레임 (대표 5장)

블랙박스 실사 스캔에서 개인정보 처리를 마친 선별본 중 **제안서 덱과 1:1 대응하는 핵심
5장**만 공개 수록한 것이다. 전체 스캔 규모·개인정보 처리 방식·전수 검증(OCR 0건)은
상위 폴더의 [`../../METHODOLOGY.md`](../../METHODOLOGY.md) 참조.

| 파일 | 내용 |
|---|---|
| `S6_A_t17_blurred_final.png` / `S6_C_t25_blurred_final.png` | 시야차단 실사 (앞 대형차가 후방 시야를 물리적으로 가리는 장면) |
| `s17_lead_vehicle_detection.png` | YOLO 선행차량 검출 (bbox + 차량검출 라벨) |
| `a01_brake_detected_dLamp44.9x_symY.png` | 브레이크등 검출 성공 (다중 단서: 밝은적색·대칭쌍·시간적 급증) |
| `d02_redbus_bright-fooled_corrected.png` | 붉은 차체 오검출을 시간·대칭 단서로 정정 |

> **개인정보**: 전 프레임 번호판·얼굴·상호·도로표지 지명·OSD를 되돌릴 수 없게 블러/크롭했고,
> OCR 재검사에서 번호판·전화 검출 0건을 확인했다. 원본 주행영상은 비공개다.
