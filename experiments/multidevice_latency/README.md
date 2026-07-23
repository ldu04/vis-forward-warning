# 다수 단말 동시 접속 시 지연 열화 측정

기존 지연 측정(트랙 B)과 **동일 방법론**: 송신측이 N대 수신 폰에 UDP 를 보내고 각 폰의
즉시 에코를 **같은 클럭(mono)** 으로 재어 RTT 측정 → 시계 동기 불필요, 무선 구간 포함,
보수적으로 RTT 를 분할하지 않고 보고. 변수는 **동시 수신 대수 N** 하나만.

- 송신(트럭 역할): **이 PC 터미널**에서 `mdl_sender.py`
- 수신(에코): **폰 최대 5대**에서 `mdl_echo.py` (Termux)
- 대표값 = RTT 중앙값, 보수 상한 = p95 (안드로이드 WiFi 절전으로 분포 우편향)

## 준비

1. **같은 WiFi**: PC 와 폰 전부 동일 공유기(AP)에 접속. 정지 환경.
2. **PC 의 WiFi IP 확인**은 불필요(송신은 폰 IP 로 보냄). **각 폰의 IP** 를 확인:
   - Termux: `ip addr show wlan0 | grep 'inet '` → 예: `192.168.0.11`
3. **폰에 Termux + python** 설치, `mdl_echo.py` 파일을 폰으로 복사(또는 붙여넣기).
4. **방화벽**: PC 에서 UDP 인바운드(에코 수신) 허용. 잘 안 되면 아래 '문제 해결' 참조.

## 실행 순서

### ① 각 폰에서 에코 실행 (대수만큼)
```
python mdl_echo.py --port 50007
```
5대면 5대 각자 실행(포트 동일 50007, IP 는 폰마다 다름).

### ② PC 터미널에서 N 을 늘려가며 송신
정지·동일 WiFi·동일 폰 고정, **동시 켜는 폰 대수만 바꾼다**. 각 N 에서 150패킷:
```
# N=1 (폰 1대만 에코 실행 중)
python mdl_sender.py --targets 192.168.0.11:50007 --packets 150 --label N1 --out raw/run_N1.csv

# N=2 (폰 2대 에코 실행 중)
python mdl_sender.py --targets 192.168.0.11:50007 192.168.0.12:50007 --packets 150 --label N2 --out raw/run_N2.csv

# N=3,4,5 … IP 를 늘려가며 동일하게
python mdl_sender.py --targets <IP1>:50007 <IP2>:50007 <IP3>:50007 --packets 150 --label N3 --out raw/run_N3.csv
```
- `--packets` 100~200 권장. `--interval` 기본 0.02s(≈50fps 송출). `--warmup` 앞 10프레임은 통계 제외(WiFi 절전 깨우기 구간).
- ★ `--targets` 에 **실제 폰 WiFi IP** 만. `127.0.0.1` 은 무선 미포함이라 막아뒀음.

### ③ PC 에서 분석 → 표·그래프·리포트
```
python analyze_multidevice.py --glob "raw/run_N*.csv" --outdir .
```
산출: `report.md`, `multidevice_stats.json`, `multidevice_latency.png`(대수 vs RTT 꺾은선).

## 산출물 해석
- N별 RTT **중앙값·p95**, 그리고 **N=1 대비 증가량**(열화)이 표/그래프로 나옴.
- 부가: **송출/프레임 중앙값(ms)** — N 이 늘 때 송신 루프가 느려지면 병목 신호.
- 문안에는 "N대 동시에도 +Xms 이내" 형태로, 표본·환경 한정 조건을 병기해 보수적으로 서술.

## 자체검증(로컬)
`_selftest/` 는 폰 없이 localhost 에코로 스크립트 배관만 검증한 결과다(RTT≈0.2ms, **무선 아님**).
실제 수치가 아니며 참고용. 실측은 위 ①~③ 로 raw/ 에 생성한다.

## 문제 해결
- **전부 유실(loss 100%)**: 폰 방화벽/절전, 또는 PC 인바운드 UDP 차단. 폰에서 에코가 뜨는지,
  PC↔폰 `ping` 되는지 먼저 확인. 공유기의 'AP 격리(client isolation)' 가 켜져 있으면 꺼야 함.
- **일부만 유실**: 정상(안드로이드 WiFi 절전). warmup 제외 후에도 남으면 `--timeout` 을 늘려볼 것.
- **한글 깨짐 없음**: 출력은 UTF-8 강제(스크립트 상단), Windows cp949 콘솔에서도 안전.
