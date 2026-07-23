"""
다수 단말 지연 측정 — 송신 노드(트럭 역할). 기존 브로드캐스트 쪽에서 실행.

동작: 매 프레임마다 연결된 N대 수신 폰 전부에게 UDP 패킷을 보내고, 각 폰의 에코를
받아 **같은 클럭(mono)** 으로 RTT를 잰다(시계 동기 불필요, 트랙 B와 동일 방법론).
변수는 '동시 수신 대수 N'만 — targets 로 N을 지정한다.

부가: 프레임당 'N대 전부에게 송출하는 데 걸린 시간(send-loop ms)'도 기록해,
수신 대수 증가가 송신 루프를 느리게 하는지 본다.

★ 루프백 우회: targets 에는 반드시 실제 WiFi IP를 준다(127.0.0.1 금지 — 무선 구간 미포함).
   (로컬 자체검증 시에만 MDL_ALLOW_LOOPBACK=1 로 예외.)

출력 CSV 컬럼: label, seq, warmup, target, send_mono_s, ack_mono_s, rtt_ms, lost, frame_send_ms

실행:
    python mdl_sender.py --targets 192.168.0.11:50007 192.168.0.12:50007 \
        --packets 150 --interval 0.02 --label N2 --out raw/run_N2.csv
"""
from __future__ import annotations
import argparse, csv, os, socket, sys, time
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def mono():
    return time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True, help="ip:port 목록 (= 동시 수신 N대)")
    ap.add_argument("--packets", type=int, default=150, help="프레임(라운드) 수")
    ap.add_argument("--interval", type=float, default=0.02, help="프레임 간 간격(초)")
    ap.add_argument("--timeout", type=float, default=0.5, help="프레임당 에코 대기 상한(초)")
    ap.add_argument("--warmup", type=int, default=10, help="통계 제외 워밍업 프레임 수")
    ap.add_argument("--label", default="N", help="이 실행의 N 라벨 (예: N1/N2/N3)")
    ap.add_argument("--out", required=True, help="출력 CSV 경로")
    a = ap.parse_args()

    targets = []
    for t in a.targets:
        ip, port = t.rsplit(":", 1)
        if ip in ("127.0.0.1", "localhost") and not os.environ.get("MDL_ALLOW_LOOPBACK"):
            raise SystemExit(f"[중단] 루프백 대상 금지({t}). 실제 WiFi IP 사용. "
                             "(로컬 자체검증 시에만 MDL_ALLOW_LOOPBACK=1)")
        targets.append((ip, int(port)))
    N = len(targets)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", 0))
    rows = []
    n_lost = 0
    print(f"[sender] N={N} targets={targets} packets={a.packets} label={a.label}", flush=True)

    for seq in range(a.packets):
        is_warm = 1 if seq < a.warmup else 0
        sent = {}          # ti -> send_mono
        ack = {}           # ti -> ack_mono
        frame_t0 = mono()
        for ti, tgt in enumerate(targets):
            send_mono = mono()
            s.sendto(f"{seq}:{ti}:{send_mono:.6f}".encode(), tgt)
            sent[ti] = send_mono
        frame_send_ms = (mono() - frame_t0) * 1000.0

        outstanding = set(range(N))
        deadline = mono() + a.timeout
        while outstanding:
            remain = deadline - mono()
            if remain <= 0:
                break
            try:
                s.settimeout(remain)
                data, _ = s.recvfrom(2048)
            except socket.timeout:
                break
            try:
                p_seq, p_ti, _sm = data.decode().split(":")
                p_seq, p_ti = int(p_seq), int(p_ti)
            except Exception:
                continue
            if p_seq == seq and p_ti in outstanding:
                ack[p_ti] = mono()
                outstanding.discard(p_ti)

        for ti, tgt in enumerate(targets):
            if ti in ack:
                rtt_ms = (ack[ti] - sent[ti]) * 1000.0
                rows.append([a.label, seq, is_warm, f"{tgt[0]}:{tgt[1]}",
                             f"{sent[ti]:.6f}", f"{ack[ti]:.6f}", f"{rtt_ms:.4f}",
                             0, f"{frame_send_ms:.4f}"])
            else:
                if not is_warm:
                    n_lost += 1
                rows.append([a.label, seq, is_warm, f"{tgt[0]}:{tgt[1]}",
                             f"{sent[ti]:.6f}", "", "", 1, f"{frame_send_ms:.4f}"])

        elapsed = mono() - frame_t0
        if a.interval - elapsed > 0:
            time.sleep(a.interval - elapsed)

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "seq", "warmup", "target", "send_mono_s",
                    "ack_mono_s", "rtt_ms", "lost", "frame_send_ms"])
        w.writerows(rows)

    got = [r for r in rows if r[2] == 0 and r[7] == 0]
    rtts = sorted(float(r[6]) for r in got)
    med = rtts[len(rtts) // 2] if rtts else float("nan")
    total_meas = sum(1 for r in rows if r[2] == 0)
    print(f"[sender] 저장 {a.out} — 측정 {total_meas} (유실 {n_lost}), "
          f"RTT 중앙값 {med:.2f} ms", flush=True)


if __name__ == "__main__":
    main()
