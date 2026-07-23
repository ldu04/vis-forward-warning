"""
다수 단말 지연 측정 — 수신(에코) 노드. 각 폰(Termux) 또는 임의 호스트에서 실행.

역할: UDP 패킷을 받는 즉시 그대로 송신측에 되돌린다(에코). 처리를 최소화해
'수신 즉시 ACK' 조건을 지킨다. 시계 동기가 불필요하도록 RTT는 송신측이 잰다
(기존 트랙 B 방법론과 동일).

실행(폰 Termux 예):
    python mdl_echo.py --port 50007
여러 폰이면 같은 포트로 각자 실행하면 된다(폰마다 IP 다름).
"""
from __future__ import annotations
import argparse, socket, sys, time
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=50007)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((a.host, a.port))
    print(f"[echo] listening {a.host}:{a.port} - 패킷 즉시 반사. Ctrl-C 종료.", flush=True)
    n = 0
    try:
        while True:
            data, addr = s.recvfrom(2048)
            s.sendto(data, addr)          # 즉시 에코
            n += 1
            if not a.quiet and n % 100 == 0:
                print(f"  echoed {n}", flush=True)
    except KeyboardInterrupt:
        print(f"\n[echo] 종료. 총 {n} 반사.", flush=True)

if __name__ == "__main__":
    main()
