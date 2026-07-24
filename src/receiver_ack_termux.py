"""
트랙 B 전용 경량 수신기 — 안드로이드 Termux에서 실행.

목적: 패킷을 받는 즉시 event_id를 담은 ACK를 송신측으로 되돌려주는 것.
송신측(sender.py --ack)이 "보낸 시각"과 "ACK 받은 시각"을 같은 클럭으로
재기 때문에 두 기기의 시계를 맞출 필요가 없다.

표준 라이브러리만 사용한다 (numpy/pygame/opencv 불필요).

Termux 준비:
    pkg install python
    # 두 기기가 같은 WiFi AP에 붙어 있어야 함
    python receiver_ack_termux.py --port 5005

옵션:
    --no-ack     ACK 없이 수신 로그만 남김
    --csv PATH   수신 시각 CSV 기록 (기본: 기록 안 함)
                 ※ 이 CSV의 시각은 폰 클럭이므로 PC 로그와 직접 빼면 안 된다.
                    트랙 B의 지연은 어디까지나 송신측 RTT로만 계산한다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
import time


def main() -> None:
    p = argparse.ArgumentParser(description="트랙 B ACK 에코 수신기 (Termux)")
    p.add_argument("--port", type=int, default=5005, help="수신 UDP 포트")
    p.add_argument("--no-ack", action="store_true", help="ACK 회신 비활성화")
    p.add_argument("--csv", default=None, help="수신 시각 CSV 경로 (선택)")
    p.add_argument(
        "--quiet", action="store_true", help="패킷마다 출력하지 않고 25개마다 요약"
    )
    args = p.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", args.port))

    writer = None
    fh = None
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        is_new = not os.path.exists(args.csv)
        fh = open(args.csv, "a", newline="", encoding="utf-8")
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(
                ["event_id", "stage", "wall_time_s", "mono_time_s", "role", "extra"]
            )

    print(f"[termux] UDP {args.port} 수신 대기 / ACK={'OFF' if args.no_ack else 'ON'}")
    print("[termux] Ctrl+C 로 종료")

    n = 0
    try:
        while True:
            data, addr = sock.recvfrom(65535)
            wall = time.time()
            mono = time.perf_counter()
            try:
                pkt = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            eid = pkt.get("event_id")
            if not eid:
                continue

            # 회신을 가장 먼저 — 폰 쪽 처리시간이 RTT에 섞이지 않게 한다.
            if not args.no_ack:
                try:
                    sock.sendto(json.dumps({"ack_event_id": eid}).encode("utf-8"), addr)
                except OSError as exc:
                    print(f"[termux] ACK 실패: {exc}", file=sys.stderr)

            n += 1
            if writer is not None:
                writer.writerow(
                    [
                        eid,
                        "udp_receive",
                        f"{wall:.6f}",
                        f"{mono:.6f}",
                        "receiver_termux",
                        str(pkt.get("risk_type", "")),
                    ]
                )
                if n % 20 == 0:
                    fh.flush()

            if args.quiet:
                if n % 25 == 0:
                    print(f"[termux] {n} 수신")
            else:
                print(f"[termux] {n:4d} {eid} from {addr[0]}")
    except KeyboardInterrupt:
        print(f"\n[termux] 종료 — 총 {n} 패킷 수신")
    finally:
        if fh is not None:
            fh.close()
        sock.close()


if __name__ == "__main__":
    main()
