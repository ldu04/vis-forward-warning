"""
지연시간 분석 — sender/receiver CSV를 event_id로 조인해 구간별 통계 산출.

사용:
    python analyze_latency.py --sender raw/trackA_sender.csv \
                              --receiver raw/trackA_receiver.csv \
                              --outdir .

트랙 B(RTT)는 sender CSV 하나만 있으면 된다 (udp_send → ack_recv):
    python analyze_latency.py --sender raw/trackB_sender.csv --outdir .

설계 메모
--------
- 트랙 A(단일 기기)는 sender/receiver가 같은 클럭이므로 wall_time_s 차이가
  그대로 편도 구간 지연이 된다.
- 트랙 B(2기기)는 시계가 어긋나므로 편도를 절대 계산하지 않는다.
  같은 기기에서 잰 udp_send → ack_recv 차이(RTT)만 쓴다.
- 워밍업 이벤트(extra에 'warmup')는 통계에서 제외하되 raw CSV에는 남아있다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

# 트랙 A 편도 구간 정의: (표시이름, 시작 stage, 끝 stage)
TRACK_A_SEGMENTS = [
    ("YOLO 추론 (capture→yolo)", "frame_capture", "yolo_detect_done"),
    ("직렬화/송신 (yolo→send)", "yolo_detect_done", "udp_send"),
    ("네트워크 편도 (send→recv)", "udp_send", "udp_receive"),
    ("수신→UI 표시 (recv→UI)", "udp_receive", "ui_alert_shown"),
    ("종단간 (capture→UI)", "frame_capture", "ui_alert_shown"),
]

# 추론 지연과 네트워크 지연을 반드시 분리해서 보고하기 위한 분류
INFERENCE_SEGMENTS = {"YOLO 추론 (capture→yolo)"}
NETWORK_SEGMENTS = {"네트워크 편도 (send→recv)"}


def read_stages(path: str) -> Dict[str, Dict[str, dict]]:
    """CSV → {event_id: {stage: row}}"""
    out: Dict[str, Dict[str, dict]] = defaultdict(dict)
    if not os.path.exists(path):
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = row.get("event_id")
            stage = row.get("stage")
            if not eid or not stage:
                continue
            # 같은 event_id+stage가 중복되면 첫 기록을 신뢰 (재전송 방어)
            if stage not in out[eid]:
                out[eid][stage] = row
    return out


def is_warmup(stages: Dict[str, dict]) -> bool:
    for row in stages.values():
        if "warmup" in (row.get("extra") or ""):
            return True
    return False


def percentile(sorted_vals: List[float], q: float) -> float:
    """선형보간 백분위수 (numpy 없이도 동작)."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def describe(vals: List[float]) -> Optional[dict]:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    var = sum((v - mean) ** 2 for v in s) / n
    return {
        "n": n,
        "mean_ms": mean,
        "median_ms": percentile(s, 0.50),
        "p95_ms": percentile(s, 0.95),
        "p99_ms": percentile(s, 0.99),
        "min_ms": s[0],
        "max_ms": s[-1],
        "std_ms": var ** 0.5,
    }


def collect_track_a(
    sender: Dict[str, Dict[str, dict]],
    receiver: Dict[str, Dict[str, dict]],
    include_warmup: bool,
) -> Dict[str, List[float]]:
    merged: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for eid, st in sender.items():
        merged[eid].update(st)
    for eid, st in receiver.items():
        merged[eid].update(st)

    series: Dict[str, List[float]] = {name: [] for name, _, _ in TRACK_A_SEGMENTS}
    for eid, stages in merged.items():
        if not include_warmup and is_warmup(stages):
            continue
        for name, a, b in TRACK_A_SEGMENTS:
            ra, rb = stages.get(a), stages.get(b)
            if not ra or not rb:
                continue
            dt = (float(rb["wall_time_s"]) - float(ra["wall_time_s"])) * 1000.0
            if dt < 0:
                continue  # 음수는 조인 오류로 보고 버린다
            series[name].append(dt)
    return series


def collect_track_b(
    sender: Dict[str, Dict[str, dict]], include_warmup: bool
) -> List[float]:
    """udp_send → ack_recv (같은 기기, 같은 클럭) = RTT"""
    rtts: List[float] = []
    for eid, stages in sender.items():
        if not include_warmup and is_warmup(stages):
            continue
        snd, ack = stages.get("udp_send"), stages.get("ack_recv")
        if not snd or not ack:
            continue
        dt = (float(ack["mono_time_s"]) - float(snd["mono_time_s"])) * 1000.0
        if dt >= 0:
            rtts.append(dt)
    return rtts


def fmt_row(name: str, d: Optional[dict]) -> str:
    if not d:
        return f"| {name} | — | — | — | — | — | — |"
    return (
        f"| {name} | {d['n']} | {d['mean_ms']:.2f} | {d['median_ms']:.2f} | "
        f"{d['p95_ms']:.2f} | {d['max_ms']:.2f} | {d['std_ms']:.2f} |"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="VIS 지연시간 구간 분석")
    p.add_argument("--sender", required=True, help="sender CSV 경로")
    p.add_argument("--receiver", default=None, help="receiver CSV 경로 (트랙 A)")
    p.add_argument("--outdir", default=".", help="결과 출력 폴더")
    p.add_argument("--label", default="trackA", help="출력 파일 접두사")
    p.add_argument(
        "--include-warmup",
        action="store_true",
        help="워밍업 이벤트도 통계에 포함 (기본: 제외)",
    )
    p.add_argument("--hist", action="store_true", help="히스토그램/박스플롯 PNG 생성")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sender = read_stages(args.sender)
    receiver = read_stages(args.receiver) if args.receiver else {}

    result: Dict[str, object] = {
        "sender_csv": args.sender,
        "receiver_csv": args.receiver,
        "warmup_excluded": not args.include_warmup,
        "sender_events": len(sender),
        "receiver_events": len(receiver),
    }

    lines: List[str] = []
    series = collect_track_a(sender, receiver, args.include_warmup)
    stats_a = {name: describe(v) for name, v in series.items()}
    result["track_a"] = stats_a

    if any(stats_a.values()):
        lines.append("## 트랙 A — 구간별 편도 지연 (ms)\n")
        lines.append("| 구간 | N | 평균 | 중앙값 | p95 | 최대 | 표준편차 |")
        lines.append("|---|---|---|---|---|---|---|")
        for name, _, _ in TRACK_A_SEGMENTS:
            lines.append(fmt_row(name, stats_a[name]))
        lines.append("")

        infer = next(
            (stats_a[n] for n in stats_a if n in INFERENCE_SEGMENTS and stats_a[n]), None
        )
        net = next(
            (stats_a[n] for n in stats_a if n in NETWORK_SEGMENTS and stats_a[n]), None
        )
        if infer and net:
            lines.append(
                f"- 추론 지연 평균 **{infer['mean_ms']:.2f} ms** vs "
                f"네트워크 편도 평균 **{net['mean_ms']:.2f} ms** "
                f"(추론이 네트워크의 {infer['mean_ms'] / max(net['mean_ms'], 1e-9):.1f}배)"
            )
            lines.append("")

    rtts = collect_track_b(sender, args.include_warmup)
    stats_b = describe(rtts)
    result["track_b_rtt"] = stats_b
    if stats_b:
        lines.append("## 트랙 B — 무선 RTT (ms)\n")
        lines.append("| 지표 | N | 평균 | 중앙값 | p95 | 최대 | 표준편차 |")
        lines.append("|---|---|---|---|---|---|---|")
        lines.append(fmt_row("RTT (udp_send→ack_recv)", stats_b))
        lines.append("")
        lines.append(
            f"- 편도 추정 = RTT/2 → 평균 {stats_b['mean_ms'] / 2:.2f} ms / "
            f"p95 {stats_b['p95_ms'] / 2:.2f} ms "
            "(추정치. 제안서 본문에는 보수적으로 RTT 기준으로 서술할 것)"
        )
        lines.append("")

    md = "\n".join(lines) if lines else "(조인된 이벤트 없음)\n"
    md_path = os.path.join(args.outdir, f"{args.label}_stats.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    json_path = os.path.join(args.outdir, f"{args.label}_stats.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(md)
    print(f"[저장] {md_path}")
    print(f"[저장] {json_path}")

    if args.hist:
        make_plots(series, rtts, args.outdir, args.label)


def make_plots(
    series: Dict[str, List[float]], rtts: List[float], outdir: str, label: str = ""
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 한글 폰트 (Windows 기본 맑은 고딕)
    for cand in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        try:
            matplotlib.rcParams["font.family"] = cand
            break
        except Exception:
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False

    panels = [(n, v) for n, v in series.items() if v]
    has_rtt = bool(rtts)
    ncols = 2
    nrows = (len(panels) + (1 if has_rtt else 0) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.4 * nrows))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    idx = 0
    for name, vals in panels:
        ax = axes[idx]
        ax.hist(vals, bins=40, color="#4C78A8", edgecolor="white")
        mean_v = sum(vals) / len(vals)
        p95_v = percentile(sorted(vals), 0.95)
        ax.axvline(mean_v, color="#E45756", lw=1.6, label=f"평균 {mean_v:.1f}ms")
        ax.axvline(p95_v, color="#F58518", lw=1.6, ls="--", label=f"p95 {p95_v:.1f}ms")
        ax.set_title(f"{name}  (N={len(vals)})", fontsize=10)
        ax.set_xlabel("ms")
        ax.set_ylabel("건수")
        ax.legend(fontsize=8)
        idx += 1

    if has_rtt:
        ax = axes[idx]
        ax.hist(rtts, bins=40, color="#54A24B", edgecolor="white")
        mean_v = sum(rtts) / len(rtts)
        p95_v = percentile(sorted(rtts), 0.95)
        ax.axvline(mean_v, color="#E45756", lw=1.6, label=f"평균 {mean_v:.1f}ms")
        ax.axvline(p95_v, color="#F58518", lw=1.6, ls="--", label=f"p95 {p95_v:.1f}ms")
        ax.set_title(f"트랙 B RTT  (N={len(rtts)})", fontsize=10)
        ax.set_xlabel("ms")
        ax.set_ylabel("건수")
        ax.legend(fontsize=8)
        idx += 1

    for j in range(idx, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    # 파일명에 label을 넣는다. 넣지 않으면 트랙 B 실행이 트랙 A의 그림을 덮어쓴다.
    suffix = "" if label in ("", "trackA") else f"_{label}"
    out = os.path.join(outdir, f"latency_hist{suffix}.png")
    fig.savefig(out, dpi=140)
    print(f"[저장] {out}")

    # 박스플롯 (구간 비교용)
    if panels:
        fig2, ax2 = plt.subplots(figsize=(11, 5))
        ax2.boxplot(
            [v for _, v in panels], labels=[n for n, _ in panels], showfliers=False
        )
        ax2.set_ylabel("ms")
        ax2.set_title("트랙 A 구간별 지연 분포 (이상치 제외)")
        plt.xticks(rotation=18, ha="right", fontsize=9)
        fig2.tight_layout()
        out2 = os.path.join(outdir, f"latency_box{suffix}.png")
        fig2.savefig(out2, dpi=140)
        print(f"[저장] {out2}")


if __name__ == "__main__":
    main()
