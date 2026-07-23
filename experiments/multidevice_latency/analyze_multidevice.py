"""
다수 단말 지연 분석 — run_N*.csv 들을 읽어 N별 RTT 열화를 산출.

- N별 RTT 중앙값·p95·평균·유실률 (워밍업 제외).
- N=1 대비 증가량(중앙값·p95 열화).
- 프레임당 송출시간(send-loop ms)의 N별 중앙값(송신 루프 열화).
- 산출: multidevice_stats.json / report.md / multidevice_latency.png(대수 vs 지연 꺾은선).

방법론: 기존 트랙 B와 동일 — 같은 클럭 RTT, 무선, 보수적(RTT 미분할). 대표값은 중앙값,
보수 상한은 p95 (안드로이드 WiFi 절전으로 분포가 우편향되기 때문).

실행:
    python analyze_multidevice.py --glob "raw/run_N*.csv" --outdir .
"""
from __future__ import annotations
import argparse, csv, glob, json, os, sys
from collections import defaultdict
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def percentile(sv, q):
    if not sv:
        return float("nan")
    if len(sv) == 1:
        return sv[0]
    pos = (len(sv) - 1) * q
    lo = int(pos); hi = min(lo + 1, len(sv) - 1); frac = pos - lo
    return sv[lo] * (1 - frac) + sv[hi] * frac


def describe(vals):
    if not vals:
        return None
    s = sorted(vals); n = len(s); mean = sum(s) / n
    return {"n": n, "mean_ms": mean, "median_ms": percentile(s, 0.50),
            "p95_ms": percentile(s, 0.95), "min_ms": s[0], "max_ms": s[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="raw/run_N*.csv")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.outdir, a.glob)) or glob.glob(a.glob))
    if not files:
        raise SystemExit(f"[중단] CSV 없음: {a.glob}")

    # N(동시 수신 대수) = 그 실행의 distinct target 수
    per_n = {}     # N -> dict(rtts, frame_ms, measured, lost, devices)
    for fp in files:
        rows = list(csv.DictReader(open(fp, encoding="utf-8")))
        devices = sorted({r["target"] for r in rows})
        N = len(devices)
        rtts, frame_ms = [], []
        measured = lost = 0
        for r in rows:
            if r["warmup"] == "1":
                continue
            measured += 1
            if r["lost"] == "1":
                lost += 1
            else:
                rtts.append(float(r["rtt_ms"]))
            if r["frame_send_ms"]:
                frame_ms.append(float(r["frame_send_ms"]))
        d = per_n.setdefault(N, {"rtts": [], "frame_ms": [], "measured": 0,
                                 "lost": 0, "devices": devices, "files": []})
        d["rtts"] += rtts; d["frame_ms"] += frame_ms
        d["measured"] += measured; d["lost"] += lost; d["files"].append(os.path.basename(fp))

    Ns = sorted(per_n)
    base = per_n[Ns[0]]
    base_stat = describe(base["rtts"])
    base_med = base_stat["median_ms"] if base_stat else float("nan")
    base_p95 = base_stat["p95_ms"] if base_stat else float("nan")

    result = {"methodology": "same-clock echo RTT, WiFi, conservative(RTT 미분할). "
              "대표=중앙값, 보수상한=p95.", "by_N": {}}
    table = ["| N(동시수신) | 측정 | 유실률 | RTT중앙값(ms) | RTT p95(ms) | "
             "중앙값Δ vs N=1 | p95Δ vs N=1 | 송출/프레임 중앙값(ms) |",
             "|---|---|---|---|---|---|---|---|"]
    xs, med_y, p95_y, frame_y = [], [], [], []
    for N in Ns:
        d = per_n[N]; st = describe(d["rtts"]); fr = describe(d["frame_ms"])
        loss = 100.0 * d["lost"] / d["measured"] if d["measured"] else 0.0
        med = st["median_ms"] if st else float("nan")
        p95 = st["p95_ms"] if st else float("nan")
        fmed = fr["median_ms"] if fr else float("nan")
        result["by_N"][N] = {"devices": d["devices"], "files": d["files"],
                             "measured": d["measured"], "loss_pct": round(loss, 2),
                             "rtt": st, "frame_send_ms": fr,
                             "delta_median_ms": round(med - base_med, 2),
                             "delta_p95_ms": round(p95 - base_p95, 2)}
        table.append(f"| {N} | {d['measured']} | {loss:.1f}% | {med:.2f} | {p95:.2f} | "
                     f"{med-base_med:+.2f} | {p95-base_p95:+.2f} | {fmed:.3f} |")
        xs.append(N); med_y.append(med); p95_y.append(p95); frame_y.append(fmed)

    maxΔp95 = max((result["by_N"][N]["delta_p95_ms"] for N in Ns), default=0.0)
    report = [
        "# 다수 단말 동시 접속 시 지연 열화", "",
        "**방법론**: 송신측이 N대 수신 폰에 UDP 를 보내고 각 폰의 즉시 에코를 **같은 클럭**으로",
        "재어 RTT 측정(시계 동기 불필요). 정지·동일 WiFi·동일 폰 고정, 변수는 동시 수신 대수 N만.",
        "기존 지연 측정(트랙 B)과 동일하게 **보수적으로 RTT 를 분할하지 않고** 보고한다.",
        "대표값은 중앙값, 보수 상한은 p95(안드로이드 WiFi 절전으로 분포가 우편향).", "",
        "## N별 RTT 및 열화", "", *table, "",
        f"**해석**: N=1 대비 최대 p95 증가량 **{maxΔp95:+.2f} ms** "
        f"(N={Ns[-1]}까지). 즉 동시 {Ns[-1]}대에서도 "
        f"기준 대비 +{maxΔp95:.0f}ms 이내의 열화.", "",
        "> ⚠ 표본·환경(정지, 동일 WiFi, 폰 대수)에 한정된 값. 실주행·다른 AP 에서는 달라질 수 있음.",
        "> 송출/프레임 중앙값이 N 에 비례해 늘면 송신 루프가 병목이 되기 시작하는 신호.", "",
    ]
    with open(os.path.join(a.outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    with open(os.path.join(a.outdir, "multidevice_stats.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    _plot(xs, med_y, p95_y, frame_y, a.outdir)
    print("\n".join(table))
    print(f"\n[저장] {os.path.join(a.outdir,'report.md')} / multidevice_stats.json / multidevice_latency.png")


def _plot(xs, med_y, p95_y, frame_y, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for c in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        try:
            matplotlib.rcParams["font.family"] = c; break
        except Exception:
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.plot(xs, med_y, "-o", color="#256abf", lw=2.2, label="RTT 중앙값")
    ax.plot(xs, p95_y, "--s", color="#e34948", lw=2.0, label="RTT p95(보수)")
    for x, y in zip(xs, med_y):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, color="#256abf")
    for x, y in zip(xs, p95_y):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, color="#e34948")
    ax.set_xlabel("동시 수신 단말 수 N"); ax.set_ylabel("무선 RTT (ms)")
    ax.set_title("다수 단말 동시 접속 시 지연 (정지·동일 WiFi)")
    ax.set_xticks(xs); ax.grid(True, color="#e1e0d9", lw=0.8)
    ax.set_ylim(bottom=0); ax.legend()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "multidevice_latency.png"), dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
