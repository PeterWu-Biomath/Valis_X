#!/usr/bin/env python
"""
Background RSS sampler — poll a PID at fixed intervals, plot a profile on exit.

Usage:
    python utils/mem_watch.py <pid> [--interval 1] [--out mem_profile.png]

The watcher polls /proc/<pid>/status every *interval* seconds.  It exits
automatically when the target PID disappears (process dies), or on SIGTERM /
SIGINT / after an optional --duration.  On exit it writes a matplotlib plot.

To launch from the pipeline process itself:

    import os, subprocess
    subprocess.Popen(["python", "utils/mem_watch.py", str(os.getpid())])
"""

import sys
import os
import time
import signal
import argparse
import numpy as np

HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    pass


def _rss_gib(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024 ** 2)  # GiB
    except Exception:
        pass
    return -1.0


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def main():
    parser = argparse.ArgumentParser(description="Background RSS watcher + plotter")
    parser.add_argument("pid", type=int, help="Target PID to monitor")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Sampling interval in seconds (default: 1)")
    parser.add_argument("--out", default="mem_profile.png",
                        help="Output plot path (default: mem_profile.png)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after N seconds (default: until PID exits)")
    parser.add_argument("--title", default=None,
                        help="Plot title")
    args = parser.parse_args()

    data = []       # list of (t_sec, rss_gib)
    peak = 0.0
    peak_at = 0.0
    t0 = time.time()
    deadline = t0 + args.duration if args.duration else None

    # sample loop
    try:
        while True:
            if not _pid_alive(args.pid):
                print(f"[mem_watch] PID {args.pid} exited — stopping")
                break
            if deadline and time.time() >= deadline:
                print(f"[mem_watch] duration {args.duration}s reached — stopping")
                break

            t = time.time() - t0
            rss = _rss_gib(args.pid)
            data.append((t, rss))
            if rss > peak:
                peak = rss
                peak_at = t
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("[mem_watch] interrupted — stopping")

    if not data:
        print("[mem_watch] no samples collected")
        return

    t_arr = np.array([d[0] for d in data])
    rss_arr = np.array([d[1] for d in data])

    # save raw data as text
    txt_path = args.out.replace(".png", ".txt")
    with open(txt_path, "w") as f:
        f.write("# t_sec  rss_gib\n")
        for t, r in zip(t_arr, rss_arr):
            f.write(f"{t:.1f}  {r:.2f}\n")
    print(f"[mem_watch] raw data saved to {txt_path}")

    # print summary
    peak_m, peak_s = divmod(peak_at, 60)
    print(f"[mem_watch] samples: {len(data)}  peak: {peak:.2f} GiB @ {int(peak_m)}:{peak_s:04.1f}  "
          f"duration: {t_arr[-1]:.0f}s")

    if not HAS_MPL:
        print("[mem_watch] matplotlib not available — skipping plot.  Data in numpy arrays.")
        # save raw data so it's not lost
        np.savez(args.out.replace(".png", ".npz"), t=t_arr, rss=rss_arr, peak=peak)
        return

    # plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_arr / 60, rss_arr, linewidth=0.5, color="steelblue")
    peak_m, peak_s = divmod(peak_at, 60)
    ax.axhline(y=peak, color="red", linestyle="--", linewidth=0.8,
               label=f"Peak: {peak:.2f} GiB @ {int(peak_m)}:{peak_s:04.1f}")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("RSS (GiB)")
    ax.set_title(args.title or f"Memory profile (PID {args.pid})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"[mem_watch] plot saved to {args.out}")
    


if __name__ == "__main__":
    main()
