"""
Pipe wrapper: annotates each line of stdin with current RSS of a target PID.
Also tracks and reports peak memory at exit.

Usage:
    python test_valis_reg.py 2>&1 | python memlog.py [--pid PID]

If --pid is not given, defaults to the current process group (the preceding pipe).
"""
import sys
import os
import argparse
import signal
import time
GB = 1024 ** 3

def get_pipe_writer_pid():
    """Return the PID of the process writing to our stdin via a pipe, or None."""
    if not sys.stdin.isatty():
        try:
            # Get the inode of the pipe attached to our stdin (fd 0)
            stdin_stat = os.fstat(0)
            if not os.path.exists(f"/proc/self/fd/0"):
                # Not a pipe or symlink missing
                return None
            pipe_inode = os.readlink("/proc/self/fd/0").split(':')[-1]
            # Scan /proc/*/fd/* to find a write end of the same pipe
            for pid_str in os.listdir('/proc'):
                if not pid_str.isdigit():
                    continue
                fd_dir = f"/proc/{pid_str}/fd"
                try:
                    for fd in os.listdir(fd_dir):
                        link = os.readlink(f"{fd_dir}/{fd}")
                        # Look for write end of a pipe: "pipe:[inode]"
                        if link.startswith('pipe:') and link.split(':')[-1] == pipe_inode:
                            # Check if the process has the pipe open for writing
                            # (readlink on a write end also shows "pipe:[inode]")
                            # It might be our own process reading – filter that out
                            if int(pid_str) != os.getpid():
                                return int(pid_str)
                except (OSError, PermissionError):
                    continue
        except Exception as e:
            # Fallback or log
            return None
    return None


def get_rss_gib(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    return -1


def main():
    parser = argparse.ArgumentParser(description="Annotate each line of stdin with memory usage.")
    parser.add_argument("--pid", type=int, default=None, help="Target PID (default: auto-detect)")
    args = parser.parse_args()

    pid = args.pid or get_pipe_writer_pid()

    # launch background plot-watcher alongside the pipe annotator
    import subprocess
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _watcher = subprocess.Popen(
        [sys.executable, os.path.join(_script_dir, "mem_watch.py"),
         str(pid), "--out", "mem_profile.png"],
    )
    sys.stderr.write(f"[memlog] mem_watch PID {_watcher.pid} launched for target {pid}\n")

    t0 = time.time()
    peak = 0.0
    peak_at = 0.0
    try:
        for line in sys.stdin:
            elapsed = time.time() - t0
            rss = get_rss_gib(pid)
            if rss > peak:
                peak = rss
                peak_at = elapsed
            mins = int(elapsed // 60)
            secs = elapsed % 60
            line = line.rstrip("\n")
            mem_str = (f"  \033[90m[MEM {rss:5.2f} GiB @ {mins}:{secs:04.1f}"
                       f" | peak {peak:5.2f} GiB @ {int(peak_at//60)}:{peak_at%60:04.1f}, {pid}]\033[0m"
                       if rss >= 0 else "")
            sys.stdout.write(f"{line}{mem_str}\n")
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        sys.stderr.write(f"[memlog] peak RSS: {peak:.2f} GiB @ {peak_at:.0f}s\n")


if __name__ == "__main__":
    main()
