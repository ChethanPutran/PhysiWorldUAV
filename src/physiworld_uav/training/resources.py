"""Per-epoch resource accounting for the trainer (peak GPU memory, resident memory, CPU time).

``ResourceMeter.start()`` at the beginning of an epoch, ``stop()`` at its end:

    gpu_peak_alloc_mb    peak memory allocated by tensors on the training device during the epoch (torch.cuda.max_memory_allocated)
    gpu_peak_reserved_mb peak memory reserved by the caching allocator (what nvidia-smi shows for the process, minus the CUDA context)
    rss_mb / rss_peak_mb resident memory of the process at the end of the epoch / its peak since the process started (VmHWM)
    cpu_s / cpu_pct      CPU time used by the process during the epoch and the share of one core it represents
    wall_s               wall-clock duration of the epoch (training + validation)

GPU values are None when training runs on the CPU. Pure stdlib + torch; Linux only for the /proc reads (None elsewhere).
"""
from __future__ import annotations

import time
from pathlib import Path

import torch


def _proc_status(key: str) -> float | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(key):
                return float(line.split()[1]) / 1024.0           # kB -> MB
    except OSError:
        pass
    return None


class ResourceMeter:
    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self._t0 = self._c0 = 0.0
        self.peak_gpu_mb: float | None = None
        self.peak_rss_mb: float | None = None

    def start(self) -> None:
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        self._t0, self._c0 = time.time(), time.process_time()

    def stop(self) -> dict:
        wall, cpu = time.time() - self._t0, time.process_time() - self._c0
        out = {"wall_s": wall, "cpu_s": cpu, "cpu_pct": 100.0 * cpu / max(wall, 1e-9),
               "rss_mb": _proc_status("VmRSS"), "rss_peak_mb": _proc_status("VmHWM"),
               "gpu_peak_alloc_mb": None, "gpu_peak_reserved_mb": None}
        if self.cuda:
            out["gpu_peak_alloc_mb"] = torch.cuda.max_memory_allocated(self.device) / 2 ** 20
            out["gpu_peak_reserved_mb"] = torch.cuda.max_memory_reserved(self.device) / 2 ** 20
            self.peak_gpu_mb = max(self.peak_gpu_mb or 0.0, out["gpu_peak_reserved_mb"])
        if out["rss_peak_mb"] is not None:
            self.peak_rss_mb = out["rss_peak_mb"]
        return out
