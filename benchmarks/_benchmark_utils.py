"""Small cross-platform helpers shared by benchmark scripts."""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from contextlib import contextmanager


TIMING_FIELDS = ("section", "label", "step", "m", "n_blocks", "seconds")


class StepTimings:
    """Persist non-overlapping benchmark leaf timings as they complete.

    ``total()`` is the sole inclusive row. Writing after every completed step
    preserves useful evidence if a long external-data run later fails.
    """

    def __init__(self, path, *, clock=None):
        self.path = os.fspath(path)
        self._clock = time.perf_counter if clock is None else clock
        self._run_started = self._clock()
        self.rows = []

    @contextmanager
    def measure(self, section, label, step, *, m=None, n_blocks=None):
        """Measure one successful leaf step and append it to ``path``."""
        started = self._clock()
        yield
        self.add(section, label, step, self._clock() - started,
                 m=m, n_blocks=n_blocks)

    def add(self, section, label, step, seconds, *, m=None, n_blocks=None):
        """Append one already measured step and flush the long-form CSV."""
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("timing seconds must be finite and nonnegative")
        self.rows.append({
            "section": str(section),
            "label": str(label),
            "step": str(step),
            "m": "" if m is None else int(m),
            "n_blocks": "" if n_blocks is None else int(n_blocks),
            "seconds": f"{seconds:.6f}",
        })
        self.write()
        return seconds

    def total(self):
        """Append the one inclusive row and return its duration."""
        return self.add("run", "all", "total",
                        self._clock() - self._run_started)

    def write(self):
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        with open(self.path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TIMING_FIELDS,
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.rows)


def peak_rss_bytes() -> int:
    """Return this process's peak resident set size in bytes."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and the other supported Unix CI hosts report KiB.
    return int(peak if sys.platform == "darwin" else peak * 1024)
