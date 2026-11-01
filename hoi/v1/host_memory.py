"""Sample Linux host memory for one process and its discovered descendants only.

PSS apportions shared pages; summing RSS would count them repeatedly. A sampled
peak is a lower bound, and process exit/spawn races can change the observed tree.
smaps_rollup does not expose exact private-anonymous memory: report Anonymous,
Pss_Anon and total private pages separately rather than inventing that quantity.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import threading
import time


_FIELDS = {
    "Rss": "rss_bytes", "Pss": "pss_bytes", "Pss_Anon": "pss_anon_bytes",
    "Anonymous": "anonymous_bytes", "Private_Clean": "private_clean_bytes",
    "Private_Dirty": "private_dirty_bytes", "Private_Hugetlb": "private_hugetlb_bytes",
}
_METRICS = (*_FIELDS.values(), "private_bytes")


def parse_smaps_rollup(text):
    """Parse only memory counters, retaining absent kernel fields as unknown."""
    values = {name: None for name in _FIELDS.values()}
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0].rstrip(":") not in _FIELDS:
            continue
        if len(fields) != 3 or fields[2] != "kB":
            raise ValueError("invalid smaps_rollup memory counter/unit")
        value = int(fields[1])
        if value < 0:
            raise ValueError("negative smaps_rollup memory counter")
        values[_FIELDS[fields[0].rstrip(":")]] = value * 1024
    if values["rss_bytes"] is None or values["pss_bytes"] is None:
        raise ValueError("smaps_rollup is missing Rss/Pss")
    private = [values["private_clean_bytes"], values["private_dirty_bytes"]]
    values["private_bytes"] = (sum(private) + (values["private_hugetlb_bytes"] or 0)
                               if all(value is not None for value in private) else None)
    values["private_anon_bytes"] = None
    return values


def _error(pid, operation, exc):
    return {"pid": pid, "operation": operation, "type": type(exc).__name__,
            "errno": getattr(exc, "errno", None), "message": str(exc)}


def sample_process_tree(root_pid, *, proc_root="/proc"):
    """Follow children files; never enumerate /proc or inspect unrelated PIDs."""
    if type(root_pid) is not int or root_pid < 1:
        raise ValueError("root_pid must be a positive integer")
    root = Path(proc_root)
    pending, seen, exited, errors = [root_pid], set(), set(), []
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            raw = (root / str(pid) / "task" / str(pid) / "children").read_text()
            children = [int(token) for token in raw.split()]
            if any(child < 1 for child in children):
                raise ValueError("invalid PID in owned children file")
            pending.extend(child for child in children if child not in seen)
        except (FileNotFoundError, ProcessLookupError):
            exited.add(pid)
        except (OSError, ValueError) as exc:
            errors.append(_error(pid, "read_children", exc))
    by_pid = {}
    for pid in sorted(seen - exited):
        try:
            by_pid[str(pid)] = parse_smaps_rollup((root / str(pid) / "smaps_rollup").read_text())
        except (FileNotFoundError, ProcessLookupError):
            exited.add(pid)
        except (OSError, ValueError) as exc:
            errors.append(_error(pid, "read_smaps_rollup", exc))
    totals = {}
    for metric in _METRICS:
        values = [record[metric] for record in by_pid.values()]
        totals[metric] = sum(values) if all(value is not None for value in values) else None
    totals["private_anon_bytes"] = None
    return {"root_pid": root_pid, "observed_pids": sorted(seen),
            "measured_pids": sorted(map(int, by_pid)), "exited_pids": sorted(exited),
            "by_pid": by_pid, "totals": totals, "errors": errors, "complete": not errors}


class HostMemorySampler:
    """Context-managed background sampler; read/permission failures stay visible.

    ``summary()`` may be called during execution. Errors invalidate complete
    measurement claims but do not interrupt the workload being measured.
    """
    def __init__(self, root_pid=None, *, interval_seconds=1.0, proc_root="/proc"):
        root_pid = os.getpid() if root_pid is None else root_pid
        if type(root_pid) is not int or root_pid < 1:
            raise ValueError("root_pid must be a positive integer")
        if type(interval_seconds) not in (int, float) or not 0 < interval_seconds < float("inf"):
            raise ValueError("interval_seconds must be finite and positive")
        self.root_pid, self.interval_seconds, self.proc_root = root_pid, float(interval_seconds), proc_root
        self.samples, self.peaks, self.sampler_errors = [], {}, []
        self._stop, self._lock = threading.Event(), threading.Lock()
        self._thread = None
        self._started = None

    def sample(self):
        started = time.monotonic()
        timestamp = datetime.now(timezone.utc).isoformat()
        sample = sample_process_tree(self.root_pid, proc_root=self.proc_root)
        sample.update(timestamp_utc=timestamp,
                      elapsed_seconds=started - (self._started if self._started is not None else started),
                      collection_duration_seconds=time.monotonic() - started)
        with self._lock:
            self.samples.append(sample)
            for metric in _METRICS:
                value = sample["totals"][metric]
                if value is not None and (metric not in self.peaks or value > self.peaks[metric]["bytes"]):
                    self.peaks[metric] = {"bytes": value, "timestamp_utc": sample["timestamp_utc"],
                        "elapsed_seconds": sample["elapsed_seconds"], "pids": sample["measured_pids"],
                        "sample_complete": sample["complete"]}
        return sample

    def _loop(self):
        while not self._stop.wait(self.interval_seconds):
            try:
                self.sample()
            except Exception as exc:
                with self._lock:
                    self.sampler_errors.append(_error(self.root_pid, "background_sample", exc))

    def summary(self):
        with self._lock:
            return {"schema": "hoi_host_memory_process_tree_v1", "root_pid": self.root_pid,
                "interval_seconds": self.interval_seconds, "units": "bytes",
                "scope": "root and recursive /proc/PID/task/PID/children descendants only",
                "sample_count": len(self.samples), "samples": list(self.samples), "peaks": dict(self.peaks),
                "sampler_errors": list(self.sampler_errors),
                "measurement_complete": bool(self.samples) and not self.sampler_errors and all(s["complete"] for s in self.samples),
                "sampled_peaks_are_lower_bounds": True,
                "private_anon_bytes": None,
                "private_anon_unavailable_reason": "smaps_rollup has no exact private-anonymous intersection; Pss_Anon and Anonymous are reported separately"}

    def __enter__(self):
        if self._thread is not None:
            raise RuntimeError("HostMemorySampler cannot be entered twice")
        self._started = time.monotonic()
        self.sample()
        self._thread = threading.Thread(target=self._loop, name="hoi-host-memory", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval_seconds * 2))
        if self._thread.is_alive():
            with self._lock:
                self.sampler_errors.append({"pid": self.root_pid, "operation": "join", "type": "TimeoutError",
                    "errno": None, "message": "host sampler did not stop within bounded join timeout"})
        else:
            try:
                self.sample()
            except Exception as exc:
                with self._lock:
                    self.sampler_errors.append(_error(self.root_pid, "final_sample", exc))
