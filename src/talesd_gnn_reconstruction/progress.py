from __future__ import annotations

import math
import os
import shutil
import sys
import time
from collections.abc import Iterator
from typing import Any, TextIO


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60.0:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minute = divmod(minutes, 60)
    return f"{hours}h{minute:02d}m"


def _progress_width() -> int:
    configured = os.environ.get("TALESD_GNN_PROGRESS_NCOLS")
    if configured:
        try:
            return max(int(configured), 40)
        except ValueError:
            pass
    columns = shutil.get_terminal_size(fallback=(100, 20)).columns
    return max(40, min(columns - 1, 120))


class LineProgress:
    def __init__(self, desc: str, total: int | None = None) -> None:
        self.desc = desc
        self.total = total
        self.count = 0
        self.started = time.perf_counter()
        self.last_report = self.started
        self.interval = max(float(os.environ.get("TALESD_GNN_PROGRESS_INTERVAL", "30")), 1.0)
        self.postfix = ""
        self.closed = False

    def update(self, value: int = 1) -> None:
        self.count += int(value)
        self._report_if_due()

    def set_postfix(self, **kwargs: Any) -> None:
        if kwargs:
            self.postfix = " " + " ".join(f"{key}={value}" for key, value in kwargs.items())

    def _line(self, final: bool = False) -> str:
        elapsed = time.perf_counter() - self.started
        rate = self.count / elapsed if elapsed > 0.0 else 0.0
        done = " done" if final else ""
        if self.total:
            percent = 100.0 * min(self.count, self.total) / max(self.total, 1)
            remaining = max(self.total - self.count, 0)
            eta = remaining / rate if rate > 0.0 else math.nan
            eta_text = " eta=unknown" if not math.isfinite(eta) else f" eta={_format_duration(eta)}"
            return (
                f"{self.desc}:{done} {self.count}/{self.total} ({percent:.1f}%) "
                f"elapsed={_format_duration(elapsed)} rate={rate:.3g}/s{eta_text}{self.postfix}"
            )
        return f"{self.desc}:{done} {self.count} elapsed={_format_duration(elapsed)} rate={rate:.3g}/s{self.postfix}"

    def _report_if_due(self) -> None:
        now = time.perf_counter()
        if now - self.last_report >= self.interval:
            print(self._line(), file=sys.stderr, flush=True)
            self.last_report = now

    def close(self) -> None:
        if not self.closed:
            print(self._line(final=True), file=sys.stderr, flush=True)
            self.closed = True


class LineProgressIterable:
    def __init__(self, iterable: Any, desc: str, total: int | None = None) -> None:
        self.iterable = iterable
        self.progress = LineProgress(desc=desc, total=total)

    def __iter__(self) -> Iterator[Any]:
        try:
            for item in self.iterable:
                yield item
                self.progress.update(1)
        finally:
            self.progress.close()

    def set_postfix(self, **kwargs: Any) -> None:
        self.progress.set_postfix(**kwargs)


class NullProgress:
    def update(self, _value: int = 1) -> None:
        return None

    def close(self) -> None:
        return None


def _tqdm_kwargs(leave: bool = True, position: int | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "leave": leave,
        "dynamic_ncols": False,
        "ncols": _progress_width(),
        "ascii": True,
        "mininterval": 0.5,
        "smoothing": 0.1,
        "file": sys.stderr,
        "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    }
    if position is not None:
        kwargs["position"] = position
    return kwargs


def progress(
    iterable: Any,
    desc: str,
    total: int | None = None,
    enabled: bool = True,
    leave: bool = True,
    position: int | None = None,
) -> Any:
    if not enabled:
        return iterable
    if not sys.stderr.isatty():
        return LineProgressIterable(iterable, desc=desc, total=total)
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc, total=total, **_tqdm_kwargs(leave=leave, position=position))
    except Exception:
        return LineProgressIterable(iterable, desc=desc, total=total)


def progress_bar(desc: str, total: int, enabled: bool = True, position: int | None = None) -> Any:
    if not enabled:
        return NullProgress()
    if not sys.stderr.isatty():
        return LineProgress(desc=desc, total=total)
    try:
        from tqdm import tqdm

        return tqdm(desc=desc, total=total, **_tqdm_kwargs(leave=True, position=position))
    except Exception:
        return LineProgress(desc=desc, total=total)


def write(message: str, *, file: TextIO | None = None, flush: bool = True) -> None:
    if file is None:
        file = sys.stderr
    if sys.stderr.isatty():
        try:
            from tqdm import tqdm

            tqdm.write(message, file=file)
            if flush:
                file.flush()
            return
        except Exception:
            pass
    print(message, file=file, flush=flush)
