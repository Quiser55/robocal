"""Runner instrumentation: where a run is, and what it is waiting for."""
from __future__ import annotations

import os
import sys
import threading
import time

#: Seconds between "still waiting" lines, without and with --debug.
DEFAULT_INTERVAL = 60.0
DEBUG_INTERVAL = 5.0

_enabled = False
_interval = DEFAULT_INTERVAL
_t0 = time.time()
_lock = threading.Lock()


def enable(debug: bool) -> None:
    global _enabled, _interval, _t0
    _enabled = bool(debug)
    _interval = DEBUG_INTERVAL if _enabled else DEFAULT_INTERVAL
    _t0 = time.time()


def is_enabled() -> bool:
    return _enabled


def _stamp() -> str:
    elapsed = int(time.time() - _t0)
    return f'{elapsed // 60:02d}:{elapsed % 60:02d}'


def emit(message: str) -> None:
    """Progress the operator needs whether or not --debug is on."""
    with _lock:
        print(f'    [{_stamp()}] {message}', file=sys.stderr, flush=True)


def debug(message: str) -> None:
    if _enabled:
        emit(message)


class waiting:
    """Announce a blocking wait, and keep announcing it until it returns"""

    def __init__(self, label: str, timeout: float):
        self.label = label
        self.timeout = float(timeout)
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0

    def __enter__(self) -> 'waiting':
        self._started = time.time()
        debug(f'-> {self.label} (timeout {self.timeout:.0f}s)')
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()
        return self

    def _tick(self) -> None:
        while not self._done.wait(_interval):
            elapsed = time.time() - self._started
            emit(f'still waiting on {self.label}: {elapsed:.0f}s elapsed, '
                 f'{max(0.0, self.timeout - elapsed):.0f}s before timeout')

    def __exit__(self, *exc) -> bool:
        self._done.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        debug(f'<- {self.label} after {time.time() - self._started:.1f}s')
        return False


class LogTail:
    """Echo a child's log file to stderr as it is written."""
    def __init__(self, path: str, prefix: str, start: int = 0):
        self.path = path
        self.prefix = prefix
        self.start = start
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._follow, daemon=True)
        self._thread.start()

    def _follow(self) -> None:
        handle = None
        try:
            while not self._stop.is_set():
                if handle is None:
                    if not os.path.exists(self.path):
                        self._stop.wait(0.2)
                        continue
                    handle = open(self.path, 'r', errors='replace')
                    handle.seek(min(self.start, os.path.getsize(self.path)))
                line = handle.readline()
                if not line:
                    self._stop.wait(0.2)
                    continue
                with _lock:
                    print(f'      [{self.prefix}] {line.rstrip()}',
                          file=sys.stderr, flush=True)
        except OSError:
            pass
        finally:
            if handle is not None:
                handle.close()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
