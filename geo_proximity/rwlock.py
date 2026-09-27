"""A small writer-preferring read/write lock (no third-party deps).

Many concurrent readers proceed in parallel; writers are exclusive.
Writers are preferred over newly arriving readers once a writer is
waiting, which prevents writer starvation under a steady read load.

The implementation only uses :mod:`threading` primitives, so it works on
CPython 3.14 (including the free-threaded build) and requires no extra
dependencies.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator


class RWLock:
    """Writer-preferring multiple-reader / single-writer lock."""

    __slots__ = (
        "_cond",
        "_readers",
        "_writer_waiting",
        "_writer_active",
    )

    def __init__(self) -> None:
        # A single condition variable serialises all lock state changes;
        # the guarded sections below are trivially short.
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer_waiting = False
        self._writer_active = False

    @contextmanager
    def read_lock(self) -> Iterator[None]:
        with self._cond:
            while self._writer_active or self._writer_waiting:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write_lock(self) -> Iterator[None]:
        with self._cond:
            self._writer_waiting = True
            try:
                while self._readers > 0 or self._writer_active:
                    self._cond.wait()
                self._writer_active = True
            finally:
                self._writer_waiting = False
        try:
            yield
        finally:
            with self._cond:
                self._writer_active = False
                self._cond.notify_all()
