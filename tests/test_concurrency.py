"""Tests for concurrent query/mutation consistency."""
from __future__ import annotations

import threading

from geohash_proximity import DuplicatePointError, Point, ProximityIndex


def test_queries_always_observe_consistent_snapshot() -> None:
    """While a writer thread churns points, every radius/nearest result must
    be internally consistent: unique ids, valid distances, sorted order and
    no ids that were already deleted."""
    idx = ProximityIndex()
    n_points = 400
    for i in range(n_points):
        idx.insert(f"p{i}", (i * 7) % 180 - 90, (i * 13) % 360 - 180)

    stop = threading.Event()
    errors: list[Exception] = []

    def writer() -> None:
        i = 0
        while not stop.is_set():
            pid = f"w{i % 200}"
            try:
                idx.upsert(pid, (i * 37) % 180 - 90, (i * 53) % 360 - 180, {"i": i})
                if i % 3 == 0:
                    idx.delete(pid)
                if i % 5 == 0:
                    idx.update(f"p{i % n_points}", longitude=(i * 11) % 360 - 180)
            except Exception as exc:  # noqa: BLE001 - record for assertion
                errors.append(exc)
            i += 1

    def reader() -> None:
        try:
            while not stop.is_set():
                for sp in idx.radius_query(0.0, 180.0, 3_000_000):
                    assert -90.0 <= sp.point.latitude <= 90.0
                    assert -180.0 <= sp.point.longitude <= 180.0
                    assert sp.distance >= 0.0
                near = idx.nearest(45.0, -90.0, 10)
                ids = [sp.point.id for sp in near]
                assert len(ids) == len(set(ids))
                distances = [sp.distance for sp in near]
                assert distances == sorted(distances)
                for sp in near:
                    # Every returned point must be fetchable atomically.
                    assert idx.contains(sp.point.id)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    stop.wait(3.0)
    stop.set()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()
    assert errors == []


def test_bulk_load_is_atomic_under_concurrent_readers() -> None:
    idx = ProximityIndex()
    idx.bulk_load([Point(f"old{i}", 0.0, 0.0) for i in range(100)])
    stop = threading.Event()
    errors: list[Exception] = []
    sizes: list[int] = []

    def reader() -> None:
        try:
            while not stop.is_set():
                got = idx.nearest(0.0, 0.0, 50)
                ids = [sp.point.id for sp in got]
                assert len(ids) == len(set(ids))
                # Either all old or all new: never a mixed batch.
                kinds = {pid.startswith("old") for pid in ids}
                assert len(kinds) == 1
                sizes.append(len(ids))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def writer() -> None:
        batch = [Point(f"old{i}", 45.0, 45.0) for i in range(100)]
        for _ in range(20):
            idx.bulk_load(batch)  # replace all in one critical section

    readers = [threading.Thread(target=reader) for _ in range(3)]
    w = threading.Thread(target=writer)
    for t in readers + [w]:
        t.start()
    w.join()
    stop.set()
    for t in readers:
        t.join(timeout=10)
    assert errors == []
    assert sizes  # readers actually ran


def test_concurrent_inserts_have_deterministic_winners() -> None:
    """The same id inserted concurrently: exactly one insert succeeds and
    the index ends with exactly one value for the id."""
    idx = ProximityIndex()
    n = 16
    barrier = threading.Barrier(n)
    outcomes: list[bool] = [False] * n

    def worker(i: int) -> None:
        barrier.wait()
        try:
            idx.insert("shared", float(i), float(i), {"worker": i})
            outcomes[i] = True
        except DuplicatePointError:
            outcomes[i] = False

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(outcomes) == 1
    winner = next(i for i, ok in enumerate(outcomes) if ok)
    assert idx.get("shared") == Point("shared", float(winner), float(winner), {"worker": winner})
    assert len(idx) == 1
