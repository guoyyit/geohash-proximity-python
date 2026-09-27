"""Concurrency: queries see one consistent snapshot during writes."""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from geo_proximity import OnDuplicate, ProximityService, ServiceConfig
from geo_proximity.geo import HALF_CIRCUMFERENCE_M

TOKYO = (35.68, 139.69)


def _populate(svc: ProximityService, n: int, seed: int = 0) -> None:
    # Dense cluster around Tokyo plus scattered points; deterministic.
    pts = []
    for i in range(n):
        lat = TOKYO[0] + ((i * 7919) % 1000) / 10000.0 - 0.05
        lon = TOKYO[1] + ((i * 6271) % 1000) / 10000.0 - 0.05
        pts.append((f"p{i:06d}", lat, lon, {"i": i}))
    svc.bulk_load(pts, on_duplicate=OnDuplicate.REPLACE)


def test_concurrent_radius_queries_are_consistent() -> None:
    svc = ProximityService(ServiceConfig(index_levels=14))
    _populate(svc, 3000)
    errors: list[BaseException] = []
    stop = threading.Event()

    def reader() -> None:
        try:
            while not stop.is_set():
                res = svc.radius_query(TOKYO[0], TOKYO[1], 5000.0)
                ids = [s.point.id for s in res]
                # Snapshot invariants: unique ids, sorted by (dist, id),
                # every reported distance is genuinely within the radius.
                assert len(ids) == len(set(ids))
                dists = [(s.distance_m, s.point.id) for s in res]
                assert dists == sorted(dists)
                assert all(s.distance_m <= 5000.0 + 1e-6 for s in res)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def writer() -> None:
        # Continuously move an existing point across cell boundaries.
        i = 0
        while not stop.is_set():
            pid = f"p{(i * 37) % 3000:06d}"
            lat = TOKYO[0] + 0.04 * math.sin(i / 7.0)
            lon = TOKYO[1] + 0.04 * math.cos(i / 5.0)
            svc.update(pid, latitude=lat, longitude=lon)
            i += 1

    with ThreadPoolExecutor(max_workers=8) as pool:
        readers = [pool.submit(reader) for _ in range(6)]
        w = pool.submit(writer)
        threading.Event().wait(2.0)
        stop.set()
        w.result()
        for f in readers:
            f.result()
    assert errors == []


def test_concurrent_writes_preserve_store_and_index() -> None:
    cfg = ServiceConfig(max_points=20_000, index_levels=12)
    svc = ProximityService(cfg)

    def worker(worker_id: int) -> int:
        count = 0
        for j in range(800):
            pid = f"w{worker_id}-{j}"
            svc.upsert((pid,
                        TOKYO[0] + (j % 50) * 0.001,
                        TOKYO[1] + (j % 50) * 0.001))
            count += 1
            if j % 3 == 0:
                svc.update(pid, latitude=TOKYO[0], longitude=TOKYO[1])
            if j % 5 == 0 and svc.delete(pid):
                count -= 1
            # Point ids are unique per worker => never a duplicate error.
        return count

    with ThreadPoolExecutor(max_workers=8) as pool:
        totals = [f.result() for f in
                  as_completed([pool.submit(worker, w) for w in range(8)])]
    # Every surviving point must be queryable exactly once.
    res = svc.radius_query(TOKYO[0], TOKYO[1], HALF_CIRCUMFERENCE_M)
    ids = [s.point.id for s in res]
    assert len(ids) == len(set(ids))
    assert len(ids) == len(svc) == sum(totals)
    # A nearest query against the mutating store returns a valid ordered
    # snapshot too.
    near = svc.nearest(TOKYO[0], TOKYO[1], 100)
    assert len(near) == 100
    assert [s.distance_m for s in near] == sorted(
        s.distance_m for s in near
    )


def test_duplicate_id_insert_under_contention_is_defined() -> None:
    svc = ProximityService(ServiceConfig(max_points=100_000))

    def insert_copy() -> tuple[bool, bool]:
        raised = False
        stored = False
        try:
            stored = svc.insert(("shared", 0.0, 0.0))
        except Exception:  # noqa: BLE001
            raised = True
        return raised, stored

    with ThreadPoolExecutor(max_workers=32) as pool:
        outcomes = [f.result() for f in
                    as_completed([pool.submit(insert_copy)
                                  for _ in range(64)])]
    # Exactly one insert wins; all others either errored (default policy)
    # -- never silently duplicated.
    stored = [o for o in outcomes if o[1]]
    assert len(stored) == 1
    assert len(svc) == 1


def test_snapshot_during_bulk_is_all_or_nothing() -> None:
    svc = ProximityService()
    _populate(svc, 500)
    seen_sizes: set[int] = set()
    stop = threading.Event()
    started = threading.Event()

    def reader() -> None:
        started.set()
        while not stop.is_set():
            seen_sizes.add(len(svc))

    def bulk() -> None:
        # Wait until the reader is actively sampling, then apply the batch
        # under one write lock.
        started.wait(2.0)
        threading.Event().wait(0.01)
        svc.bulk_load([(f"b{i}", TOKYO[0], TOKYO[1]) for i in range(500)],
                      on_duplicate=OnDuplicate.ERROR)

    t_read = threading.Thread(target=reader)
    t_bulk = threading.Thread(target=bulk)
    t_read.start()
    t_bulk.start()
    t_bulk.join()
    stop.set()
    t_read.join()
    # Readers may observe pre/post sizes, but never an intermediate one.
    assert seen_sizes <= {500, 1000}
    assert len(svc) == 1000
