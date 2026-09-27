"""Spherical-Earth geometry helpers.

Distance convention (documented for callers and pinned by tests):

* The Earth is modelled as a sphere of radius :data:`EARTH_RADIUS_M`
  = 6 371 008.8 metres, the IUGG *mean* Earth radius
  ``(2a + b) / 3`` evaluated with the WGS-84 axes
  ``a = 6_378_137.0`` m and ``b = 6_356_752.3142`` m
  (rounded value 6 371 008.8 m).
* Distances are **great-circle distances** computed with the haversine
  formula, in metres, and therefore lie in ``[0, HALF_CIRCUMFERENCE_M]``.
* A radius query returns points with ``distance <= radius`` (the boundary
  is inclusive).
* Longitudes are interpreted modulo 360 degrees and normalized into
  ``[-180, 180)`` on input; the geometry itself works on unwrapped
  longitudes so queries crossing the antimeridian (±180°) are handled
  without seams.
"""

from __future__ import annotations

import math

EARTH_RADIUS_M: float = 6_371_008.8
"""Mean Earth radius used for all distance computations (metres)."""

HALF_CIRCUMFERENCE_M: float = math.pi * EARTH_RADIUS_M
"""Antipodal distance (~20 037 508 m): the maximum possible distance."""

LatLng = tuple[float, float]
"""A ``(latitude, longitude)`` pair in degrees."""


def normalize_latitude(value: float) -> float:
    """Validate ``value`` as a latitude and return it as a float.

    Latitude is **not** wrapped: values outside ``[-90, 90]`` are rejected.
    NaN and infinities are rejected.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("latitude must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("latitude must be finite")
    if not -90.0 <= value <= 90.0:
        raise ValueError(f"latitude must be in [-90, 90], got {value}")
    return value


def normalize_longitude(value: float) -> float:
    """Normalize a longitude into the half-open range ``[-180, 180)``.

    NaN/infinities are rejected.  Integers and floats are accepted.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("longitude must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("longitude must be finite")
    lon = (value + 180.0) % 360.0 - 180.0
    # ``%`` yields -0.0 for -180.0; canonicalize the pole edge to -180.0.
    if lon == -180.0:
        return -180.0
    return lon


def validate_latlng(latitude: float, longitude: float) -> LatLng:
    """Validate a coordinate pair, returning normalized floats.

    Raises:
        ValueError (often wrapped as InvalidCoordinatesError by the
            service layer) for missing/NaN/out-of-range coordinates.
    """
    try:
        lat = normalize_latitude(latitude)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid latitude: {exc}") from None
    try:
        lon = normalize_longitude(longitude)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid longitude: {exc}") from None
    return lat, lon


def haversine_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Great-circle distance between two coordinates, in metres.

    Inputs are validated/normalized.  The result uses
    :data:`EARTH_RADIUS_M`; identical (or antipodally equivalent) points
    return ``0.0``.
    """
    lat1, lon1 = validate_latlng(lat1, lon1)
    lat2, lon2 = validate_latlng(lat2, lon2)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    # Longitude difference wrapped to [-180, 180] so ±180 is seamless.
    dlambda = math.radians((lon2 - lon1 + 180.0) % 360.0 - 180.0)
    sin_dphi = math.sin(dphi / 2.0)
    sin_dlambda = math.sin(dlambda / 2.0)
    h = sin_dphi * sin_dphi + math.cos(phi1) * math.cos(phi2) * sin_dlambda * sin_dlambda
    # Clamp to [0, 1] against floating-point overshoot (antipodal points).
    h = min(1.0, max(0.0, h))
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def spherical_cap_bbox(
    latitude: float,
    longitude: float,
    radius_m: float,
) -> tuple[float, float, float, float]:
    """Bounding box of a spherical cap.

    Returns ``(min_lat, max_lat, min_lon, max_lon)`` where the longitude
    bounds are **unwrapped** relative to ``longitude`` (they may lie outside
    ``[-180, 180]``).  When the cap contains a pole the latitude extent is
    clamped to ``[-90, 90]`` and the longitude extent is marked as global
    (``longitude +/- 360``); callers use :func:`longitude_extent_at_lat`
    for per-latitude precision instead of relying on this box alone.

    Raises:
        ValueError for invalid coordinates or a negative radius.
    """
    lat, lon = validate_latlng(latitude, longitude)
    if isinstance(radius_m, bool) or not isinstance(radius_m, (int, float)):
        raise TypeError("radius_m must be a number")
    radius_m = float(radius_m)
    if not math.isfinite(radius_m) or radius_m < 0.0:
        raise ValueError("radius_m must be a finite, non-negative number")

    angular = radius_m / EARTH_RADIUS_M  # central angle in radians
    min_lat = max(-90.0, lat - math.degrees(angular))
    max_lat = min(90.0, lat + math.degrees(angular))

    # Does the cap contain either pole?  angular >= pi/2 -/+ phi0.
    phi0 = math.radians(lat)
    contains_north = angular >= (math.pi / 2.0 - phi0) - 1e-12
    contains_south = angular >= (math.pi / 2.0 + phi0) - 1e-12
    if contains_north and contains_south:
        # Covers both poles: every parallel is wholly inside.
        return -90.0, 90.0, lon - 360.0, lon + 360.0

    # Maximum absolute longitude deviation at the cap's own latitude.
    # cos(c) = sin(phi0)^2 + cos(phi0)^2 cos(dlambda)
    cos_phi0 = math.cos(phi0)
    denom = cos_phi0 * cos_phi0
    if abs(denom) < 1e-28:
        # Cap centered exactly on a pole: the bbox longitude is irrelevant
        # (the grid index scans full rings from the pole containment test).
        dlon = 180.0
    else:
        numer = math.cos(angular) - math.sin(phi0) ** 2
        ratio = numer / denom
        ratio = min(1.0, max(-1.0, ratio))
        dlon = math.degrees(math.acos(ratio))
    return min_lat, max_lat, lon - dlon, lon + dlon


def unwrap_longitude_relative(lon: float, reference: float) -> float:
    """Shift normalized ``lon`` by whole 360° turns nearest to ``reference``."""
    return lon - round((lon - reference) / 360.0) * 360.0


def longitude_extent_at_lat(
    cap_lat: float,
    cap_lon: float,
    radius_m: float,
    latitude: float,
) -> float | None:
    """Half-width in degrees of the cap at a given latitude.

    Returns the non-negative value ``d`` such that the cap intersects the
    parallel ``latitude`` exactly at longitudes
    ``[cap_lon - d, cap_lon + d]`` (unwrapped).  Returns ``math.inf`` when
    the whole parallel is inside the cap (polar coverage), and ``None``
    when the parallel does not intersect the cap at all.

    Derivation: for central angle ``c`` and center latitude ``phi0``,
    ``cos(c) = sin(phi0) sin(phi) + cos(phi0) cos(phi) cos(dlambda)``.
    """
    c = radius_m / EARTH_RADIUS_M
    phi0 = math.radians(cap_lat)
    phi = math.radians(latitude)
    cos_c = math.cos(c)
    a = math.cos(phi0) * math.cos(phi)
    b = math.sin(phi0) * math.sin(phi)
    if abs(math.cos(phi)) < 1e-14:
        # The queried parallel is a pole: compare its exact angular
        # distance to the cap centre (|pi/2 - phi0| for the north pole,
        # |-pi/2 - phi0| for the south pole) against the cap angle.
        pole = math.pi / 2.0 if latitude >= 0.0 else -math.pi / 2.0
        if abs(pole - phi0) <= c + 1e-12:
            return math.inf
        return None
    ratio = (cos_c - b) / a
    # ``cos(d) = b + a cos(dlambda)``; the cap contains the points of the
    # parallel whose cosine distance is >= cos(c):
    #   ratio >  1 -> no point on the parallel is inside;
    #   ratio < -1 -> the *whole* parallel (a polar ring) is inside;
    #   in between the intersection is [-acos(ratio), +acos(ratio)].
    # A tiny tolerance at ratio == -1 keeps the ring-boundary parallel at
    # exactly 180° instead of flipping to a full ring on round-off.
    if ratio > 1.0:
        return None
    if ratio < -1.0 - 1e-12:
        return math.inf
    return math.degrees(math.acos(min(1.0, max(-1.0, ratio))))


def max_longitude_extent(
    cap_lat: float,
    radius_m: float,
    lat_lo: float,
    lat_hi: float,
) -> float | None:
    """Maximum finite cap half-width (degrees) over a latitude interval.

    The cross-section half-width ``h(phi)`` is NOT generally maximal at the
    cap latitude; its only interior extremum satisfies
    ``sin(phi*) = sin(phi0) / cos(c)``.  The maximum over a closed interval
    is therefore attained at an interval endpoint or at ``phi*``.

    Returns ``None`` if no parallel in the interval intersects the cap
    (outside its polar full-ring region).  ``inf`` is never returned:
    fully covered parallels are handled separately by callers via
    :func:`ring_threshold_latitudes`.
    """
    c = radius_m / EARTH_RADIUS_M
    phi0 = math.radians(cap_lat)
    cos_c = math.cos(c)
    sin_phi0 = math.sin(phi0)

    candidates = [math.radians(lat_lo), math.radians(lat_hi)]
    if abs(cos_c) > 1e-15:
        s_star = sin_phi0 / cos_c
        if -1.0 <= s_star <= 1.0:
            phi_star = math.asin(s_star)
            if math.radians(lat_lo) <= phi_star <= math.radians(lat_hi):
                candidates.append(phi_star)
    # The finite section meets the polar full-ring section at the ring
    # threshold parallels, where the finite half-width is exactly 180°;
    # note them separately so floating-point noise cannot turn that exact
    # 180° into a full-ring classification.
    north_thr, south_thr = ring_threshold_latitudes(cap_lat, radius_m)
    threshold_hits = [
        thr for thr in (north_thr, south_thr)
        if thr is not None and lat_lo - 1e-12 <= thr <= lat_hi + 1e-12
    ]
    if threshold_hits:
        return 180.0

    best: float | None = None
    for phi in candidates:
        a = math.cos(phi0) * math.cos(phi)
        b = sin_phi0 * math.sin(phi)
        if abs(a) < 1e-14:
            continue  # pole parallel: no finite (point) extent
        ratio = (cos_c - b) / a
        if ratio > 1.0:
            continue  # parallel does not intersect the cap
        if ratio < -1.0:
            continue  # whole parallel covered (polar ring), not finite
        h = math.degrees(math.acos(min(1.0, max(-1.0, ratio))))
        if best is None or h > best:
            best = h
    return best


def ring_threshold_latitudes(
    cap_lat: float,
    radius_m: float,
) -> tuple[float | None, float | None]:
    """Latitudes poleward of which parallels are wholly inside the cap.

    Returns ``(north_threshold, south_threshold)`` in degrees, or ``None``
    when the cap does not contain that pole.  A parallel at latitude
    ``phi`` is a full longitude ring iff its farthest point (antipodal
    longitude, ``dlambda = pi``) is within the cap angle ``c``:

    * north side: ``pi - (phi - phi0) <= c`` -> ``phi >= phi0 + pi - c``;
    * south side: ``pi - (phi0 - phi) <= c`` -> ``phi <= phi0 - pi + c``.
    """
    c = radius_m / EARTH_RADIUS_M
    phi0 = math.radians(cap_lat)
    # A parallel's farthest point from the cap centre is the one on the
    # antipodal longitude (dlambda = pi).  It is inside iff
    # ``pi - |phi - phi0| <= c``:
    #   north parallel (phi >= phi0): phi >= pi - c + phi0 ... with the
    #     sign of the far-side longitude giving phi >= pi - c - phi0;
    #   south parallel: phi <= c - pi - phi0.
    north = math.degrees(math.pi - c - phi0)
    south = math.degrees(c - math.pi - phi0)
    return (
        north if north <= 90.0 + 1e-12 else None,
        south if south >= -90.0 - 1e-12 else None,
    )
