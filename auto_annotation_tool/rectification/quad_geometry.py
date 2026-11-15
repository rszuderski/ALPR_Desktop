"""Pure quad topology, separate from the learned identity of Pose keypoints.

Image XY uses Y down: positive signed area means a clockwise display cycle.
No routine here certifies the learned meanings of kpt[0..3].  Tolerances are
relative to the local span, not an arbitrary ten-pixel area cutoff: tiny but
well-shaped quads remain valid, while nearly collapsed ones fail closed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import hypot, isfinite
from typing import Any


POLICY_VERSION = "alpr.quad_geometry.v1"
AMBIGUOUS = "CORNER_IDENTITY_AMBIGUOUS"
SCREEN_ORDER = "SCREEN_ORDER_COMPATIBLE"


@dataclass(frozen=True)
class QuadAnalysis:
    status: str
    is_geometrically_valid: bool = False
    signed_area: float = 0.0
    area: float = 0.0
    winding: str = "UNDEFINED"
    corner_identity_status: str = AMBIGUOUS
    intersecting_edges: tuple[tuple[int, int], ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalQuad:
    polygon: list[tuple[float, float]] | None
    permutation: list[int] | None
    input_analysis: QuadAnalysis
    output_analysis: QuadAnalysis | None
    reordered: bool
    corner_identity_status: str = AMBIGUOUS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _points(points):
    try:
        if points is None or len(points) != 4:
            return None, "KEYPOINTS_MISSING", "Exactly four XY pairs are required"
        result = []
        for point in points:
            if len(point) != 2 or any(isinstance(v, (bool, str, bytes)) for v in point):
                return None, "DEGENERATE", "Malformed XY pair"
            x, y = float(point[0]), float(point[1])
            if not isfinite(x) or not isfinite(y):
                return None, "DEGENERATE", "Non-finite coordinate"
            result.append((x, y))
        return result, "", ""
    except (TypeError, ValueError, OverflowError):
        return None, "DEGENERATE", "Malformed XY pair"


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _tolerances(points):
    span = max(max(p[0] for p in points) - min(p[0] for p in points),
               max(p[1] for p in points) - min(p[1] for p in points))
    return max(1e-9, span * 1e-7), max(1e-18, span * span * 1e-9)


def _segments_intersect(a, b, c, d, length_epsilon, cross_epsilon):
    values = (_cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b))
    def sign(value):
        return 1 if value > cross_epsilon else -1 if value < -cross_epsilon else 0
    signs = list(map(sign, values))
    if signs[0] * signs[1] < 0 and signs[2] * signs[3] < 0:
        return True
    def on_segment(p, q, r):
        return (min(p[0], q[0]) - length_epsilon <= r[0] <= max(p[0], q[0]) + length_epsilon
                and min(p[1], q[1]) - length_epsilon <= r[1] <= max(p[1], q[1]) + length_epsilon)
    return ((signs[0] == 0 and on_segment(a, b, c))
            or (signs[1] == 0 and on_segment(a, b, d))
            or (signs[2] == 0 and on_segment(c, d, a))
            or (signs[3] == 0 and on_segment(c, d, b)))


def _screen_order_compatible(points, epsilon):
    """Conservative legacy export compatibility, never a Pose identity proof."""
    a, b, c, d = points
    width = (hypot(b[0]-a[0], b[1]-a[1]) + hypot(c[0]-d[0], c[1]-d[1])) / 2
    height = (hypot(d[0]-a[0], d[1]-a[1]) + hypot(c[0]-b[0], c[1]-b[1])) / 2
    return (max(a[1], b[1]) + epsilon < min(c[1], d[1])
            and max(a[0], d[0]) + epsilon < min(b[0], c[0])
            and width > height + epsilon)


def analyze_quad(points, image_size=None) -> QuadAnalysis:
    """Inspect the supplied order without sorting, clipping or moving points.

    Reversed winding is valid topology.  Optional image bounds are strict and
    inclusive [0,width] x [0,height], matching continuous detector coordinates.
    """
    pts, status, reason = _points(points)
    if pts is None:
        return QuadAnalysis(status, reason=reason)
    length_eps, cross_eps = _tolerances(pts)
    if any(hypot(pts[i][0]-pts[j][0], pts[i][1]-pts[j][1]) <= length_eps
           for i in range(4) for j in range(i+1, 4)):
        return QuadAnalysis("DEGENERATE", reason="Duplicate or indistinguishable points")
    # Translating the shoelace origin avoids cancellation for large world XY.
    signed = (_cross(pts[0], pts[1], pts[2]) + _cross(pts[0], pts[2], pts[3])) / 2
    if not isfinite(signed):
        return QuadAnalysis("DEGENERATE", reason="Coordinate span is not numerically representable")
    winding = "CLOCKWISE_IMAGE_XY" if signed > cross_eps else "COUNTERCLOCKWISE_IMAGE_XY" if signed < -cross_eps else "UNDEFINED"
    common = dict(signed_area=signed, area=abs(signed), winding=winding)
    intersections = []
    for i, j in ((0, 2), (1, 3)):
        if _segments_intersect(pts[i], pts[(i+1) % 4], pts[j], pts[(j+1) % 4], length_eps, cross_eps):
            intersections.append((i, j))
    if intersections:
        return QuadAnalysis("SELF_INTERSECTING", intersecting_edges=tuple(intersections), **common)
    crosses = [_cross(pts[i], pts[(i+1) % 4], pts[(i+2) % 4]) for i in range(4)]
    if abs(signed) <= cross_eps or any(abs(value) <= cross_eps for value in crosses):
        return QuadAnalysis("DEGENERATE", reason="Collinear or nearly collapsed quad", **common)
    if not (all(value > cross_eps for value in crosses) or all(value < -cross_eps for value in crosses)):
        return QuadAnalysis("CONCAVE", **common)
    if image_size is not None:
        try:
            width, height = map(float, image_size)
            if not all(isfinite(v) and v > 0 for v in (width, height)):
                raise ValueError("Image size must be positive and finite")
        except (TypeError, ValueError, OverflowError):
            return QuadAnalysis("OUT_OF_BOUNDS", reason="Invalid oriented image dimensions", **common)
        if any(not (0 <= x <= width and 0 <= y <= height) for x, y in pts):
            return QuadAnalysis("OUT_OF_BOUNDS", reason="Point outside oriented image", **common)
    status = "VALID" if signed > 0 else "REVERSED_WINDING"
    identity = SCREEN_ORDER if signed > 0 and _screen_order_compatible(pts, length_eps) else AMBIGUOUS
    return QuadAnalysis(status, True, corner_identity_status=identity, **common)


def canonicalize_polygon(points, anchor_index=0, image_size=None) -> CanonicalQuad:
    """Build a positive hull cycle anchored at one unchanged input point.

    A four-vertex convex hull is required.  A missing/inner/duplicate point or
    out-of-image point cannot be repaired by reindexing.  The returned mapping
    is display_index -> input_index; it does not assign semantic TL/TR names.
    """
    analysis = analyze_quad(points, image_size)
    pts, _, _ = _points(points)
    empty = CanonicalQuad(None, None, analysis, None, False)
    if pts is None or analysis.status in {"KEYPOINTS_MISSING", "DEGENERATE", "OUT_OF_BOUNDS", "CONCAVE"}:
        return empty
    if isinstance(anchor_index, bool) or not isinstance(anchor_index, int) or not 0 <= anchor_index < 4:
        raise ValueError("anchor_index must identify one of the four original points")
    _, cross_eps = _tolerances(pts)
    sorted_indices = sorted(range(4), key=lambda i: pts[i])
    def half(indices):
        result = []
        for index in indices:
            while len(result) >= 2 and _cross(pts[result[-2]], pts[result[-1]], pts[index]) <= cross_eps:
                result.pop()
            result.append(index)
        return result
    order = half(sorted_indices)[:-1] + half(list(reversed(sorted_indices)))[:-1]
    if len(order) != 4:
        return empty
    start = order.index(anchor_index)
    order = order[start:] + order[:start]
    polygon = [pts[i] for i in order]
    output = analyze_quad(polygon, image_size)
    if not output.is_geometrically_valid or output.signed_area <= 0:
        return empty
    reordered = order != list(range(4))
    return CanonicalQuad(polygon, order, analysis, output, reordered,
                         AMBIGUOUS if reordered else output.corner_identity_status)


def validated_pose_label_points(points, image_size=None, *, corner_identity_confirmed=False):
    """Validate approved label indexing; never silently reorder or use bbox.

    Legacy upright TL/TR/BR/BL sequences remain compatible.  Perspective or
    rotated sequences require an explicit new operator identity confirmation.
    Confirmation does not waive invalid geometry or reverse-winding checks.
    """
    analysis = analyze_quad(points, image_size)
    if analysis.status != "VALID":
        raise ValueError(f"Nieprawidłowa etykieta Pose: {analysis.status} ({analysis.reason})")
    if not corner_identity_confirmed and analysis.corner_identity_status != SCREEN_ORDER:
        raise ValueError("CORNER_IDENTITY_AMBIGUOUS: wymagane jawne potwierdzenie indeksów narożników")
    return _points(points)[0]


def pose_label_identity_confirmation(attributes=None, geometry=None) -> bool:
    """Read only the explicit new operator flag, never generic GT approval."""
    attrs = attributes or {}
    confirmed = str(attrs.get("pose_corner_identity", "")).strip().lower() == "confirmed"
    if geometry is None and attrs.get("alpr_quad_geometry"):
        try:
            geometry = json.loads(attrs["alpr_quad_geometry"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Nieprawidłowe metadane geometrii Pose") from exc
    if geometry is not None:
        if not isinstance(geometry, dict) or geometry.get("schema") != "alpr.pose_geometry.v1":
            raise ValueError("Nieznany kontrakt geometrii Pose")
        if geometry.get("provenance_error"):
            raise ValueError("Nieprawidłowe pochodzenie geometrii Pose")
        if not confirmed and any(geometry.get(key) for key in ("bbox_fallback", "reordered", "manual_reordered")):
            raise ValueError("CORNER_IDENTITY_AMBIGUOUS: obrys zmieniono lub użyto bbox; wymagany odbiór indeksów")
    return confirmed


def format_pose_label(points, image_size, *, class_id=0) -> str:
    """Serialize already-approved points, validating six-decimal precision too."""
    width, height = map(float, image_size)
    normalized = [(float(f"{x/width:.6f}"), float(f"{y/height:.6f}")) for x, y in points]
    result = analyze_quad(normalized, (1, 1))
    if result.status != "VALID":
        raise ValueError(f"Etykieta Pose po zaokrągleniu: {result.status}")
    xs, ys = zip(*points)
    box = [((min(xs)+max(xs))/2)/width, ((min(ys)+max(ys))/2)/height,
           (max(xs)-min(xs))/width, (max(ys)-min(ys))/height]
    if any(float(f"{v:.6f}") <= 0 for v in box[2:]):
        raise ValueError("DEGENERATE: bbox Pose zanika po zaokrągleniu")
    return str(class_id) + " " + " ".join(f"{value:.6f}" for value in box + [v for p in normalized for v in p])
