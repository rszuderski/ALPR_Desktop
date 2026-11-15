"""Quad topology regression: nonzero shoelace area does not exclude a bowtie."""
import cv2
import numpy as np
import pytest
from itertools import permutations

from auto_annotation_tool.rectification.polygon_validator import PolygonValidator
from auto_annotation_tool.rectification.plate_rectifier import PlateRectifier
from auto_annotation_tool.rectification.quad_geometry import analyze_quad, canonicalize_polygon


RAW = [(0., 0.), (40., 0.), (50., 20.), (60., 130.)]
LEGACY = [RAW[0], RAW[1], RAW[3], RAW[2]]


def crossing(points):
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    def intersects(a, b, c, d):
        return cross(a, b, c)*cross(a, b, d) < 0 and cross(c, d, a)*cross(c, d, b) < 0
    return intersects(*points) or intersects(points[1], points[2], points[3], points[0])


def test_nonzero_area_bowtie_is_rejected():
    assert cv2.contourArea(np.array(RAW, np.float32)) == 3050
    assert cv2.contourArea(np.array(LEGACY, np.float32)) == 50
    assert not crossing(RAW) and crossing(LEGACY)
    assert not PolygonValidator.is_valid_quad(LEGACY)


def test_rectifier_does_not_create_bowtie_from_convex_points():
    result = PlateRectifier.order_quad_points(RAW).tolist()
    assert set(map(tuple, result)) == set(RAW)
    assert not crossing(result)


def test_quad_rejects_nonfinite_and_concave_points():
    assert not PolygonValidator.is_valid_quad([(0., 0.), (40., 0.), (50., float('nan')), (60., 130.)])
    assert not PolygonValidator.is_valid_quad([(0., 0.), (40., 0.), (10., 10.), (0., 40.)])


def test_reverse_cycle_is_valid_topology_not_bowtie_or_identity_certificate():
    points = [RAW[0], RAW[3], RAW[2], RAW[1]]
    result = analyze_quad(points)
    assert result.status == "REVERSED_WINDING"
    assert result.is_geometrically_valid and result.signed_area == -3050
    assert not result.intersecting_edges
    assert result.corner_identity_status == "CORNER_IDENTITY_AMBIGUOUS"
    assert points == [RAW[0], RAW[3], RAW[2], RAW[1]]


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_all_point_permutations_rotations_keep_locations_anchor_and_idempotence(angle):
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    points = np.round(np.array(RAW) @ rotation.T + 500, 10).tolist()
    for candidate in permutations(points):
        result = canonicalize_polygon(candidate, image_size=(1000, 1000))
        assert result.polygon is not None
        assert result.polygon[0] == tuple(candidate[0])
        assert [tuple(candidate[i]) for i in result.permutation] == result.polygon
        assert set(result.polygon) == set(map(tuple, candidate))
        assert not crossing(result.polygon)
        assert result.output_analysis.status == "VALID"
        second = canonicalize_polygon(result.polygon)
        assert second.polygon == result.polygon
        assert second.permutation == [0, 1, 2, 3] and not second.reordered


@pytest.mark.parametrize("points,status", [
    ([], "KEYPOINTS_MISSING"), (RAW[:3], "KEYPOINTS_MISSING"),
    (RAW + [(8, 9)], "KEYPOINTS_MISSING"),
    ([(0, 0), (40, 0), (40, 0), (0, 40)], "DEGENERATE"),
    ([(0, 0), (40, 0), (float('inf'), 10), (0, 40)], "DEGENERATE"),
    ([(0, 0), (40, 0), (float('nan'), 10), (0, 40)], "DEGENERATE"),
    ([(0, 0), (40, 0), (10, 10), (0, 40)], "CONCAVE"),
    ([(0, 0), (1, 0), (2, 0), (0, 1)], "DEGENERATE"),
    ([(0, 0), (100, 0), (100, 1e-10), (0, 1e-10)], "DEGENERATE"),
    ([('x', 0), (40, 0), (40, 10), (0, 10)], "DEGENERATE"),
])
def test_bad_point_set_never_turns_into_valid_pose(points, status):
    result = analyze_quad(points)
    assert result.status == status and not result.is_geometrically_valid
    assert canonicalize_polygon(points).polygon is None


def test_collinear_overlapping_nonadjacent_edges_are_intersection():
    result = analyze_quad([(0, 0), (4, 0), (1, 0), (3, 0)])
    assert result.status == "SELF_INTERSECTING"
    assert result.intersecting_edges
    assert canonicalize_polygon([(0, 0), (4, 0), (1, 0), (3, 0)]).polygon is None


@pytest.mark.parametrize("scale", [1e-5, 1, 1e5])
def test_scale_relative_tolerance_does_not_reject_small_true_quad(scale):
    points = [(scale*x, scale*y) for x, y in [(0, 0), (2, 0), (2, 1), (0, 1)]]
    assert analyze_quad(points).is_geometrically_valid
    assert PolygonValidator.is_valid_quad(points)


def test_bounds_are_optional_and_use_oriented_image_size():
    points = [(10, 100), (60, 100), (60, 140), (10, 140)]
    assert analyze_quad(points).status == "VALID"
    assert analyze_quad(points, (80, 160)).status == "VALID"
    assert analyze_quad(points, (160, 80)).status == "OUT_OF_BOUNDS"
    assert canonicalize_polygon(points, image_size=(160, 80)).polygon is None


def test_legacy_fix_is_positive_idempotent_but_does_not_certify_pose_indices():
    ordered = PolygonValidator.fix_polygon(LEGACY)
    assert analyze_quad(ordered).status == "VALID"
    assert ordered == PolygonValidator.fix_polygon(ordered)
    assert analyze_quad(ordered).corner_identity_status == "CORNER_IDENTITY_AMBIGUOUS"
