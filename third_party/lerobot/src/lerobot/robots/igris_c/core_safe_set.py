"""SafeSetPolygon reused from igris_core/safety/safe_set.py.

Source SHA256: 9a4bc6813bbd9bd60d9e211eaea1a50b8a91905274894452d7d922fdec4c69e4
Convex polygon validation is retained; local projection conditions endpoints
inside the polygon and scalar-limit intersection without moving unowned axes.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_DEFAULT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class SafeSetPolygon:
    """Validated vendor polygon in one coordinate space."""

    name: str
    group: str
    space: str
    x_axis: int
    y_axis: int
    vertices: tuple[tuple[float, float], ...]
    source_path: str
    margin_ratio: float

    def scaled(self, coefficient: float) -> "SafeSetPolygon":
        """Return a copy scaled about the vendor polygon centroid.

        The deployed controller defines HARD/SOFT as a radial contraction from
        the arithmetic-mean polygon centre.  Scaling from ``(0, 0)`` would
        change an asymmetric safe set (the waist and wrist sets are
        asymmetric) and would not reproduce that contract.
        """

        scale = float(coefficient)
        if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
            raise ValueError("safe-set coefficient must be finite and in (0, 1]")
        center_x = sum(x for x, _ in self.vertices) / len(self.vertices)
        center_y = sum(y for _, y in self.vertices) / len(self.vertices)
        return SafeSetPolygon(
            name=self.name,
            group=self.group,
            space=self.space,
            x_axis=self.x_axis,
            y_axis=self.y_axis,
            vertices=tuple(
                (
                    center_x + scale * (x - center_x),
                    center_y + scale * (y - center_y),
                )
                for x, y in self.vertices
            ),
            source_path=self.source_path,
            margin_ratio=self.margin_ratio,
        )

    def contains(self, point: Sequence[float], *, tolerance: float = _DEFAULT_TOLERANCE) -> bool:
        """Return whether a point is strictly inside this convex polygon."""

        try:
            x, y = (float(value) for value in point)
        except (TypeError, ValueError) as exc:
            raise ValueError("safe-set point must contain two finite values") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("safe-set point must contain two finite values")
        tol = float(tolerance)
        if not math.isfinite(tol) or tol < 0.0:
            raise ValueError("safe-set tolerance must be finite and non-negative")

        # Vertices are ordered around the convex polygon in the vendor
        # artifacts.  Use the signed edge cross products so clockwise input
        # remains valid if a future profile changes orientation.
        crosses = []
        for first, second in zip(self.vertices, self.vertices[1:] + self.vertices[:1], strict=False):
            edge_x = second[0] - first[0]
            edge_y = second[1] - first[1]
            crosses.append(edge_x * (y - first[1]) - edge_y * (x - first[0]))
        area2 = sum(
            first[0] * second[1] - first[1] * second[0]
            for first, second in zip(self.vertices, self.vertices[1:] + self.vertices[:1], strict=False)
        )
        orientation = 1.0 if area2 > 0.0 else -1.0
        # The vendor artifact policy is strict interior.  A point on the
        # boundary is rejected even when scalar position limits pass.
        return all(orientation * cross > tol for cross in crosses)

    def project(self, point, lower, upper, *, movable=(True, True)):
        """Nearest point in the polygon/box intersection, preserving fixed axes.

        The closed projection region is inset beyond ``contains``' strict
        cross-product threshold. Bounds are already inset by the caller.
        """
        point = np.asarray(point, dtype=float)
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        movable = np.asarray(movable, bool)
        vertices = np.asarray(self.vertices, float)
        following = np.roll(vertices, -1, axis=0)
        edges = following - vertices
        lengths = np.linalg.norm(edges, axis=1)
        orientation = np.sign(np.sum(vertices[:, 0] * following[:, 1] - vertices[:, 1] * following[:, 0]))
        if not orientation or np.any(lengths == 0):
            raise ValueError(f"degenerate {self.name} polygon")
        normals = orientation * np.column_stack((edges[:, 1], -edges[:, 0])) / lengths[:, None]
        # Extra nanoradian distance avoids accepting a rounded strict boundary.
        bounds = np.sum(normals * vertices, axis=1) - _DEFAULT_TOLERANCE / lengths - 1e-9
        matrix = np.vstack((normals, np.eye(2)[movable], -np.eye(2)[movable]))
        bounds = np.concatenate((bounds, upper[movable], -lower[movable]))
        if np.all(matrix @ point <= bounds):
            return point.copy()
        owned = np.flatnonzero(movable)
        if len(owned) == 1:
            axis = owned[0]
            fixed = 1 - axis
            lo, hi = lower[axis], upper[axis]
            for row, bound in zip(matrix, bounds, strict=True):
                remaining = bound - row[fixed] * point[fixed]
                if row[axis] > 0:
                    hi = min(hi, remaining / row[axis])
                elif row[axis] < 0:
                    lo = max(lo, remaining / row[axis])
                elif remaining < 0:
                    raise ValueError(f"{self.name} has no feasible point with fixed unowned axis")
            if lo > hi:
                raise ValueError(f"{self.name} has no feasible point with fixed unowned axis")
            result = point.copy()
            result[axis] = np.clip(point[axis], lo, hi)
            return result
        if len(owned) != 2:
            raise ValueError(f"{self.name} projection requires an owned axis")
        candidates = []
        # A nearest point in a bounded convex polygon is on an edge or vertex.
        for i, row in enumerate(matrix):
            candidate = point - row * ((row @ point - bounds[i]) / (row @ row))
            if np.all(matrix @ candidate <= bounds + 1e-12):
                candidates.append(candidate)
            for j in range(i):
                pair = matrix[[i, j]]
                if abs(np.linalg.det(pair)) < 1e-12:
                    continue
                candidate = np.linalg.solve(pair, bounds[[i, j]])
                if np.all(matrix @ candidate <= bounds + 1e-12):
                    candidates.append(candidate)
        if not candidates:
            raise ValueError(f"{self.name} polygon and scalar limits have no feasible intersection")
        candidates = np.asarray(candidates)
        return candidates[np.argmin(np.sum((candidates - point) ** 2, axis=1))]
