from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from common_types import FloatArray, PointsN3


class CADModel:
    """
    Queryable CAD surface for calibration.

    Responsibilities:
    - Load a mesh
    - Sample surface points
    - Provide closest-point queries
    - Provide smooth surface normals
    - Provide local surface variation / curvature proxies
    """

    def __init__(
        self,
        obj_path: str | Path,
        *,
        n_surface_samples: int = 50_000,
        validate: bool = True,
    ) -> None:
        self.obj_path = Path(obj_path)
        self.n_surface_samples = n_surface_samples
        self.mesh = self._load_mesh()

        if validate:
            self._validate_mesh()

        self._rebuild_surface_representation()

    @classmethod
    def from_config(
        cls,
        cfg: str | Path | Mapping[str, object],
    ) -> "CADModel":
        """
        Accept either:
        - a path string / Path
        - a cad_model config dict, e.g. {"model_path": "...", "n_surface_samples": 50000}
        """
        if isinstance(cfg, (str, Path)):
            return cls(obj_path=cfg)

        return cls(
            obj_path=str(cfg["model_path"]),
            n_surface_samples=int(cfg.get("n_surface_samples", 50_000)),
            validate=bool(cfg.get("validate", True)),
        )

    def _load_mesh(self) -> o3d.geometry.TriangleMesh:
        if not self.obj_path.exists():
            raise FileNotFoundError(f"CAD model not found: {self.obj_path}")

        mesh = o3d.io.read_triangle_mesh(str(self.obj_path))
        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            raise ValueError(f"Failed to load valid mesh from {self.obj_path}")

        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        self._ensure_outward_normals(mesh)
        return mesh

    def _ensure_outward_normals(self, mesh: o3d.geometry.TriangleMesh) -> None:
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        tris = np.asarray(mesh.triangles, dtype=np.int32)
        tri_normals = np.asarray(mesh.triangle_normals, dtype=np.float64)

        centroid = verts.mean(axis=0)
        tri_centroids = (verts[tris[:, 0]] + verts[tris[:, 1]] + verts[tris[:, 2]]) / 3.0
        outward = tri_centroids - centroid
        score = float(np.mean(np.sum(outward * tri_normals, axis=1)))

        if score < 0.0:
            flipped = tris[:, ::-1].copy()
            mesh.triangles = o3d.utility.Vector3iVector(flipped)
            mesh.compute_triangle_normals()
            mesh.compute_vertex_normals()

    def _validate_mesh(self) -> None:
        bbox = self.mesh.get_axis_aligned_bounding_box()
        dims = bbox.max_bound - bbox.min_bound
        max_dim = float(np.max(dims))

        if max_dim < 0.05:
            print(f"[CADModel] Warning: mesh seems very small: max dim = {max_dim:.3f} m")
        if max_dim > 20.0:
            print(f"[CADModel] Warning: mesh seems very large: max dim = {max_dim:.3f} m")
        if not self.mesh.is_watertight():
            print("[CADModel] Warning: mesh is not watertight")

    def _rebuild_surface_representation(self) -> None:
        self.surface_points = self.mesh.sample_points_uniformly(
            number_of_points=self.n_surface_samples
        )

        if not self.surface_points.has_normals():
            self.surface_points.estimate_normals()
            self.surface_points.orient_normals_consistent_tangent_plane(50)

        surface_np = np.asarray(self.surface_points.points, dtype=np.float64)
        self.surface_kdtree = cKDTree(surface_np)

        self.raycasting_scene = o3d.t.geometry.RaycastingScene()
        self.raycasting_scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        )

    def get_surface_area(self) -> float:
        return float(self.mesh.get_surface_area())

    def sample_surface_centers(
        self,
        n_centers: int,
        method: Literal["poisson_disk", "uniform", "fps"] = "poisson_disk",
        fps_pool_size: int = 200_000,
        seed: int = 42,
    ) -> PointsN3:
        if method == "poisson_disk":
            pcd = self.mesh.sample_points_poisson_disk(
                number_of_points=n_centers,
                init_factor=5,
            )
            return np.asarray(pcd.points, dtype=np.float64)

        if method == "uniform":
            pcd = self.mesh.sample_points_uniformly(number_of_points=n_centers)
            return np.asarray(pcd.points, dtype=np.float64)

        if method == "fps":
            pool_size = max(n_centers, fps_pool_size)
            pcd = self.mesh.sample_points_uniformly(number_of_points=pool_size)
            pts = np.asarray(pcd.points, dtype=np.float64)
            return farthest_point_sampling(pts, n_centers, seed=seed)

        raise ValueError(f"Unknown sampling method: {method}")

    def project_points_to_surface_exact(
        self,
        points: PointsN3,
    ) -> tuple[PointsN3, FloatArray]:
        points_np = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(points_np) == 0:
            return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)

        query = o3d.core.Tensor(points_np, dtype=o3d.core.Dtype.Float32)
        result = self.raycasting_scene.compute_closest_points(query)

        closest = result["points"].numpy().astype(np.float64)
        distances = np.linalg.norm(points_np.astype(np.float64) - closest, axis=1)
        return closest, distances

    def compute_point_to_surface_distances(
        self,
        points: PointsN3,
        method: Literal["kdtree", "exact"] = "kdtree",
    ) -> tuple[FloatArray, PointsN3]:
        points_np = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(points_np) == 0:
            return np.empty((0,), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

        if method == "exact":
            closest, distances = self.project_points_to_surface_exact(points_np)
            return distances, closest

        distances, idx = self.surface_kdtree.query(points_np)
        closest = np.asarray(self.surface_points.points, dtype=np.float64)[idx]
        return np.asarray(distances, dtype=np.float64), closest

    def get_surface_normals_at_points(self, points: PointsN3) -> PointsN3:
        points_np = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(points_np) == 0:
            return np.empty((0, 3), dtype=np.float64)

        if not self.mesh.has_vertex_normals():
            self.mesh.compute_vertex_normals()

        verts = np.asarray(self.mesh.vertices, dtype=np.float64)
        tris = np.asarray(self.mesh.triangles, dtype=np.int32)
        vnorms = np.asarray(self.mesh.vertex_normals, dtype=np.float64)

        query = o3d.core.Tensor(points_np, dtype=o3d.core.Dtype.Float32)
        result = self.raycasting_scene.compute_closest_points(query)
        tri_ids = result["primitive_ids"].numpy()
        closest = result["points"].numpy().astype(np.float64)

        normals = np.zeros((len(points_np), 3), dtype=np.float64)
        for i, (tri_id, p) in enumerate(zip(tri_ids, closest)):
            tri = tris[int(tri_id)]
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            bary = _barycentric_coords(p, v0, v1, v2)

            n = (
                bary[0] * vnorms[tri[0]]
                + bary[1] * vnorms[tri[1]]
                + bary[2] * vnorms[tri[2]]
            )
            normals[i] = n / (np.linalg.norm(n) + 1e-12)

        return normals

    def compute_surface_variation_at_points(
        self,
        points: PointsN3,
        k_neighbors: int = 12,
    ) -> FloatArray:
        points_np = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        surface_pts = np.asarray(self.surface_points.points, dtype=np.float64)

        curvatures = np.zeros(len(points_np), dtype=np.float64)
        for i, point in enumerate(points_np):
            _, idx = self.surface_kdtree.query(point, k=k_neighbors)
            nbrs = surface_pts[np.atleast_1d(idx)]
            diff = nbrs - point
            C = (diff.T @ diff) / max(len(nbrs), 1)
            eigvals = np.linalg.eigvalsh(C)
            curvatures[i] = eigvals[0] / (eigvals.sum() + 1e-12)

        return curvatures

    def apply_transformation(self, T: FloatArray) -> None:
        self.mesh.transform(np.asarray(T, dtype=np.float64))
        self.mesh.compute_triangle_normals()
        self.mesh.compute_vertex_normals()
        self._rebuild_surface_representation()


def farthest_point_sampling(
    points: PointsN3,
    n_centers: int,
    seed: int = 42,
) -> PointsN3:
    points_np = np.asarray(points, dtype=np.float64)
    N = len(points_np)
    if N == 0:
        return np.empty((0, 3), dtype=np.float64)

    n_centers = min(n_centers, N)
    rng = np.random.default_rng(seed)

    chosen = np.empty(n_centers, dtype=np.int64)
    chosen[0] = int(rng.integers(0, N))

    diff = points_np - points_np[chosen[0]]
    dists = np.einsum("ij,ij->i", diff, diff)

    for i in range(1, n_centers):
        chosen[i] = int(np.argmax(dists))
        diff = points_np - points_np[chosen[i]]
        new_dists = np.einsum("ij,ij->i", diff, diff)
        dists = np.minimum(dists, new_dists)

    return points_np[chosen]


def _barycentric_coords(
    p: FloatArray,
    v0: FloatArray,
    v1: FloatArray,
    v2: FloatArray,
) -> FloatArray:
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0

    d00 = np.dot(v0v1, v0v1)
    d01 = np.dot(v0v1, v0v2)
    d11 = np.dot(v0v2, v0v2)
    d20 = np.dot(v0p, v0v1)
    d21 = np.dot(v0p, v0v2)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    bary = np.array([u, v, w], dtype=np.float64)
    bary = np.clip(bary, 0.0, 1.0)
    bary /= bary.sum() + 1e-12
    return bary