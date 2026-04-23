from abc import ABC, abstractmethod
from typing import List, Tuple

import open3d as o3d
import numpy as np
from cad_model import CADModel
from common_types import (
    Alpha,
    FloatArray,
    PointCloud3N,
    PrecisionMatrices,
    Priors,
    Rotation,
    Translation,
    ViewTransforms,
    as_translation,
)
from error_metrics import rotation_translation_error
import yaml
import transformPCDs


class GMMBase(ABC):
    """
    Base class for GMM-based registration methods.

    Internal conventions:
    - self.V, self.X, self.TV are stored as shape (3, N)
    - some helper routines convert temporarily to (N, 3)
    """

    R: list[Rotation]
    t: list[Translation]
    TV: list[PointCloud3N]
    T: list[ViewTransforms]

    def __init__(
        self,
        config: dict,
        V: list[np.ndarray],
        Xin: np.ndarray,
        cad_model: CADModel,
        debug_config: dict | None = None,
        pcd_loader=None,
    ) -> None:
        self.V = V
        self.Xin = Xin
        self.cad_model = cad_model
        self.debug_config = debug_config or {}
        self.config = config

        self.visualizer = None
        self.pcd_loader = pcd_loader
        self.iteration = 0

        self._initialize_views()
        self._initialize_model()
        self._initialize_common_parameters()
        self._initialize_transforms()
        self._initialize_debug_state()

    def _debug(self, msg: str) -> None:
        if self.debug_config.get("verbose", True):
            print(msg)

    def _warn(self, msg: str) -> None:
        print(f"[GMMBase] Warning: {msg}")

    def _initialize_views(self) -> None:
        self.characterize_point_density(self.V[0])

        if self.debug_config.get("remove_ground", False):
            self.V = self._remove_ground_points(self.V)

        self.V = [np.transpose(i) for i in self.V]

    def _initialize_model(self) -> None:

        self.X = np.transpose(self.Xin)
        self.normals = None
        self.a_per_center = None
        self.sigma2 = None
        self.pk = None

    def _initialize_transforms(self) -> None:
        self.R = [np.eye(3) for _ in range(self.M)]
        self.t: list[Translation] = [
            np.zeros(3, dtype=np.float64) for _ in range(len(self.V))
        ]

        self.TV = [
            R @ V + as_translation(t).reshape(3, 1)
            for R, V, t in zip(self.R, self.V, self.t)
        ]

        self.T = []

    def _initialize_common_parameters(self) -> None:
        self.epsilon = 1e-9
        self.updatePriors = False
        self.gamma = 0.1
        self.M = len(self.V)
        self.dim, self.K = self.X.shape

    def _initialize_debug_state(self) -> None:
        self.errors_history = {"rotation": [], "translation": []}
        self.T_gt = self.debug_config.get("T_gt", None)

    def _remove_ground_points(
        self, V, normal_threshold=0.8, min_inliers=20, z_percentile=20
    ):

        filtered = []
        total_removed = 0
        thresholds = [0.05, 0.10, 0.15, 0.20]

        for i, pc in enumerate(V):
            # Step 1: isolate low points for RANSAC
            z_vals = pc[:, 2] if pc.shape[1] == 3 else pc[2, :]
            z_cutoff = np.percentile(z_vals, z_percentile)
            low_mask = z_vals < z_cutoff

            ground_candidates = pc[low_mask] if pc.shape[1] == 3 else pc[:, low_mask].T

            if len(ground_candidates) < 10:
                self._debug(f"View {i}: too few low points, skipping ground removal")
                filtered.append(pc)
                continue

            # Step 2: run RANSAC on low points only
            pcd_candidates = o3d.geometry.PointCloud()
            pcd_candidates.points = o3d.utility.Vector3dVector(
                np.ascontiguousarray(ground_candidates, dtype=np.float64)
            )

            plane_found = False
            plane_model = None
            for threshold in thresholds:
                plane_model, inliers = pcd_candidates.segment_plane(
                    distance_threshold=threshold, ransac_n=3, num_iterations=1000
                )
                a, b, c, d = plane_model
                if c > normal_threshold and len(inliers) >= min_inliers:
                    plane_found = True
                    break

            if not plane_found:
                a, b, c, d = plane_model
                self._warn(
                    f"View {i}: no ground plane found "
                    f"(last normal=({a:.2f},{b:.2f},{c:.2f}), abs(c)={abs(c):.3f})"
                )
                filtered.append(pc)
                continue

            # Step 3: apply fitted plane to full point cloud
            full_pc = pc if pc.shape[1] == 3 else pc.T
            pcd_full = o3d.geometry.PointCloud()
            pcd_full.points = o3d.utility.Vector3dVector(
                np.ascontiguousarray(full_pc, dtype=np.float64)
            )

            # Compute signed distance of all points to fitted plane
            points = np.asarray(pcd_full.points)
            distances = np.abs(points @ np.array([a, b, c]) + d)
            ground_mask = distances < threshold

            pcd_filtered = pcd_full.select_by_index(np.where(~ground_mask)[0])
            removed = ground_mask.sum()
            total_removed += removed
            self._debug(
                f"View {i}: removed {removed} ground points "
                f"(threshold={threshold:.2f}m, normal=({a:.2f},{b:.2f},{c:.2f}))"
            )

            result = np.asarray(pcd_filtered.points)
            # restore original shape convention
            filtered.append(result if pc.shape[1] == 3 else result.T)

        total_points = sum(len(v) if v.shape[1] == 3 else v.shape[1] for v in V)
        self._debug(
            f"Ground removal: {total_removed}/{total_points} points removed "
            f"({total_removed / total_points * 100:.1f}%)"
        )
        return filtered

    def sse(self, A, B):
        """Return pairwise squared Euclidean distances between columns of A and B."""

        A = np.moveaxis(A[np.newaxis, :, :], 0, -1)  # results in a (3, N, 1) matrix
        B = np.swapaxes(
            np.moveaxis(B[np.newaxis, :, :], 0, -1), 1, -1
        )  # results in a (3, 1, K) matrix

        C = np.sum(
            np.power((A - B), 2), axis=0
        )  # sum over the the first axis of the A and B (three dimensions)
        if isinstance(C, (list, tuple, np.ndarray)):
            return C
        else:
            return C[0][0]

    def update_viz(self, it) -> None:

        # compute error for this iteration
        if self.T_gt is not None and self.T:
            T_gt = self.T_gt
            nObs = len(self.V)
            T = self.T
            T_1 = [
                transformPCDs.homogeneous_transform(
                    T[-1][0][i], T[-1][1][i].reshape(-1)
                )
                for i in range(nObs // 2)
            ]
            T_2 = [
                transformPCDs.homogeneous_transform(
                    T[-1][0][i], T[-1][1][i].reshape(-1)
                )
                for i in range(nObs // 2, nObs)
            ]

            self.log_error(T_gt, T_1, T_2, it)

        # update 3D plot
        if self.visualizer is not None:
            self.visualizer.update_gmm_state(
                self.X,
                it,
                self.TV,
                self.Q,
                self.normals,
                self.sigma2,
                self.a_per_center,
            )
            self.visualizer.log_pointcloud_surface_distances(
                self.TV, self.cad_model, it
            )
            self.visualizer.log_pointcloud_distances_visual(self.TV, self.cad_model, it)

        return

    def sample_visible_centers(
        self,
        K: int,
        sensor_positions: List[np.ndarray],
        oversample_factor: int = 4,
        max_attempts: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample exactly K centers, preferring surfaces visible from at least one sensor.

        Strategy:
        1. Oversample surface candidates.
        2. Keep candidates whose normals face at least one sensor.
        3. If too many, subsample to K with farthest point sampling.
        4. If too few, retry with more candidates.
        5. If still too few, top up using the best non-visible candidates ranked by
            maximum facing score across sensors.

        Returns:
            visible_centers: (K, 3)
            visible_normals: (K, 3)
        """

        best_visible_centers = None
        best_visible_normals = None

        n_candidates = max(K * oversample_factor, K)

        for attempt in range(max_attempts):
            candidates = self.cad_model.sample_surface_centers(
                n_candidates, method="poisson_disk"
            )
            normals = self.cad_model.get_surface_normals_at_points(candidates)
            normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)

            # Max facing score over all sensors
            max_facing = np.full(len(candidates), -np.inf, dtype=np.float64)

            for sensor_pos in sensor_positions:
                sensor_pos = np.asarray(sensor_pos, dtype=np.float64).reshape(1, 3)
                to_sensor = sensor_pos - candidates  # (N, 3)
                to_sensor = to_sensor / (
                    np.linalg.norm(to_sensor, axis=1, keepdims=True) + 1e-9
                )
                facing = np.sum(normals * to_sensor, axis=1)  # (N,)
                max_facing = np.maximum(max_facing, facing)

            visible_mask = max_facing > 0.0
            visible_centers = candidates[visible_mask]
            visible_normals = normals[visible_mask]

            self._debug(
                f"Visibility filter attempt {attempt + 1}: "
                f"{visible_mask.sum()}/{len(candidates)} visible "
                f"({100.0 * visible_mask.sum() / len(candidates):.1f}%)"
            )

            # Keep best attempt in case we still need fallback later
            if best_visible_centers is None or len(visible_centers) > len(
                best_visible_centers
            ):
                best_visible_centers = visible_centers
                best_visible_normals = visible_normals

            if len(visible_centers) >= K:
                idx = self._farthest_point_sample(visible_centers, K)
                return visible_centers[idx], visible_normals[idx]

            n_candidates *= 2  # retry with more candidates

        # Fallback: use best attempt and top up from near-visible candidates
        visible_centers = best_visible_centers
        visible_normals = best_visible_normals

        if visible_centers is None:
            raise RuntimeError("Failed to sample any surface candidates.")

        n_visible = len(visible_centers)

        if n_visible == 0:
            # Hard fallback: sample exactly K from full surface
            self._warn(
                "No visible centers found; falling back to unconstrained surface sampling."
            )
            centers = self.cad_model.sample_surface_centers(K, method="poisson_disk")
            normals = self.cad_model.get_surface_normals_at_points(centers)
            normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)
            return centers, normals

        if n_visible < K:
            self._warn(
                f"Only {n_visible} visible centers found after retries; topping up to {K}."
            )

            # Resample a large pool and rank non-visible candidates by visibility score
            fallback_candidates = self.cad_model.sample_surface_centers(
                max(4 * K, K), method="poisson_disk"
            )
            fallback_normals = self.cad_model.get_surface_normals_at_points(
                fallback_candidates
            )
            fallback_normals = fallback_normals / (
                np.linalg.norm(fallback_normals, axis=1, keepdims=True) + 1e-9
            )

            max_facing = np.full(len(fallback_candidates), -np.inf, dtype=np.float64)
            for sensor_pos in sensor_positions:
                sensor_pos = np.asarray(sensor_pos, dtype=np.float64).reshape(1, 3)
                to_sensor = sensor_pos - fallback_candidates
                to_sensor = to_sensor / (
                    np.linalg.norm(to_sensor, axis=1, keepdims=True) + 1e-9
                )
                facing = np.sum(fallback_normals * to_sensor, axis=1)
                max_facing = np.maximum(max_facing, facing)

            # Prefer near-visible points first (largest max_facing, even if <= 0)
            order = np.argsort(-max_facing)
            needed = K - n_visible

            extra_centers = fallback_candidates[order[:needed]]
            extra_normals = fallback_normals[order[:needed]]

            visible_centers = np.vstack([visible_centers, extra_centers])
            visible_normals = np.vstack([visible_normals, extra_normals])

        # If top-up created duplicates or too many, enforce exactly K with FPS
        if len(visible_centers) > K:
            idx = self._farthest_point_sample(visible_centers, K)
            visible_centers = visible_centers[idx]
            visible_normals = visible_normals[idx]

        assert len(visible_centers) == K, (
            f"Expected {K} centers, got {len(visible_centers)}"
        )
        return visible_centers, visible_normals

    def compute_visibility_with_occlusion(self, centers, normals, sensor_positions):
        """
        Check if center is visible:
        1. Normal faces sensor (dot product > 0)
        2. Ray from center to sensor doesn't hit other geometry
        """
        import trimesh

        mesh = self.cad_model.mesh  # assuming trimesh

        visible = np.zeros(len(centers), dtype=bool)

        for sensor_pos in sensor_positions:
            for i, (center, normal) in enumerate(zip(centers, normals)):
                # First check: normal facing sensor?
                to_sensor = sensor_pos - center
                to_sensor_norm = to_sensor / (np.linalg.norm(to_sensor) + 1e-9)

                if np.dot(normal, to_sensor_norm) <= 0:
                    continue  # back-facing, skip

                # Second check: ray hits anything?
                ray_origin = (
                    center + normal * 0.001
                )  # offset slightly to avoid self-hit
                ray_direction = to_sensor_norm

                hits = mesh.ray.intersects_location(
                    ray_origins=[ray_origin], ray_directions=[ray_direction]
                )[0]

                if len(hits) == 0:
                    visible[i] = True
                else:
                    # Check if hit is beyond sensor
                    hit_dist = np.linalg.norm(hits[0] - center)
                    sensor_dist = np.linalg.norm(to_sensor)
                    if hit_dist > sensor_dist:
                        visible[i] = True

        return visible

    def _farthest_point_sample(self, points: np.ndarray, K: int, seed=42) -> np.ndarray:
        """Subsample K points using farthest point sampling for even coverage."""

        rng = np.random.default_rng(seed)
        N = len(points)
        selected = [int(rng.integers(N))]
        distances = np.full(N, np.inf)

        for _ in range(K - 1):
            last = points[selected[-1]]
            dist_to_last = np.linalg.norm(points - last, axis=1)
            distances = np.minimum(distances, dist_to_last)
            selected.append(np.argmax(distances))

        return np.array(selected)

    def _aggregate_alpha_across_views(self, alpha_list):
        """
        Aggregate alpha from multiple views into single per-center measure.

        Args:
            alpha_list: List of alpha arrays, each (N_i, K)
                    Each row is a point, each column is a center

        Returns:
            alpha_aggregated: (K,) - mean responsibility per center across all views
        """
        K = self.K

        # Initialize aggregation
        total_responsibility = np.zeros(K)
        total_points = 0

        for m, alpha_m in enumerate(alpha_list):
            # Handle both numpy array and matrix types
            if hasattr(alpha_m, "A"):
                # Convert matrix to array
                alpha_array = np.asarray(alpha_m)
            else:
                alpha_array = alpha_m

            N_m, K_m = alpha_array.shape  # Points x Centers

            # Debug shape mismatch
            if K_m != K:
                self._warn(f"View {m} has {K_m} centers, expected {K}")

            # Sum across points (axis=0) to get total responsibility per center
            resp_per_center = np.sum(alpha_array, axis=0)  # (K,) ← THIS IS THE KEY LINE

            if K_m != K:
                # Handle mismatch
                K_safe = min(K_m, K)
                total_responsibility[:K_safe] += resp_per_center[:K_safe]
            else:
                total_responsibility += resp_per_center

            total_points += N_m

        # Average responsibility per point
        alpha_aggregated = total_responsibility / total_points
        return alpha_aggregated

    def characterize_point_density(self, point_cloud):
        """Compute spatial point density."""
        from scipy.spatial import cKDTree

        points = (
            np.asarray(point_cloud.points)
            if hasattr(point_cloud, "points")
            else point_cloud
        )

        # Build KD-tree
        tree = cKDTree(points)

        # Query k=10 nearest neighbors for each point
        distances, _ = tree.query(points, k=10)

        # Use mean distance to 10th neighbor as density metric
        nn_distances = distances[:, -1]  # 10th nearest

        self._debug("=== POINT DENSITY ===")
        self._debug(f"Points: {len(points)}")
        self._debug(f"Mean NN distance: {np.mean(nn_distances) * 1000:.2f}mm")
        self._debug(f"Median NN distance: {np.median(nn_distances) * 1000:.2f}mm")
        self._debug(f"Std NN distance: {np.std(nn_distances) * 1000:.2f}mm")
        self._debug(
            f"Min/Max: [{np.min(nn_distances) * 1000:.1f}, {np.max(nn_distances) * 1000:.1f}]mm"
        )

        # Density classification
        median_mm = np.median(nn_distances) * 1000
        if median_mm < 5:
            density = "Very Dense"
        elif median_mm < 15:
            density = "Dense"
        elif median_mm < 30:
            density = "Medium"
        elif median_mm < 50:
            density = "Sparse"
        else:
            density = "Very Sparse"

        self._debug(f"Classification: {density}")

        return nn_distances

    def log_error(self, T_gt, T_1, T_2, it):

        # compute mean relative transformation error across views
        T_calib = [np.dot(np.linalg.inv(T_2[i]), T_1[i]) for i in range(len(T_1))]
        T_est = transformPCDs.mean_transform(T_calib)
        T_error = rotation_translation_error(T_gt, T_est)

        # Compute total rotation error
        rot_error_total = np.sqrt(
            T_error["roll_rad"] ** 2
            + T_error["pitch_rad"] ** 2
            + T_error["yaw_rad"] ** 2
        )
        rot_error_deg = np.degrees(rot_error_total)

        # Compute total translation error
        trans_error_m = np.sqrt(
            T_error["dx"] ** 2 + T_error["dy"] ** 2 + T_error["dz"] ** 2
        )
        sigma = np.sqrt(1.0 / np.mean(self.Q))
        self._debug(
            f"iter={it} | sigma2={sigma**2:.6f} | sigma={sigma:.4f} | "
            f"trans_err_m={trans_error_m:.4f} | rot_err_deg={rot_error_deg:.4f} | "
            f"beta={self.beta:.6f}"
        )

        # Track history
        self.errors_history["rotation"].append(rot_error_deg)
        self.errors_history["translation"].append(trans_error_m)

        if self.visualizer is not None:
            self.visualizer.log_calibration_error_detailed(T_gt, T_est, it)

    def run_optimization(self, num_iter):

        self._debug(f"Running EM for {num_iter} iterations")

        self.max_iterations = num_iter
        if self.visualizer and self.pcd_loader:
            pcd_loader = self.pcd_loader
            # Get first scan from each sensor
            left_sensor_scans = pcd_loader.get_sensor_scans_full(0)
            right_sensor_scans = pcd_loader.get_sensor_scans_full(1)

            scan_limit = 5
            for i, scan in enumerate(left_sensor_scans):
                self.visualizer.log_raw_sensor_data(
                    points=np.asarray(scan.points),
                    sensor_name=f"left_sensor/scan_{i:03d}",
                    color=[255, 100, 100],
                )
                if i > scan_limit:
                    break

            for i, scan in enumerate(right_sensor_scans):
                self.visualizer.log_raw_sensor_data(
                    points=np.asarray(scan.points),
                    sensor_name=f"right_sensor/scan_{i:03d}",
                    color=[100, 100, 255],
                )
                if i > scan_limit:
                    break
        else:
            self._debug("No pcd, not logging raw")

        # expensive
        if False:
            self.visualizer.log_pointcloud_surface_distances(self.TV, self.cad_model, 0)
            self.visualizer.log_pointcloud_distances_visual(self.TV, self.cad_model, 0)

        for it in range(num_iter):
            print(f"EM Iteration {it}")
            self.em_step(it)
            self.update_viz(it + 1)
            self.iteration += 1

        if self.visualizer is not None:
            self.visualizer.log_final_calibrated_scans(self.pcd_loader, self.R, self.t)
            self.visualizer.log_convergence_plot(self.errors_history)

        return self.X, self.TV, self.T, self.pk

    @abstractmethod
    def e_step(
        self,
        TV: list[PointCloud3N],
    ) -> tuple[list[Alpha], list[np.ndarray]]:
        raise NotImplementedError

    @abstractmethod
    def m_step(
        self,
        V: list[PointCloud3N],
        alpha_cad: list[Alpha],
    ) -> tuple[list[PointCloud3N], ViewTransforms, PrecisionMatrices, Priors]:
        raise NotImplementedError

    @abstractmethod
    def em_step(self, it) -> None:
        raise NotImplementedError("Subclass must implement em_step")
