import os
from typing import Any

import numpy as np
from numba import njit
from numpy.typing import NDArray

import transformPCDs
from common_types import (
    Alpha,
    Centers3K,
    NormalsN3,
    PointCloud3N,
    PointsN3,
    PrecisionMatrices,
    PrecisionVector,
    Priors,
    Rotation,
    Transform,
    Translation,
    ViewTransforms,
    as_rotation,
    as_translation,
)
from gmm_base import GMMBase
from rerun_gmm_visualizer import RerunGMMVisualizer


def _skew(v: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


# numba implementations
@njit(cache=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def skew3_numba(v: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z = v[0], v[1], v[2]
    out = np.empty((3, 3), dtype=np.float64)
    out[0, 0] = 0.0
    out[0, 1] = -z
    out[0, 2] = y
    out[1, 0] = z
    out[1, 1] = 0.0
    out[1, 2] = -x
    out[2, 0] = -y
    out[2, 1] = x
    out[2, 2] = 0.0
    return out


@njit(cache=True)  # pyright: ignore[reportUntypedFunctionDecorator]
def accumulate_gn_system_numba(
    Y: NDArray[np.float64],
    M_0: NDArray[np.float64],
    M_1: NDArray[np.float64],
    R: NDArray[np.float64],
    t: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    N = Y.shape[0]
    H = np.zeros((6, 6), dtype=np.float64)
    g = np.zeros(6, dtype=np.float64)

    for n in range(N):
        y_n = Y[n]
        p_n = R @ y_n + t
        M0n = M_0[n]
        grad_p = M0n @ p_n - M_1[n]

        J = np.zeros((3, 6), dtype=np.float64)
        J[:, :3] = -R @ skew3_numba(y_n)
        J[:, 3:] = R

        H += J.T @ M0n @ J
        g += J.T @ grad_p

    H = 0.5 * (H + H.T)
    return H, g


# point cloud helper
def as_points_n3(pc: PointCloud3N | PointsN3) -> PointsN3:
    """Return point cloud as shape (N, 3). Accepts either (N, 3) or (3, N)."""
    if pc.ndim != 2:
        raise ValueError(f"Expected 2D point cloud, got {pc.shape}")
    if pc.shape[1] == 3:
        return np.asarray(pc, dtype=np.float64)
    if pc.shape[0] == 3:
        return np.asarray(pc.T, dtype=np.float64)
    raise ValueError(f"Expected (N,3) or (3,N), got {pc.shape}")


# ============================================================================ #
#                           Module-level helpers                               #
# ============================================================================ #


def build_anisotropic_precision_matrices(
    normals: NormalsN3,
    alphas: PrecisionVector,
    sigma_sq: float,
) -> PrecisionMatrices:
    normals = np.asarray(normals)
    alphas = np.asarray(alphas)
    nnT = normals[:, :, np.newaxis] * normals[:, np.newaxis, :]
    Lambda = alphas[:, np.newaxis, np.newaxis] * nnT + np.eye(3)
    return Lambda / sigma_sq


class CADCalib(GMMBase):
    """
    CAD-informed multi-view LiDAR calibration with fixed CAD-derived GMM centers.

    State layout:
    - self.X, self.normals, self.a_per_center: current CAD/GMM geometry
    - self.R_extrinsic, self.t_extrinsic: shared per-sensor extrinsics
    - self.X3, self.x_sq, self.x_dot_n, self.Lambda_x: cached center terms
    - self.sigma2, self.F_k, self.pk: current probabilistic model parameters

    Conventions:
    - external point clouds may be (3, N) or (N, 3)
    - internal per-point computations use (N, 3), normalized via as_points_n3()
    """

    R_extrinsic: list[Rotation]
    t_extrinsic: list[Translation]

    def __init__(
        self,
        config: dict[str, Any],
        V: list[PointCloud3N],
        Xin: Centers3K,
        cad_model: Any,
        debug_config: dict[str, Any] | None = None,
        pcd_loader: Any | None = None,
    ) -> None:

        super().__init__(config, V, Xin, cad_model, debug_config, pcd_loader)

        # centers/normals -> sensor setup -> scalar distribution params ->
        # optional ground augmentation -> anisotropy -> cached center terms

        self._initialize_centers_and_normals()
        self._initialize_sensor_setup()
        self._initialize_distribution_parameters()
        self._maybe_add_ground_centers()
        self._initialize_anisotropy()
        self._initialize_bounding_volume()
        self._compute_f_matrix_factor()
        self._refresh_center_caches()
        self._initialize_visualizer()
        self._log_init_summary()
        self._log_initial_gt_error()

    # init helpers
    def _initialize_centers_and_normals(self) -> None:

        use_visibility = self.debug_config.get("visibility_filter", False)
        if use_visibility:
            sensor_positions_cfg = self.debug_config["sensor_positions"]
            visible_centers, visible_normals = self.sample_visible_centers(
                self.K, sensor_positions_cfg
            )
            self.X = visible_centers.T
            self.K = self.X.shape[1]
            self.normals = np.asarray(visible_normals, dtype=np.float64)
            self.normals /= np.linalg.norm(self.normals, axis=1, keepdims=True) + 1e-12
        else:
            self.X = self.Xin.T
            self.normals = self.cad_model.get_surface_normals_at_points(self.X.T)
            self.normals /= np.linalg.norm(self.normals, axis=1, keepdims=True) + 1e-12
        self.initial_centers = self.X.copy()

    def _initialize_sensor_setup(self) -> None:

        n_sensors = 2
        views_per_sensor = len(self.V) // n_sensors
        if self.debug_config.get("sensor_positions"):
            sensor_positions = [
                np.asarray(p, dtype=np.float64)
                for p in self.debug_config["sensor_positions"]
            ]
            self.sensor_positions = sensor_positions

            views_per_sensor = len(self.V) // len(sensor_positions)
            self.sensor_positions_per_view = [
                np.zeros(3, dtype=np.float64) for _ in range(len(self.V))
            ]

            n_sensors = len(sensor_positions)

        self.sensor_assignment = [
            s_idx for s_idx in range(n_sensors) for _ in range(views_per_sensor)
        ]

        self.R_extrinsic = [
            np.asarray(self.R[s_idx * views_per_sensor], dtype=np.float64).copy()
            for s_idx in range(n_sensors)
        ]
        self.t_extrinsic = [
            as_translation(self.t[s_idx * views_per_sensor]).copy()
            for s_idx in range(n_sensors)
        ]

    def _initialize_distribution_parameters(self) -> None:

        # Initial sigma² from bounding box SSE
        minXYZ, maxXYZ = [], []
        TVX = self.TV.copy()
        TVX.append(self.X)
        for pc in TVX:
            minXYZ.append(np.min(pc, axis=1))
            maxXYZ.append(np.max(pc, axis=1))

        minXYZ_arr = np.min(minXYZ, axis=0).reshape((self.dim, 1))
        maxXYZ_arr = np.max(maxXYZ, axis=0).reshape((self.dim, 1))

        self.Q: PrecisionVector = (
            np.multiply(np.ones((1, self.K)), (1.0 / self.sse(minXYZ_arr, maxXYZ_arr)))
            .reshape((self.K, 1))
            .astype(np.float64)
        )
        self.sigma2 = 1.0 / float(self.Q[0, 0])
        self.sigma2_init = self.sigma2
        self.pk: Priors = 1 / (self.K * (self.gamma + 1))
        self.pk = np.transpose(self.pk)

        self.h: float = np.divide(2, np.mean(self.Q))
        self.beta: float = np.divide(self.gamma, np.multiply(self.h, self.gamma + 1))

        self.T: list[ViewTransforms] = []
        self.updatePriors = False

        self.w = self.debug_config.get("outlier_ratio", 0.1)
        sorted_hash = hash(np.sort(self.X, axis=1).tobytes())
        self._debug(f"Sorted center hash: {sorted_hash}")
        self._debug(f"First 3 centers:\n{self.X[:, :3]}")
        self._debug(f"Centers 0-9:\n{self.X[:, :10].T}")
        self._debug(f"Centers 10-20:\n{self.X[:, 10:20].T}")

    def _maybe_add_ground_centers(self) -> None:

        self._n_ground_centers = 0

        ground_config = self.debug_config.get("ground_plane_centers", {})
        if ground_config.get("enabled", False):
            n_ground = ground_config.get("n_centers", 120)
            z_offset = ground_config.get("z_offset", -0.03)  # how far below bbox min

            x_min, y_min, z_min = self.X.min(axis=1)
            x_max, y_max, _ = self.X.max(axis=1)
            z_ground = z_min - z_offset

            n_side = int(np.sqrt(n_ground))
            x = np.linspace(x_min, x_max, n_side)
            y = np.linspace(y_min, y_max, n_side)
            xx, yy = np.meshgrid(x, y)

            ground_centers = np.column_stack(
                [xx.ravel(), yy.ravel(), np.full(n_side * n_side, z_ground)]
            )

            ground_normals = np.tile([0.0, 0.0, 1.0], (len(ground_centers), 1))

            self.X = np.hstack([self.X, ground_centers.T])
            self.normals = np.vstack([self.normals, ground_normals])
            self.K = self.X.shape[1]

            self._debug(
                f"Added {len(ground_centers)} ground centers at z={z_ground:.3f}m "
                f"(bbox z_min={z_min:.3f}m - offset={z_offset:.3f}m), "
                f"total K={self.K}"
            )

            self._n_ground_centers = (
                len(ground_centers) if ground_config.get("enabled", False) else 0
            )

    def _initialize_anisotropy(self) -> None:

        # LSG-CPD anisotropy hyperparameters
        a_max = self.debug_config.get("alimit", 30)

        # compute a_per_center on CAD centers only, then append ground
        n_cad = self.K - self._n_ground_centers
        curvature = self.cad_model.compute_surface_variation_at_points(
            self.X[:, :n_cad].T
        )

        curv = np.log1p(curvature)

        p10, p90 = np.percentile(curv, [10, 90])
        curv_norm = np.clip((curv - p10) / (p90 - p10 + 1e-12), 0.0, 1.0)

        # low curvature -> high anisotropy
        self.a_per_center = a_max * (1.0 - curv_norm)

        curv_pct = np.percentile(curvature, [0, 25, 50, 75, 90, 95, 99, 99.5, 100])
        a_pct = np.percentile(self.a_per_center, [0, 25, 50, 75, 90, 95, 99, 100])
        self._debug(f"Curvature percentiles: {curv_pct}")
        self._debug(f"a_k percentiles: {a_pct}")

        self.curvature = curvature

        if self._n_ground_centers > 0:
            self.a_per_center = np.concatenate(
                [self.a_per_center, np.full(self._n_ground_centers, a_max)]
            )

    def _initialize_bounding_volume(self) -> None:
        all_points = np.hstack([self.X] + self.TV)
        self.bounding_volume = np.prod(
            np.max(all_points, axis=1) - np.min(all_points, axis=1)
        )

    def _refresh_center_caches(self) -> None:
        """recompute cached center-dependent quantities after geometry change"""
        self.Lambda_per_center = self._build_lambda_per_center()
        self.X3 = self.X.T
        self.normals_arr = np.asarray(self.normals, dtype=np.float64)
        self.x_sq = np.sum(self.X3 * self.X3, axis=1)
        self.x_dot_n = np.sum(self.X3 * self.normals_arr, axis=1)
        self.Lambda_x = np.einsum("kij,kj->ki", self.Lambda_per_center, self.X3)

    def _initialize_visualizer(self) -> None:
        self.enable_rerun = os.environ.get("RERUN", "ON").upper() != "OFF"
        self.enable_visualizer = bool(self.debug_config.get("enable_visualizer", True))
        use_visualizer = self.enable_rerun and self.enable_visualizer

        if use_visualizer:
            self.visualizer = RerunGMMVisualizer(
                cad_model=self.cad_model,
                point_clouds=self.V,
                initial_centers=self.initial_centers,
                initial_Q=self.Q,
                normals=self.normals,
                view_labels=self.pcd_loader.view_labels,
                config=self.config,
                debug_config=self.debug_config,
            )
        else:
            self.visualizer = None

    def _log_init_summary(self) -> None:
        self._debug("Rerun visualization initialized successfully")
        self._debug("\n=== Basic Parameters ===")
        self._debug(f"K (centers): {self.K}")
        self._debug(f"Beta: {self.beta}  Gamma: {self.gamma}")
        self._debug(f"Sigma: {np.sqrt(self.sigma2) * 1000:.1f}mm")
        self._debug(f"Update priors: {self.updatePriors}")
        self._debug(
            f"Curvature: min={self.curvature.min():.3f}, "
            f"max={self.curvature.max():.3f}, "
            f"mean={self.curvature.mean():.3f}"
        )
        self._debug(
            f"Anisotropy a_k: min={self.a_per_center.min():.1f}, "
            f"max={self.a_per_center.max():.1f}, "
            f"mean={self.a_per_center.mean():.1f}"
        )
        self._debug(f"w0={self.w0:.6f}")
        self._debug(
            f"F_k stats: min={self.F_k.min():.3e}, "
            f"max={self.F_k.max():.3e}, "
            f"mean={self.F_k.mean():.3e}"
        )

    def _log_initial_gt_error(self) -> None:
        if self.T_gt is not None:
            nObs = len(self.V)
            T_1_init = [
                transformPCDs.homogeneous_transform(self.R[i], self.t[i])
                for i in range(nObs // 2)
            ]
            T_2_init = [
                transformPCDs.homogeneous_transform(self.R[i], self.t[i])
                for i in range(nObs // 2, nObs)
            ]
            self.log_error(self.T_gt, T_1_init, T_2_init, 0)

    # utils
    def get_sensor_origin_for_view(self, view_idx: int) -> Translation:
        s = self.sensor_assignment[view_idx]
        return as_translation(self.t_extrinsic[s])

    # probability model
    def _compute_f_matrix_factor(self) -> None:
        """
        build multiplicative factor for each center.
        scalar w0 (global outlier weight),
        per-center F_k only through f_Y = pk * sqrt(1 + a_k)
        No confidence filtering => confidence_X = 1, f_X is scalar
        """
        vol = np.sqrt(1.0 + self.a_per_center)  # (K,)
        pk_vec = np.asarray(self.pk, dtype=np.float64).reshape(-1)
        f_Y = pk_vec * vol  # (K,)

        # Scalar w0 matching MATLAB: V * w * sum(f_Y * vol) * (2π σ²)^(-3/2)
        w0 = (
            self.bounding_volume
            * self.w
            * float(np.sum(f_Y * vol))
            * (2.0 * np.pi * self.sigma2) ** (-1.5)
        )
        w0 = w0 / (1.0 - self.w + w0)
        self.w0 = w0

        f_X = (1.0 - w0) / (w0 + 1e-12)  # scalar
        self.F_k = f_X * f_Y  # (K,)

    def _build_lambda_per_center(self) -> PrecisionMatrices:
        normals = np.asarray(self.normals, dtype=np.float64)
        a = np.asarray(self.a_per_center, dtype=np.float64)
        nnT = normals[:, :, None] * normals[:, None, :]
        return np.eye(3, dtype=np.float64)[None, :, :] + a[:, None, None] * nnT

    # optimization
    def _compute_gn_system_python_reference(
        self,
        Y: PointsN3,
        M_0: NDArray[np.float64],
        M_1: NDArray[np.float64],
        R: Rotation,
        t: Translation,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Slow reference implementation for debugging against numba kernel"""
        N = Y.shape[0]
        H = np.zeros((6, 6), dtype=np.float64)
        g = np.zeros(6, dtype=np.float64)

        for n in range(N):
            y_n = Y[n]
            p_n = R @ y_n + t
            M0n = M_0[n]
            grad_p = M0n @ p_n - M_1[n]
            J = np.zeros((3, 6), dtype=np.float64)
            J[:, :3] = -R @ _skew(y_n)
            J[:, 3:] = R
            H += J.T @ M0n @ J
            g += J.T @ grad_p

        return 0.5 * (H + H.T), g

    def _compute_gn_system(
        self,
        Y: PointsN3,
        M_0: NDArray[np.float64],
        M_1: NDArray[np.float64],
        R: Rotation,
        t: Translation,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return accumulate_gn_system_numba(
            np.asarray(Y, dtype=np.float64),
            np.asarray(M_0, dtype=np.float64),
            np.asarray(M_1, dtype=np.float64),
            as_rotation(R),
            as_translation(t),
        )

    @staticmethod
    def _apply_gn_delta(
        R: Rotation,
        t: Translation,
        delta: NDArray[np.float64],
    ) -> Transform:
        omega = delta[:3]
        v = delta[3:]
        theta = np.linalg.norm(omega)

        if theta < 1e-12:
            dR, V = np.eye(3), np.eye(3)
        else:
            K = _skew(omega / theta)
            dR = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
            V = (
                np.eye(3)
                + (1 - np.cos(theta)) / theta**2 * _skew(omega)
                + (theta - np.sin(theta)) / theta**3 * (_skew(omega) @ _skew(omega))
            )

        R_new = R @ dR
        t_new = t + R @ (V @ v)
        U, _, Vt = np.linalg.svd(R_new)
        R_new = U @ Vt
        if np.linalg.det(R_new) < 0:
            R_new = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
        return R_new, t_new

    def _solve_shared_extrinsics(
        self,
        V: list[PointsN3],
        alpha_cad: list[Alpha],
        max_gn_iters: int = 2,
        damping: float = 1e-6,
    ) -> ViewTransforms:
        n_sensors = len(self.R_extrinsic)
        Lambda = self.Lambda_per_center
        Lambda_x = self.Lambda_x
        for _ in range(max_gn_iters):
            H_per = [np.zeros((6, 6)) for _ in range(n_sensors)]
            g_per = [np.zeros(6) for _ in range(n_sensors)]

            for i, (Y_raw, a_cad_i) in enumerate(zip(V, alpha_cad, strict=True)):
                s = self.sensor_assignment[i]
                Y = as_points_n3(np.asarray(Y_raw, dtype=np.float64))

                M_0 = np.einsum("nk,kij->nij", a_cad_i, Lambda)
                M_1 = np.einsum("nk,ki->ni", a_cad_i, Lambda_x)
                H_i, g_i = self._compute_gn_system(
                    Y, M_0, M_1, self.R_extrinsic[s], self.t_extrinsic[s]
                )
                H_per[s] += H_i
                g_per[s] += g_i

            max_delta = 0.0
            for s in range(n_sensors):
                try:
                    delta = np.linalg.solve(H_per[s] + damping * np.eye(6), -g_per[s])
                except np.linalg.LinAlgError:
                    delta = np.zeros(6)
                R_new, t_new = self._apply_gn_delta(
                    self.R_extrinsic[s], self.t_extrinsic[s], delta
                )
                self.R_extrinsic[s] = R_new
                self.t_extrinsic[s] = t_new
                max_delta = max(max_delta, np.linalg.norm(delta))

            if max_delta < 1e-8:
                break

        if self.debug_config.get("log_hessian", True):
            for s in range(n_sensors):
                eigvals, eigvecs = np.linalg.eigh(H_per[s])
                self._debug(f"\n=== Sensor {s} Hessian Eigenvalues ===")
                self._debug(f"  {eigvals}")
                self._debug(f"  Condition: {eigvals[-1] / (eigvals[0] + 1e-12):.1f}")
                self._debug(f"  Min: {eigvals[0]:.6e}")
                self._debug(f"  Weakest:  {eigvecs[:, 0]}  [ωx, ωy, ωz, vx, vy, vz]")
                self._debug(f"  2nd weak: {eigvecs[:, 1]}")

        R_list = [self.R_extrinsic[self.sensor_assignment[i]] for i in range(len(V))]
        t_list = [
            self.t_extrinsic[self.sensor_assignment[i]].copy() for i in range(len(V))
        ]
        return R_list, t_list

    # ======================================================================== #
    #                               E-step                                     #
    # ======================================================================== #

    def e_step(
        self, TV: list[PointsN3]
    ) -> tuple[list[Alpha], list[NDArray[np.float64]]]:

        X3 = self.X3
        normals = self.normals_arr
        x_sq = self.x_sq
        x_dot_n = self.x_dot_n
        sigma2 = self.sigma2

        alpha_cad_list = []
        alpha_outlier_list = []

        bg = (2.0 * np.pi * sigma2) ** 1.5 / (self.bounding_volume + 1e-12)

        for cloud in TV:
            Y = as_points_n3(np.asarray(cloud, dtype=np.float64))

            y_sq = np.sum(Y * Y, axis=1)  # (N,)
            yx = Y @ X3.T  # (N, K)
            yn = Y @ normals.T  # (N, K)

            r_sq = y_sq[:, None] + x_sq[None, :] - 2.0 * yx
            n_dot = yn - x_dot_n[None, :]
            mahal = r_sq + self.a_per_center[None, :] * (n_dot**2)

            alpha_unnorm = self.F_k[None, :] * np.exp(-0.5 * mahal / (sigma2 + 1e-12))
            row_sum = alpha_unnorm.sum(axis=1) + bg

            alpha_cad_list.append(alpha_unnorm / (row_sum[:, None] + 1e-12))
            alpha_outlier_list.append(bg / (row_sum + 1e-12))

        return alpha_cad_list, alpha_outlier_list

    # ======================================================================== #
    #                               M-step                                     #
    # ======================================================================== #

    def m_step(
        self,
        V: list[PointsN3],
        alpha_cad: list[Alpha],
    ) -> tuple[list[PointsN3], ViewTransforms, PrecisionMatrices, Priors]:

        num_views = len(V)
        R_list, t_list = self._solve_shared_extrinsics(
            V=V, alpha_cad=alpha_cad, max_gn_iters=2, damping=1e-6
        )

        TV = [R_list[i] @ V[i] + t_list[i].reshape(3, 1) for i in range(num_views)]

        den = np.sum(np.asarray([np.sum(a, axis=0) for a in alpha_cad]), axis=0)

        total_num, total_den = 0.0, 0.0
        for i, Y_i in enumerate(TV):
            A_i = alpha_cad[i]
            Y = as_points_n3(np.asarray(Y_i, dtype=np.float64))
            y_sq = np.sum(Y * Y, axis=1)
            yx = Y @ self.X3.T
            yn = Y @ self.normals_arr.T

            r_sq = np.maximum(y_sq[:, None] + self.x_sq[None, :] - 2.0 * yx, 0.0)
            n_dot = yn - self.x_dot_n[None, :]
            r_prec = r_sq + self.a_per_center[None, :] * (n_dot**2)

            total_num += np.sum(A_i * r_prec)
            total_den += np.sum(A_i)

        self.sigma2 = total_num / (3.0 * total_den + 1e-12)
        Q_new = build_anisotropic_precision_matrices(
            self.normals, self.a_per_center, self.sigma2
        )
        pk = den / (np.sum(den) + 1e-12) if self.updatePriors else self.pk
        self.den = den

        return TV, (R_list, t_list), Q_new, pk

    # ======================================================================== #
    #                              EM Loop                                     #
    # ======================================================================== #

    def em_step(self, it: int) -> None:

        V, TV = self.V, self.TV

        # E-step
        alpha_cad, alpha_outlier = self.e_step(TV)

        out_frac = [f"{(a > 0.5).mean() * 100:.0f}%" for a in alpha_outlier]
        self._debug(f"Outlier >50%: {' | '.join(out_frac)}")
        self.alpha_outlier = alpha_outlier

        TV, (R, t), Q, pk = self.m_step(V=V, alpha_cad=alpha_cad)

        self.R = [self.R_extrinsic[s] for s in self.sensor_assignment]
        self.t = [self.t_extrinsic[s].copy() for s in self.sensor_assignment]

        self.T.append((R, t))
        self.TV, self.Q, self.pk = TV, Q, pk
        self.alpha = alpha_cad

        if self.visualizer:
            self.visualizer.log_em_convergence_stats(
                iteration=self.iteration,
                sigma2=self.sigma2,
                Q=self.Q,
                alpha=self.alpha[0],
                centers=np.asarray(self.X),
                TV=None,
                den=self.den,
                a_values=None,
            )

            self.visualizer.log_outlier_probabilities(
                TV=TV,
                alpha_outlier=alpha_outlier,
                iteration=self.iteration + 1,
                threshold=0.5,
            )

        self._debug("\n=== Basic Parameters ===")
        self._debug(
            f"K={self.K}  sigma={np.sqrt(self.sigma2) * 1000:.1f}mm  "
            f"sigma2={self.sigma2 * 1e6:.1f}mm²"
        )
