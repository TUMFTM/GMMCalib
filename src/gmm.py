import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from numpy.typing import NDArray

from common_types import Alpha, Centers3K, PointCloud3N, PrecisionVector, Priors, Rotation, Translation, ViewTransforms
import transformPCDs
from gmm_base import GMMBase
from rerun_gmm_visualizer import RerunGMMVisualizer


class GMM(GMMBase):
    def __init__(
        self,
        config: Dict[str, Any],
        V: List[PointCloud3N],
        Xin: Centers3K,
        cad_model: Any,
        debug_config: Optional[Dict[str, Any]] = None,
        pcd_loader: Optional[Any] = None,
    ) -> None:
        super().__init__(config, V, Xin, cad_model, debug_config, pcd_loader)

        self.updatePriors = True
        self.normals: Optional[NDArray[np.float64]] = None
        self._debug("Init GMM")

        # Initial covariance / precision estimate from global bounding box
        min_xyz: List[NDArray[np.float64]] = []
        max_xyz: List[NDArray[np.float64]] = []
        tvx: List[NDArray[np.float64]] = self.TV.copy()
        tvx.append(self.X)

        for pc in tvx:
            min_xyz.append(np.min(pc, axis=1))
            max_xyz.append(np.max(pc, axis=1))

        min_xyz_arr = np.min(min_xyz, axis=0).reshape(self.dim, 1)
        max_xyz_arr = np.max(max_xyz, axis=0).reshape(self.dim, 1)

        self.Q: PrecisionVector = (
            np.ones((1, self.K), dtype=np.float64)
            * (1.0 / self.sse(min_xyz_arr, max_xyz_arr))
        ).reshape(self.K, 1)
        self.sigma2 = float(1.0 / np.mean(self.Q))
        # Uniform priors, shape (K, 1)
        self.pk: Priors = np.full(
            (self.K, 1),
            1.0 / (self.K * (self.gamma + 1.0)),
            dtype=np.float64,
        )

        self.h: float = float(2.0 / np.mean(self.Q))
        self.beta: float = float(self.gamma / (self.h * (self.gamma + 1.0)))

        # Per-iteration transform history: each entry is ([R_i], [t_i])
        self.T: List[ViewTransforms] = []

        if self.pcd_loader is not None:

            self.visualizer = RerunGMMVisualizer(
                cad_model=self.cad_model,
                point_clouds=self.V,
                initial_centers=self.X,
                initial_Q=self.Q,
                normals=None,
                view_labels=self.pcd_loader.view_labels,
                config=self.config,
                debug_config=self.debug_config,
            )
            self._debug("Rerun visualization initialized successfully")

        if self.T_gt is not None:
            n_obs = len(self.V)
            t_1_init = [
                transformPCDs.homogeneous_transform(self.R[i], self.t[i].reshape(-1))
                for i in range(n_obs // 2)
            ]
            t_2_init = [
                transformPCDs.homogeneous_transform(self.R[i], self.t[i].reshape(-1))
                for i in range(n_obs // 2, n_obs)
            ]
            self.log_error(self.T_gt, t_1_init, t_2_init, 0)



    def e_step(
        self,
        X: Centers3K,
        Q: PrecisionVector,
        TV: List[PointCloud3N],
        pk: Priors,
        beta: float,
    ) -> List[Alpha]:
        """
        Returns:
            alpha: list of correspondence matrices, each of shape (N_i, K)
        """
        alpha_unnorm: List[NDArray[np.float64]] = []

        for cloud in TV:
            # self.sse returns pairwise squared distances between columns:
            # X: (3, K), cloud: (3, N) -> (K, N)
            dist_kn = self.sse(np.asarray(X), np.asarray(cloud))  # (K, N)

            prob_kn = (
                pk * (Q ** 1.5) * np.exp(-0.5 * Q * dist_kn)
            )  # broadcast to (K, N)

            alpha_unnorm.append(prob_kn)

        alpha: List[Alpha] = []
        for prob_kn in alpha_unnorm:
            denom = np.sum(prob_kn, axis=0, keepdims=True) + beta  # (1, N)
            alpha_nk = (prob_kn / denom).T  # (N, K)
            alpha.append(alpha_nk)

        if self.iteration % 10 == 0:
            per_view_alpha = [np.max(alpha_view, axis=1).mean() for alpha_view in alpha]
            global_mean = float(np.mean(per_view_alpha))
            self._debug(f"Free centers: mean max α = {global_mean:.3f}")

        return alpha

    def m_step(
        self,
        X: Centers3K,
        V: List[PointCloud3N],
        Q: PrecisionVector,
        alpha: List[Alpha],
        gamma: float,
        epsilon: float,
    ) -> Tuple[List[PointCloud3N], ViewTransforms, Centers3K, PrecisionVector, Priors]:
        """
        Returns:
            TV: transformed point clouds
            (R, t): per-view rigid transforms
            X: updated centers
            Q: updated precisions
            pk: updated priors
        """
        # Lambda per view: (K, 1)
        lmda: List[NDArray[np.float64]] = [
            np.sum(alpha_i, axis=0, keepdims=True).T.astype(np.float64)
            for alpha_i in alpha
        ]

        # Weighted sums of points: (3, K)
        W: List[NDArray[np.float64]] = [
            (V[i] @ alpha[i]) * Q.T
            for i in range(len(V))
        ]

        # Weighted lambda: (K, 1)
        b: List[NDArray[np.float64]] = [lmda_i * Q for lmda_i in lmda]

        # Weighted means
        mW: List[NDArray[np.float64]] = [
            np.sum(W_i, axis=1, keepdims=True)
            for W_i in W
        ]
        mX: List[NDArray[np.float64]] = [
            X @ b_i
            for b_i in b
        ]

        sum_of_weights: List[float] = [
            float(np.sum(lmda_i * Q))
            for lmda_i in lmda
        ]
        
        # Cross-covariances
        P: List[NDArray[np.float64]] = [
            X @ W[i].T - (mX[i] @ mW[i].T) / sum_of_weights[i]
            for i in range(len(W))
        ]

        # Solve per-view rigid transforms
        R_list: List[Rotation] = []
        t_list: List[Translation] = []

        for i, P_i in enumerate(P):
            U, _, Vt = np.linalg.svd(P_i)
            C = np.eye(3, dtype=np.float64)
            C[2, 2] = np.linalg.det(U @ Vt)
            R_i = U @ C @ Vt
            t_i = ((mX[i] - R_i @ mW[i]) / sum_of_weights[i]).reshape(3)

            R_list.append(R_i)
            t_list.append(t_i)

        # Transform views
        TV: List[PointCloud3N] = [
            R_list[i] @ V[i] + t_list[i].reshape(3, 1)
            for i in range(len(V))
        ]

        # Update centers
        lmda_matrix = np.hstack(lmda)                 # (K, M)
        den = np.sum(lmda_matrix, axis=1, keepdims=True)  # (K, 1)

        x_statistical: List[NDArray[np.float64]] = [
            TV[i] @ alpha[i]
            for i in range(len(TV))
        ]
        x_sum = np.sum(np.stack(x_statistical, axis=0), axis=0)  # (3, K)
        X_new = x_sum / den.T

        # Update precision
        wnormes: List[NDArray[np.float64]] = [
            np.sum(
                alpha[i] * self.sse(np.asarray(X_new), np.asarray(TV[i])).T,
                axis=0,
            )
            for i in range(len(TV))
        ]

        wnormes_sum = np.sum(np.stack(wnormes, axis=0), axis=0).reshape(self.K, 1)
        Q_new: PrecisionVector = (3.0 * den) / (wnormes_sum + 3.0 * den * epsilon)

        pk_new = self.pk
        if self.updatePriors:
            pk_new = den / ((gamma + 1.0) * np.sum(den))

        self.den = den

        return TV, (R_list, t_list), X_new, Q_new, pk_new

    def em_step(self, it: int) -> None:
        X = self.X
        Q = self.Q
        pk = self.pk
        beta = self.beta
        gamma = self.gamma
        epsilon = self.epsilon

        V = self.V
        TV = self.TV

        alpha = self.e_step(X=X, TV=TV, pk=pk, Q=Q, beta=beta)

        TV, (R, t), X, Q, pk = self.m_step(
            X=X,
            V=V,
            Q=Q,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
        )

        self.R = R
        self.t = t

        self.T.append((R, t))

        self.TV = TV
        self.X = X
        self.Q = Q
        self.sigma2 = float(1.0 / np.mean(self.Q))
        self.pk = pk
        self.alpha = alpha

        if self.visualizer is not None:
            self.visualizer.log_em_convergence_stats(
                iteration=self.iteration,
                sigma2=float(np.mean(1.0 / self.Q)),
                Q=self.Q,
                alpha=self.alpha[0],
                centers=np.asarray(self.X),
                TV=None,
                den=self.den,
                a_values=None,
            )

        self._debug(
            f"K={self.K}  sigma={np.sqrt(np.mean(1.0 / self.Q)) * 1000:.1f}mm"
        )