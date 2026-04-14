import rerun as rr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

import os
os.environ["RERUN_FLUSH_TIMEOUT_SEC"] = "10.0"  # Increase timeout
os.environ["RERUN_CHUNK_MAX_BYTES"] = "10485760"  # 10MB chunks

class RerunGMMVisualizer:
    """
    Real-time GMM visualization using Rerun.
    Updated for latest Rerun API.
    """
    
    def __init__(self, cad_model, point_clouds, initial_centers, initial_Q, normals=None, view_labels=None, config=None, debug_config=None):
        """
        Initialize Rerun visualization.
        
        Args:
            cad_model: CAD model instance
            point_clouds: List of point cloud arrays
            initial_centers: Initial GMM centers (K, 3)
            initial_Q: Initial precision matrices (K, 3, 3) or (K,)
        """
        
        
        
        self.view_labels = view_labels
        self.normals=normals

        # Initialize Rerun
        rr.init("GMM_CAD_Registration", spawn=True)
        
        if config is not None:
            import json
            config_text = "=== Config ===\n"
            config_text += json.dumps(config, indent=2, default=str)
            rr.log("config/experiment", rr.TextDocument(config_text), static=True)
        
        if debug_config is not None:
            import json
            debug_text = "=== Debug Config ===\n"
            debug_text += json.dumps(debug_config, indent=2, default=str)
            rr.log("config/debug", rr.TextDocument(debug_text), static=True)

        self.debug_config = debug_config or {}

        self.cad_model = cad_model
        self.point_clouds = point_clouds
        self.iteration = 0
        
        # Log static data once
        self._log_static_data()
        
        # Log initial state
        self.update_gmm_state(initial_centers.T, 0, point_clouds, initial_Q, normals)

    def _debug(self, msg: str) -> None:
        if self.debug_config.get("verbose", True):
            print(msg)

    def _warn(self, msg: str) -> None:
        print(f"[Rerun] Warning: {msg}")

    def _log_static_data(self):
        """Log CAD model and other static elements."""
        if self.cad_model is not None:
            vertices = np.asarray(self.cad_model.mesh.vertices).astype(np.float32)
            triangles = np.asarray(self.cad_model.mesh.triangles).astype(np.uint32)
            
            # Log mesh with correct parameter name
            # Create semi-transparent vertex colors with alpha
            num_vertices = len(vertices)
            # RGBA: light blue with 30% opacity
            vertex_colors = np.full((num_vertices, 4), [150, 150, 200, 25], dtype=np.uint8)
            
            # Log mesh with transparency
            rr.log(
                "cad_model/mesh",
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=triangles,
                    albedo_factor=[0.7, 0.7, 0.8, 0.3]  # RGBA with 30% opacity
                ),
                static=True
            )
            
            # Log bounding box
            bbox = self.cad_model.mesh.get_axis_aligned_bounding_box()
            rr.log(
                "cad_model/bbox",
                rr.Boxes3D(
                    centers=[bbox.get_center()],
                    sizes=[bbox.max_bound - bbox.min_bound],
                    colors=[[0, 255, 0, 50]]
                ),
                static=True
            )
    
    def log_outlier_probabilities(self, TV, alpha_outlier, iteration, threshold=0.5):
        """
        Visualize per-point outlier probabilities in Rerun.

        Args:
            TV: list of point clouds, each (3, N) or (N, 3)
            alpha_outlier: list of (N,) arrays, one per view
            iteration: EM iteration
            threshold: probability threshold for explicit outlier overlay
        """
        import numpy as np
        

        rr.set_time_sequence("iteration", iteration)

        for view_idx, (tv, a_out) in enumerate(zip(TV, alpha_outlier)):
            points = tv.T if tv.shape[0] == 3 else tv
            a_out = np.asarray(a_out).reshape(-1)
            a_out = np.clip(a_out, 0.0, 1.0)

            if len(points) != len(a_out):
                print(f"[log_outlier_probabilities] shape mismatch in view {view_idx}: "
                    f"points={points.shape}, alpha_outlier={a_out.shape}")
                continue

            # Blue (inlier) -> Red (outlier)
            colors = np.zeros((len(points), 3), dtype=np.uint8)
            colors[:, 0] = (255.0 * a_out).astype(np.uint8)          # red
            colors[:, 2] = (255.0 * (1.0 - a_out)).astype(np.uint8)  # blue

            rr.log(
                f"diagnostics/outlier_prob/{self.view_labels[view_idx]}",
                rr.Points3D(
                    positions=points,
                    colors=colors,
                    radii=0.005
                )
            )

            # Explicit high-outlier overlay
            mask = a_out > threshold
            if np.any(mask):
                rr.log(
                    f"diagnostics/outlier_prob_high/{self.view_labels[view_idx]}",
                    rr.Points3D(
                        positions=points[mask],
                        colors=[[255, 255, 0]],   # yellow highlight
                        radii=0.01
                    )
                )
            else:
                # Clear path with empty point cloud if no high outliers this iter
                rr.log(
                    f"diagnostics/outlier_prob_high/{self.view_labels[view_idx]}",
                    rr.Points3D(
                        positions=np.zeros((0, 3), dtype=np.float32),
                        colors=np.zeros((0, 3), dtype=np.uint8),
                        radii=0.01
                    )
                )

            # Optional text summary per view
            # summary = (
            #     f"Iteration {iteration}\n"
            #     f"mean={a_out.mean():.4f}\n"
            #     f"median={np.median(a_out):.4f}\n"
            #     f"95th={np.percentile(a_out,95):.4f}\n"
            #     f">{threshold:.2f} = {(mask.mean()*100):.2f}%"
            # )
            # rr.log(
            #     f"diagnostics/outlier_prob_stats/{self.view_labels[view_idx]}",
            #     rr.TextDocument(summary)
            # )

    def update_gmm_state(self, X, iteration, TV=None, Q=None, normals=None, sigma2=None, a_per_center=None):
        """
        Update GMM visualization in Rerun.
        
        Args:
            X: Current centers (3, K)
            iteration: Current iteration number
            TV: Transformed point clouds (optional)
            Q: Precision matrices (K, 3, 3) or (K,)
        """
        self.iteration = iteration
        rr.set_time_sequence("iteration", iteration)
        
        # Log GMM centers
        centers = X.T if X.shape[0] == 3 else X
        K = centers.shape[0]
        for k in range(K):
            rr.log(
                f"gmm/centers/{k}",
                rr.Points3D(
                    centers[k],
                    colors=[[255, 0, 0]],
                    radii=0.01
                )
            )
            
        # Log covariance ellipsoids
        if Q.ndim == 3 and Q.shape[1] == 3 and Q.shape[2] == 3:
            # Anisotropic covariances (K, 3, 3)
            self._log_anisotropic_covariance_ellipsoids(centers, Q, normals, sigma2, a_per_center)
        else:
            # Isotropic (old behavior)
            self._log_covariance_ellipsoids(centers, Q)
        
        # Log transformed point clouds
        if TV is not None:
            for i, tv in enumerate(TV):
                points = tv.T if tv.shape[0] == 3 else tv
                
                # Color based on view index
                colors = self._get_view_color(i)
                
                rr.log(
                    f"point_clouds/sensor_overlay/{self.view_labels[i]}",
                    rr.Points3D(
                        points,
                        colors=[colors],
                        radii=0.005
                    )
                )
        
        # Log statistics
        # self._log_statistics(centers, Q)
        if self.normals is not None:
            self.log_center_surface_normals(centers, self.cad_model, self.normals, scale=0.06, color=(0,255,0))

    
    def _get_view_color(self, idx):
        label = self.view_labels[idx].lower() if self.view_labels else ""
        if "sensor_1" in label:
            return [255, 100, 100]   # Red for left
        elif "sensor_2" in label:
            return [100, 100, 255]   # Blue for right
        else:
            # Fallback for unlabeled views
            colors = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255]]
            return colors[idx % len(colors)]
    
    def _log_covariance_ellipsoids(self, centers, Q):
        """Isotropic ellipsoids. Q is (K,) scalar precisions."""
        K = len(centers)
        Q_flat = np.asarray(Q).flatten()
        
        all_half_sizes = []
        all_quaternions = []
        
        for k in range(K):
            q = float(Q_flat[k]) if k < len(Q_flat) else 1.0
            sigma = 1.0 / np.sqrt(max(q, 1e-12))
            half_size = sigma * 2.0  # 2-sigma
            all_half_sizes.append([half_size, half_size, half_size])
            all_quaternions.append([0.0, 0.0, 0.0, 1.0])  # identity
        
        all_half_sizes = np.array(all_half_sizes, dtype=np.float32)
        all_quaternions = np.array(all_quaternions, dtype=np.float32)
        
        for k in range(K):
            rr.log(
                f"gmm/ellipsoids/{k}",
                rr.Ellipsoids3D(
                    centers=centers[k],
                    half_sizes=all_half_sizes[k],
                    quaternions=all_quaternions[k],
                    colors=[100, 100, 255, 100],
                )
            )

    def _log_isotropic_ellipsoids(self, centers, sigmas):
        """Fallback: anisotropic Q present but no normals — display as spheres."""
        K = len(centers)
        for k in range(K):
            s = float(sigmas[k]) * 2.0
            rr.log(
                f"gmm/ellipsoids/{k}",
                rr.Ellipsoids3D(
                    centers=centers[k],
                    half_sizes=[s, s, s],
                    quaternions=[0.0, 0.0, 0.0, 1.0],
                    colors=[100, 180, 100, 100],
                )
            )

    def _log_statistics(self, centers, Q):
        """Log statistics as text."""
        K = len(centers)
        
        # Compute statistics
        if Q is not None:
            if Q.ndim == 3:
                precisions = [np.trace(Q[k]) for k in range(K)]
                mean_precision = np.mean(precisions)
                widths = [1.0/np.sqrt(p/3) for p in precisions if p > 0]
            else:
                mean_precision = np.mean(Q)
                widths = [1.0/np.sqrt(q) for q in Q if q > 0]
            
            mean_width = np.mean(widths) if widths else 0.1
        else:
            mean_precision = 0
            mean_width = 0
        
        # Center spread
        center_std = np.std(centers, axis=0)
        
        # Log as text
        stats_text = f"""Iteration: {self.iteration}
Components: {K}
Mean Precision: {mean_precision:.2f}
Mean Width: {mean_width:.3f}m
Center Spread: [{center_std[0]:.3f}, {center_std[1]:.3f}, {center_std[2]:.3f}]m
"""
        
        rr.log(
            "statistics",
            rr.TextDocument(stats_text)
        )
    
    def log_curvature_analysis(self, centers, curvatures, normals):
        """
        Visualize curvature analysis on CAD surface.
        
        Args:
            centers: GMM centers
            curvatures: Curvature values at centers
            normals: Surface normals at centers
        """
        # Color map curvatures
        curv_normalized = (curvatures - np.min(curvatures)) / (np.max(curvatures) - np.min(curvatures) + 1e-8)
        colors = plt.cm.coolwarm(curv_normalized)[:, :3]
        colors = (colors * 255).astype(np.uint8)
        
        rr.log(
            "analysis/curvature_points",
            rr.Points3D(
                centers,
                colors=colors.tolist(),
                radii=0.01
            )
        )
        
        # Log normals as arrows
        rr.log(
            "analysis/normals",
            rr.Arrows3D(
                origins=centers,
                vectors=normals * 0.05,
                colors=[[0, 255, 0]]
            )
        )

    def log_calibration_error(self, T_gt, T_est, iteration):
        """
        Log calibration error metrics to Rerun.
        """
        if T_gt is None or T_est is None:
            return
        
        rr.set_time_sequence("iteration", iteration)
        
        # Compute rotation and translation errors
        R_gt = T_gt[:3, :3]
        R_est = T_est[:3, :3]
        t_gt = T_gt[:3, 3]
        t_est = T_est[:3, 3]
        
        # Rotation error (in degrees)
        R_error = R_gt @ R_est.T
        rot_error_rad = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
        rot_error_deg = np.degrees(rot_error_rad)
        
        # Translation error (in meters)
        trans_error = np.linalg.norm(t_gt - t_est)
        
        # Log scalar metrics - pass values directly
        rr.log("metrics/rotation_error_deg", rr.Scalars(rot_error_deg))
        rr.log("metrics/translation_error_m", rr.Scalars(trans_error))
        
        # Log as text for easy reading
        error_text = f"""Calibration Error (Iteration {iteration})
    Rotation: {rot_error_deg:.2f}°
    Translation: {trans_error:.3f}m
    """
        rr.log("metrics/error_text", rr.TextDocument(error_text))
        
        # Visualize transformations as coordinate frames
        self._log_transform_frames(T_gt, T_est)

    def log_convergence_plot(self, errors_history):
        """
        Log convergence history as a plot.
        """
        if not errors_history:
            return
        
        # Log as time series
        for i, (rot_err, trans_err) in enumerate(zip(
            errors_history['rotation'], 
            errors_history['translation']
        )):
            rr.set_time_sequence("iteration", i)
            rr.log("convergence/rotation_deg", rr.Scalars(rot_err))
            rr.log("convergence/translation_m", rr.Scalars(trans_err))
            
    def _log_transform_frames(self, T_gt, T_est):
        """
        Visualize ground truth vs estimated transformations as coordinate frames.
        """
        # Origin point
        origin = np.array([0, 0, 0])
        
        # Axes length
        axis_length = 0.2
        
        # Ground truth frame (green)
        R_gt = T_gt[:3, :3]
        t_gt = T_gt[:3, 3]
        
        rr.log(
            "calibration/ground_truth_frame",
            rr.Arrows3D(
                origins=[t_gt, t_gt, t_gt],
                vectors=[
                    R_gt[:, 0] * axis_length,  # X axis
                    R_gt[:, 1] * axis_length,  # Y axis
                    R_gt[:, 2] * axis_length   # Z axis
                ],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]
            )
        )
        
        # Estimated frame (blue)
        R_est = T_est[:3, :3]
        t_est = T_est[:3, 3]
        
        rr.log(
            "calibration/estimated_frame",
            rr.Arrows3D(
                origins=[t_est, t_est, t_est],
                vectors=[
                    R_est[:, 0] * axis_length,
                    R_est[:, 1] * axis_length,
                    R_est[:, 2] * axis_length
                ],
                colors=[[200, 100, 100], [100, 200, 100], [100, 100, 200]]
            )
        )

    def log_responsibility_field(self, TV, alpha, iteration):
        """Create spatial heatmap of responsibilities"""
        # Sample the responsibility field on a grid
        all_points = np.hstack([tv for tv in TV])
        
        # Create bounding box
        min_pt = np.min(all_points, axis=1) - 0.1
        max_pt = np.max(all_points, axis=1) + 0.1
        
        # Sample grid (coarse for performance)
        grid_size = 20
        x = np.linspace(min_pt[0], max_pt[0], grid_size)
        y = np.linspace(min_pt[1], max_pt[1], grid_size)
        z = np.linspace(min_pt[2], max_pt[2], grid_size)
        
        # For each grid point, find max responsibility
        grid_points = []
        grid_values = []
        
        for xi in x[::3]:  # Subsample for performance
            for yi in y[::3]:
                for zi in z[::3]:
                    pt = np.array([xi, yi, zi])
                    # Find nearest component
                    dists = np.linalg.norm(X - pt.reshape(3, 1), axis=0)
                    nearest_k = np.argmin(dists)
                    # Get its average responsibility
                    avg_resp = np.mean([alpha[i][:, nearest_k].mean() 
                                    for i in range(len(alpha))])
                    grid_points.append(pt)
                    grid_values.append(avg_resp)
        
        # Log as colored points
        values = np.array(grid_values)
        colors = plt.cm.viridis(values / values.max())[:, :3]
        colors = (colors * 255).astype(np.uint8)
        
        rr.log(
            "analysis/responsibility_field",
            rr.Points3D(
                np.array(grid_points),
                colors=colors.tolist(),
                radii=0.02
            )
        )

    def log_transform_decomposition(self, R, t, iteration):
        """Decompose and visualize rotation"""
        from scipy.spatial.transform import Rotation
        
        for i, R_i in enumerate(R):
            rot = Rotation.from_matrix(R_i)
            euler = rot.as_euler('xyz', degrees=True)
            
            rr.set_time_sequence("iteration", iteration)
            rr.log(f"transforms/{self.view_labels[i]}/roll_deg", rr.Scalars(euler[0]))  # Scalars not Scalar
            rr.log(f"transforms/{self.view_labels[i]}/pitch_deg", rr.Scalars(euler[1]))
            rr.log(f"transforms/{self.view_labels[i]}/yaw_deg", rr.Scalars(euler[2]))
            rr.log(f"transforms/{self.view_labels[i]}/translation_norm", rr.Scalars(np.linalg.norm(t[i])))
                

    def log_component_movements(self, X_before, X_after, iteration):
        """Visualize component movements as arrows"""
        movements = X_after - X_before
        movement_magnitudes = np.linalg.norm(movements, axis=0)
        
        # Only show components that moved significantly
        significant_movers = movement_magnitudes > 0.001  # >1mm
        
        if np.any(significant_movers):
            origins = X_before[:, significant_movers].T
            vectors = movements[:, significant_movers].T * 100  # Scale up for visibility
            
            # Color by movement magnitude
            colors = plt.cm.Reds(movement_magnitudes[significant_movers] / movement_magnitudes.max())
            colors = (colors[:, :3] * 255).astype(np.uint8)
            
            rr.log(
                "debug/component_movements",
                rr.Arrows3D(
                    origins=origins,
                    vectors=vectors,
                    colors=colors.tolist()
                )
            )
            
            # Log text summary
            rr.log(
                "debug/movement_stats",
                rr.TextDocument(f"Iteration {iteration}\n"
                            f"Max movement: {movement_magnitudes.max():.4f}m\n"
                            f"Moving components: {np.sum(significant_movers)}/{len(movement_magnitudes)}")
            )

    def log_component_analysis(self, X, alpha, iteration):
        """Visualize component importance"""
        K = X.shape[1]
        centers = X.T
        
        # Calculate total responsibilities
        total_resp = np.array([
            sum(alpha[i][:, k].sum() for i in range(len(alpha)))
            for k in range(K)
        ])
        
        # Color by responsibility (red=high, blue=low)
        resp_normalized = (total_resp - total_resp.min()) / (total_resp.max() - total_resp.min())
        colors = plt.cm.coolwarm(resp_normalized)[:, :3]
        colors = (colors * 255).astype(np.uint8)
        
        # Size by responsibility
        radii = 0.005 + 0.02 * resp_normalized  # 5mm to 25mm
        
        rr.log(
            "analysis/component_responsibilities",
            rr.Points3D(
                centers,
                colors=colors.tolist(),
                radii=radii.tolist()
            )
        )
        
        # Label outlier components
        low_resp_threshold = np.percentile(total_resp, 20)
        for k in range(K):
            if total_resp[k] < low_resp_threshold:
                rr.log(
                    f"analysis/outliers/component_{k}",
                    rr.Points3D(
                        centers[k:k+1],
                        colors=[[255, 0, 0]],
                        radii=0.03
                    )
                )
                
    def log_sse_metrics(self, TV, X, alpha, iteration):
        """SSE metrics - bulletproof version."""
        try:
            rr.set_time_sequence("iteration", iteration)
            
            K = X.shape[1]
            total_weighted_sse = 0.0
            center_responsibility = np.zeros(K)
            
            for sensor_idx, (tv, alpha_i) in enumerate(zip(TV, alpha)):
                # Force everything to be regular numpy arrays
                tv = np.array(tv)
                alpha_i = np.array(alpha_i)
                X_arr = np.array(X)
                
                N = tv.shape[1]
                
                # Compute squared distances manually
                weighted_sse = 0.0
                for k in range(K):
                    center_k = X_arr[:, k]  # (3,)
                    diffs = tv - center_k.reshape(3, 1)  # (3, N)
                    sq_dists = np.sum(diffs ** 2, axis=0)  # (N,)
                    
                    # Get responsibilities for this center
                    if alpha_i.shape == (K, N):
                        resp_k = alpha_i[k, :]  # (N,)
                    elif alpha_i.shape == (N, K):
                        resp_k = alpha_i[:, k]  # (N,)
                    else:
                        print(f"Unexpected alpha shape: {alpha_i.shape}")
                        continue
                    
                    resp_k = np.array(resp_k).flatten()  # Force to 1D array
                    weighted_sse += np.sum(sq_dists * resp_k)
                    center_responsibility[k] += np.sum(resp_k)
                
                total_weighted_sse += weighted_sse
                rr.log(f"sse/per_sensor/sensor_{sensor_idx}/weighted", 
                    rr.Scalars(float(weighted_sse)))
            
            # Log summary
            rr.log("sse/total/weighted", rr.Scalars(float(total_weighted_sse)))
            num_active = int(np.sum(center_responsibility > 0.1))
            rr.log("sse/centers/num_active", rr.Scalars(num_active))
            
            summary = f"Iteration {iteration}: SSE={total_weighted_sse:.2f}, Active={num_active}/{K}"
            rr.log("sse/summary_text", rr.TextDocument(summary))
            
        except Exception as e:
            import traceback
            print(f"SSE logging failed: {e}")
            traceback.print_exc()


    def log_sse_detailed_per_center(self, TV, X, alpha, iteration, top_n=10):
        """
        Log detailed SSE breakdown for top N centers.
        Useful for debugging specific components.
        
        Args:
            TV: Transformed point clouds
            X: GMM centers (3 x K)
            alpha: Responsibilities
            iteration: Current iteration
            top_n: Number of top/bottom centers to analyze
        """
        rr.set_time_sequence("iteration", iteration)
        
        K = X.shape[1]
        
        # Compute per-center SSE and responsibilities
        center_sse = np.zeros(K)
        center_resp = np.zeros(K)
        
        for sensor_idx, (tv, alpha_i) in enumerate(zip(TV, alpha)):
            N = tv.shape[1]
            
            squared_dists = np.zeros((N, K))
            for k in range(K):
                diff = tv - X[:, k:k+1]
                squared_dists[:, k] = np.sum(diff ** 2, axis=0)
            
            print(f"DEBUG: tv.shape={tv.shape}, N={N}, K={K}")  
            print(f"DEBUG: alpha_i.shape={alpha_i.shape}")
            print(f"DEBUG: squared_dists.shape={squared_dists.shape}")

            # Fix alpha_i dimensions if needed
            if alpha_i.shape[0] == K and alpha_i.shape[1] == N:
                alpha_i_correct = alpha_i.T  # Now (N x K)
            else:
                alpha_i_correct = alpha_i
            
            weighted_dists = squared_dists * alpha_i_correct
            
            for k in range(K):
                center_sse[k] += np.sum(weighted_dists[:, k])
                center_resp[k] += np.sum(alpha_i_correct[:, k])
        
        # Sort by responsibility
        sorted_indices = np.argsort(center_resp)[::-1]
        
        # Log top N centers
        top_text = f"Top {top_n} Centers (Iteration {iteration}):\n"
        top_text += "="*50 + "\n"
        
        for rank, idx in enumerate(sorted_indices[:top_n]):
            top_text += f"Rank {rank+1}: Center {idx}\n"
            top_text += f"  Responsibility: {center_resp[idx]:.2f}\n"
            top_text += f"  SSE: {center_sse[idx]:.2f}\n"
            top_text += f"  Position: [{X[0,idx]:.3f}, {X[1,idx]:.3f}, {X[2,idx]:.3f}]\n"
            
            # Log individual center metric
            rr.log(f"sse/top_centers/rank_{rank+1}/responsibility", 
                rr.Scalars(center_resp[idx]))
            rr.log(f"sse/top_centers/rank_{rank+1}/sse", 
                rr.Scalars(center_sse[idx]))
        
        rr.log("sse/top_centers_text", rr.TextDocument(top_text))
        
        # Log worst N centers (bottom performers)
        worst_text = f"Worst {top_n} Centers (Iteration {iteration}):\n"
        worst_text += "="*50 + "\n"
        
        for rank, idx in enumerate(sorted_indices[-top_n:]):
            worst_text += f"Bottom {rank+1}: Center {idx}\n"
            worst_text += f"  Responsibility: {center_resp[idx]:.2f}\n"
            worst_text += f"  SSE: {center_sse[idx]:.2f}\n"
            worst_text += f"  Position: [{X[0,idx]:.3f}, {X[1,idx]:.3f}, {X[2,idx]:.3f}]\n"
        
        rr.log("sse/worst_centers_text", rr.TextDocument(worst_text))


    def log_sse_convergence_analysis(self, sse_history, iteration):
        """
        Analyze SSE convergence trajectory.
        
        Args:
            sse_history: List of (iteration, weighted_sse) tuples
            iteration: Current iteration
        """
        if len(sse_history) < 2:
            return
        
        rr.set_time_sequence("iteration", iteration)
        
        # Compute recent improvement rate
        recent_window = min(10, len(sse_history))
        recent_sse = [sse for _, sse in sse_history[-recent_window:]]
        
        if len(recent_sse) >= 2:
            initial_recent = recent_sse[0]
            final_recent = recent_sse[-1]
            improvement_rate = (initial_recent - final_recent) / recent_window if initial_recent > 0 else 0
            
            rr.log("sse/convergence/improvement_rate", rr.Scalars(improvement_rate))
            
            # Check for stagnation
            if len(recent_sse) >= 5:
                recent_std = np.std(recent_sse[-5:])
                recent_mean = np.mean(recent_sse[-5:])
                stagnation_ratio = recent_std / (recent_mean + 1e-10)
                
                rr.log("sse/convergence/stagnation_ratio", rr.Scalars(stagnation_ratio))
                
                if stagnation_ratio < 0.001:
                    rr.log("sse/convergence/status", 
                        rr.TextDocument("WARNING: Convergence stagnated!"))
        
        # Overall improvement
        initial_sse = sse_history[0][1]
        current_sse = sse_history[-1][1]
        total_improvement = (initial_sse - current_sse) / initial_sse * 100 if initial_sse > 0 else 0
        
        rr.log("sse/convergence/total_improvement_percent", rr.Scalars(total_improvement))


    def _get_heatmap_colors(self, values):
        """
        Convert normalized values [0, 1] to heatmap colors.
        
        Args:
            values: Array of normalized values
        
        Returns:
            List of [R, G, B] colors (0-255)
        """
        import matplotlib.pyplot as plt
        
        colors = plt.cm.viridis(values)[:, :3]  # Get RGB, drop alpha
        colors_255 = (colors * 255).astype(np.uint8)
        return colors_255.tolist()


    # ============================================================================
    # INTEGRATION EXAMPLE: Add to your EM loop
    # ============================================================================

    def example_em_loop_integration():
        """
        Example showing how to integrate SSE logging in your EM loop.
        
        Replace the relevant section in jgmm_with_cad_constraints().
        """
        
        # In your EM loop (modelgenerator.py, around line 160):
        
        # Track SSE history for convergence analysis
        sse_history = []
        
        for it in range(maxNumIter):
            print("GMM Iteration: ", it)
            
            # ... existing E-step code ...
            # ... existing M-step code ...
            
            # ===== ADD SSE LOGGING HERE =====
            if visualizer is not None:
                # Basic SSE metrics every iteration
                visualizer.log_sse_metrics(TV, X, alpha, it)
                
                # Detailed per-center analysis every 10 iterations
                if it % 10 == 0:
                    visualizer.log_sse_detailed_per_center(TV, X, alpha, it, top_n=10)
                
                # Track SSE for convergence analysis
                total_weighted_sse = sum([
                    np.sum(compute_squared_dists(TV[i], X) * alpha[i])
                    for i in range(len(TV))
                ])
                sse_history.append((it, total_weighted_sse))
                
                # Convergence analysis every 5 iterations
                if it % 5 == 0 and len(sse_history) > 5:
                    visualizer.log_sse_convergence_analysis(sse_history, it)
            # ================================
            
            # ... rest of EM iteration ...


    def _rotation_to_quaternion(self, R):
        """
        Convert rotation matrix to quaternion (x, y, z, w).
        """
        import numpy as np
        from scipy.spatial.transform import Rotation
        
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # Returns [x, y, z, w]
        return quat
    
    def _log_anisotropic_covariance_ellipsoids(self, centers, inv_covariances_shape, normals, sigma2, a_per_center):
        import numpy as np
        from scipy.spatial.transform import Rotation

        K = len(centers)
        all_half_sizes = []
        all_quaternions = []
        all_colors = []
        anisotropy_ratios = []

        for k in range(K):
            nk = np.asarray(normals[k], dtype=np.float64)
            nk = nk / (np.linalg.norm(nk) + 1e-12)
            a_k = a_per_center[k]

            # Covariance eigenvalues for I + a * nnT precision
            sigma_n = np.sqrt(sigma2 / (1.0 + a_k))  # compressed along normal
            sigma_t = np.sqrt(sigma2)                  # free along tangent

            # Build deterministic frame from normal
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(np.dot(nk, ref)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            t1 = ref - np.dot(ref, nk) * nk
            t1 = t1 / (np.linalg.norm(t1) + 1e-12)
            t2 = np.cross(nk, t1)
            t2 = t2 / (np.linalg.norm(t2) + 1e-12)

            R_axes = np.column_stack([nk, t1, t2])
            if np.linalg.det(R_axes) < 0:
                R_axes[:, 2] *= -1.0

            quat = Rotation.from_matrix(R_axes).as_quat()  # [x, y, z, w]

            scale_factor = 2.0
            half_sizes = scale_factor * np.array([sigma_n, sigma_t, sigma_t])

            anisotropy_ratio = sigma_t / (sigma_n + 1e-12)
            anisotropy_ratios.append(anisotropy_ratio)

            all_half_sizes.append(half_sizes.astype(np.float32))
            all_quaternions.append(quat.astype(np.float32))
            all_colors.append(get_anisotropy_color(a_k))

        all_half_sizes = np.asarray(all_half_sizes)
        all_quaternions = np.asarray(all_quaternions)
        all_colors = np.asarray(all_colors)
        anisotropy_ratios = np.asarray(anisotropy_ratios)

        for k in range(K):
            rr.log(
                f"gmm/ellipsoids/{k}",
                rr.Ellipsoids3D(
                    centers=centers[k],
                    half_sizes=all_half_sizes[k],
                    quaternions=all_quaternions[k],
                    colors=all_colors[k],
                )
            )
        

    def _log_arrows3d(self, path, origins, vectors, color=(0, 255, 0), scale=0.05, radii=None):
        """Utility to log a batch of arrows."""
        origins = np.asarray(origins, dtype=np.float32)
        vectors = np.asarray(vectors, dtype=np.float32) * float(scale)
        if radii is None:
            rr.log(path, rr.Arrows3D(origins=origins, vectors=vectors, colors=[list(color)]))
        else:
            rr.log(path, rr.Arrows3D(origins=origins, vectors=vectors, radii=radii, colors=[list(color)]))

    def log_center_surface_normals(self, centers, cad_model, normals, scale=0.06, color=(0, 255, 0)):
        """Log CAD surface normals at GMM centers as arrows."""
        if cad_model is None or centers is None or len(centers) == 0:
            return
        # centers is (K,3)
        self._log_arrows3d("normals/centers_cad", centers, normals, color=color, scale=scale)


    def log_calibration_error_detailed(self, T_gt, T_est, iteration):
        """Log detailed per-axis calibration errors."""
        from scipy.spatial.transform import Rotation
        
        rr.set_time_sequence("iteration", iteration)
        
        # Extract rotations and translations
        R_gt = T_gt[:3, :3]
        R_est = T_est[:3, :3]
        t_gt = T_gt[:3, 3]
        t_est = T_est[:3, 3]
        
        # Per-axis rotation errors (Euler angles)
        rot_gt = Rotation.from_matrix(R_gt)
        rot_est = Rotation.from_matrix(R_est)
        rot_error = rot_gt.inv() * rot_est
        euler_error = rot_error.as_euler('xyz', degrees=False)  # radians
        
        # Per-axis translation errors
        trans_error = t_est - t_gt
        
        # Log per-axis
        rr.log("calibration/rotation/roll_rad", rr.Scalars(abs(euler_error[0])))
        rr.log("calibration/rotation/pitch_rad", rr.Scalars(abs(euler_error[1])))
        rr.log("calibration/rotation/yaw_rad", rr.Scalars(abs(euler_error[2])))
        
        rr.log("calibration/translation/x_m", rr.Scalars(abs(trans_error[0])))
        rr.log("calibration/translation/y_m", rr.Scalars(abs(trans_error[1])))
        rr.log("calibration/translation/z_m", rr.Scalars(abs(trans_error[2])))
        
        # Summary
        error_text = f"""Calibration Error (Iteration {iteration})
    Roll:   {abs(euler_error[0]):.4f} rad ({np.degrees(abs(euler_error[0])):.3f}°)
    Pitch:  {abs(euler_error[1]):.4f} rad ({np.degrees(abs(euler_error[1])):.3f}°)
    Yaw:    {abs(euler_error[2]):.4f} rad ({np.degrees(abs(euler_error[2])):.3f}°)

    X: {abs(trans_error[0]):.4f} m
    Y: {abs(trans_error[1]):.4f} m
    Z: {abs(trans_error[2]):.4f} m

    Paper GMM Target:
    Roll/Pitch/Yaw: 0.0033, 0.0036, 0.0020 rad
    X/Y/Z: 0.015, 0.027, 0.018 m
    """
        rr.log("calibration/detailed_error", rr.TextDocument(error_text))
        
    def log_em_convergence_stats(self, iteration, sigma2, Q, alpha, centers, 
                                TV, den=None, a_values=None, log_likelihood=None, X_prev=None):
        """Log critical EM convergence metrics."""
        rr.set_time_sequence("iteration", iteration)
        
        # 1. Global variance shrinkage
        rr.log("em/sigma2", rr.Scalars(sigma2))
        rr.log("em/sigma_std_dev_m", rr.Scalars(np.sqrt(sigma2)))
        
        # 2. Responsibility strength (critical for detecting collapse)
        max_alpha_per_point = np.max(alpha, axis=1)  # (N,) max across K centers
        mean_max_alpha = np.mean(max_alpha_per_point)
        
        rr.log("em/responsibilities/mean_max_alpha", rr.Scalars(mean_max_alpha))
        rr.log("em/responsibilities/weak_assignments_pct", 
            rr.Scalars(np.sum(max_alpha_per_point < 0.3) / len(max_alpha_per_point) * 100))
        
        # 3. Precision (Q) statistics
        if Q.ndim == 1:  # (K,) scalar precision
            rr.log("em/precision/mean_Q", rr.Scalars(np.mean(Q)))
            rr.log("em/precision/Q_range", rr.Scalars(np.ptp(Q)))
        elif Q.ndim == 2 and Q.shape[1] == 1:  # (K, 1) scalar precision
            Q_flat = Q.flatten()
            rr.log("em/precision/mean_Q", rr.Scalars(np.mean(Q_flat)))
            rr.log("em/precision/Q_range", rr.Scalars(np.ptp(Q_flat)))
        else:  # (K, 3, 3) anisotropic
            traces = np.trace(Q, axis1=1, axis2=2)
            rr.log("em/precision/mean_trace", rr.Scalars(np.mean(traces)))
        # 4. Anisotropy levels (if available)
        if a_values is not None:
            rr.log("em/anisotropy/mean_a", rr.Scalars(np.mean(a_values)))
            rr.log("em/anisotropy/saturated_pct", 
                rr.Scalars(np.sum(a_values > 25) / len(a_values) * 100))
        
        # 5. Effective sample size per center
        # In validate or em_step:

        if den is not None:
            self._debug(f"Den stats: min={np.min(den):.2e}, max={np.max(den):.2e}, mean={np.mean(den):.2e}")

            rr.log("em/effective_samples/mean", rr.Scalars(np.mean(den)))
            rr.log("em/effective_samples/min", rr.Scalars(np.min(den)))
            
            # Dead centers (no responsibilities)
            dead_centers = np.sum(den < 0.1)
            rr.log("em/centers/dead_count", rr.Scalars(dead_centers))
        
        # 6. Distance to closest center
        if TV is not None and len(TV) > 0:
            # Sample 1000 random points for efficiency
            sample_tv = TV[0][:, np.random.choice(TV[0].shape[1], 
                                                min(1000, TV[0].shape[1]), 
                                                replace=False)]
            
            dists = np.sqrt(np.sum((sample_tv[:, :, None] - centers[:, None, :])**2, axis=0))
            min_dists = np.min(dists, axis=1)
            
            rr.log("em/distances/mean_to_closest_center_m", 
                rr.Scalars(np.mean(min_dists)))
            rr.log("em/distances/max_to_closest_center_m", 
                rr.Scalars(np.max(min_dists)))
        
        delta_ll = 0
        if log_likelihood is not None:
            rr.log("em/log_likelihood", rr.Scalars(log_likelihood))
            
            # Convergence criterion
            if hasattr(self, 'prev_log_likelihood'):
                delta_ll = log_likelihood - self.prev_log_likelihood
                rr.log("em/convergence/delta_log_likelihood", rr.Scalars(delta_ll))
                
                # Flag if LL decreases (should never happen!)
                if delta_ll < 0:
                    rr.log("em/convergence/WARNING", rr.TextDocument("LOG-LIKELIHOOD DECREASED!"))
            
            self.prev_log_likelihood = log_likelihood
        
        # 2. CENTER MOVEMENT
        if X_prev is not None:
            displacements = np.linalg.norm(centers - X_prev, axis=0)
            rr.log("em/convergence/mean_displacement_m", rr.Scalars(np.mean(displacements)))
            rr.log("em/convergence/max_displacement_m", rr.Scalars(np.max(displacements)))
            
            # Convergence indicator: movement should → 0
            converged = np.mean(displacements) < 1e-4
            if converged:
                rr.log("em/convergence/status", rr.TextDocument("✓ CONVERGED"))

        
        # ===== ENHANCED SUMMARY =====
        status_text = f"""EM Status (Iteration {iteration})
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        
        if log_likelihood is not None:
            status_text += f"Log-likelihood: {log_likelihood:.2f}"
            if hasattr(self, 'prev_log_likelihood'):
                status_text += f" (Δ={delta_ll:.4f})\n"
            else:
                status_text += "\n"
        
        status_text += f"Sigma²: {sigma2:.6f} (σ = {np.sqrt(sigma2):.4f} m)\n"
        status_text += f"Mean max alpha: {mean_max_alpha:.3f} {'✓' if mean_max_alpha > 0.3 else 'WEAK'}\n"
        
        if X_prev is not None:
            status_text += f"Mean displacement: {np.mean(displacements):.4f}m\n"
        
        if den is not None:
            status_text += f"Dead centers: {dead_centers}/{len(den)}\n"
        if a_values is not None:
            status_text += f"Mean anisotropy: {np.mean(a_values):.1f}\n"
        
        rr.log("em/status_summary", rr.TextDocument(status_text))

    
    def log_raw_sensor_data(self, points, sensor_name, color):
        """Log raw sensor data (before any transforms)"""
        rr.log(
            f"raw_sensors/{sensor_name}",
            rr.Points3D(
                points,
                colors=[color],
                radii=0.005
            ),
            static=True
        )

    def _figure_to_array(self, fig):
        """Convert matplotlib figure to numpy array (macOS compatible)."""
        # Save to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to numpy array
        img = Image.open(buf)
        img_array = np.array(img)
        buf.close()
        
        return img_array
    
    # DEPRECATED: This method causes issues on macOS due to matplotlib backend conflicts
    # Use _log_statistics_text_only instead
    def _log_curvature_statistics_DEPRECATED(self, curvatures, a_values):
        """Log statistical plots for debugging (macOS compatible)."""
        
        # Create histogram of curvatures
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Curvature histogram
        axes[0, 0].hist(curvatures, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Curvature Distribution')
        axes[0, 0].set_xlabel('Curvature')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].axvline(np.mean(curvatures), color='red', linestyle='--', label=f'Mean: {np.mean(curvatures):.4f}')
        axes[0, 0].legend()
        
        # a_values histogram
        axes[0, 1].hist(a_values, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_title('a_values Distribution')
        axes[0, 1].set_xlabel('a_value')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].axvline(np.mean(a_values), color='red', linestyle='--', label=f'Mean: {np.mean(a_values):.1f}')
        axes[0, 1].legend()
        
        # Scatter plot: curvature vs a_values
        axes[1, 0].scatter(curvatures, a_values, alpha=0.5)
        axes[1, 0].set_xlabel('Curvature')
        axes[1, 0].set_ylabel('a_value')
        axes[1, 0].set_title('Curvature vs a_value Mapping')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plots
        axes[1, 1].boxplot([curvatures, a_values / np.max(a_values)], 
                           labels=['Curvature', 'a_values (normalized)'])
        axes[1, 1].set_title('Distribution Comparison')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to image using the fixed method
        image = self._figure_to_array(fig)
        
        rr.log(
            "debug/statistics",
            rr.Image(image)
        )
        
        plt.close(fig)
        
        # Log text statistics
        stats_text = f"""
Curvature Statistics:
  Range: [{np.min(curvatures):.6f}, {np.max(curvatures):.6f}]
  Mean: {np.mean(curvatures):.6f}
  Std: {np.std(curvatures):.6f}
  Percentiles [10, 25, 50, 75, 90]: {np.percentile(curvatures, [10, 25, 50, 75, 90])}

a_values Statistics:
  Range: [{np.min(a_values):.1f}, {np.max(a_values):.1f}]
  Mean: {np.mean(a_values):.1f}
  Std: {np.std(a_values):.1f}
  Distribution:
    Near 0 (curved): {np.sum(a_values < 5)} centers ({100*np.sum(a_values < 5)/len(a_values):.1f}%)
    Mid [5-25]: {np.sum((a_values >= 5) & (a_values <= 25))} centers ({100*np.sum((a_values >= 5) & (a_values <= 25))/len(a_values):.1f}%)
    Near max (flat): {np.sum(a_values > 25)} centers ({100*np.sum(a_values > 25)/len(a_values):.1f}%)
"""
        
        rr.log(
            "debug/statistics_text",
            rr.TextDocument(stats_text)
        )
    
    # DEPRECATED: This method causes issues on macOS due to matplotlib backend conflicts
    def _log_colorbar_DEPRECATED(self, values, label, cmap_name='viridis'):
        """Create and log a colorbar legend (macOS compatible)."""
        fig, ax = plt.subplots(figsize=(6, 1))
        fig.subplots_adjust(bottom=0.5)
        
        cmap = cm.get_cmap(cmap_name)
        norm = mcolors.Normalize(vmin=np.min(values), vmax=np.max(values))
        
        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                         cax=ax, orientation='horizontal', label=label)
        
        # Convert to image using the fixed method
        image = self._figure_to_array(fig)
        
        rr.log(f"debug/colorbar_{label}", rr.Image(image))
        plt.close(fig)

    def visualize_curvature_debug(self, centers, normals, curvatures, a_values):
        """
        Comprehensive curvature debugging visualization.
        NOW WITH: Curvature visualized directly on CAD surface!
        """
        
        # 1. MOST USEFUL: Visualize curvature on CAD mesh surface
        if self.cad_model is not None:
            self._visualize_curvature_on_mesh(curvatures, centers)
        
        # 2. Visualize centers colored by curvature (less useful, but kept for reference)
        curv_norm = (curvatures - np.min(curvatures)) / (np.max(curvatures) - np.min(curvatures) + 1e-10)
        
        colors_curv = np.zeros((len(centers), 3))
        colors_curv[:, 0] = curv_norm  # Red channel for high curvature
        colors_curv[:, 2] = 1 - curv_norm  # Blue channel for low curvature
        
        rr.log(
            "debug/centers_curvature",
            rr.Points3D(
                positions=centers,
                colors=colors_curv,
                radii=0.005
            )
        )
        
        # 3. Visualize centers colored by a_values
        a_norm = a_values / (np.max(a_values) + 1e-10)
        
        colors_a = np.zeros((len(centers), 3))
        colors_a[:, 1] = a_norm  # Green channel for high a_values
        colors_a[:, 0] = 1 - a_norm  # Red channel for low a_values
        
        rr.log(
            "debug/centers_a_values",
            rr.Points3D(
                positions=centers,
                colors=colors_a,
                radii=0.005
            )
        )
        
        # 4. Visualize normals
        arrow_origins = centers
        arrow_vectors = normals * 0.02
        
        rr.log(
            "debug/normals",
            rr.Arrows3D(
                origins=arrow_origins,
                vectors=arrow_vectors,
                colors=[[0, 1, 0]] * len(centers),
                radii=0.001
            )
        )
        
        # 5. Log statistics as text (no matplotlib)
        self._log_statistics_text_only(curvatures, a_values)
        
        # 6. Visualize sample ellipsoids for extreme values
        self._visualize_sample_ellipsoids(centers, normals, a_values, curvatures)
    
    def _visualize_curvature_on_mesh(self, curvatures, centers):
        """
        Visualize curvature directly on the CAD mesh surface.
        Uses DENSE SURFACE SAMPLING for high-resolution visualization.
        """
        from cad_curvature import CADCurvatureComputer
        
        try:
            # Sample dense points uniformly on the CAD surface
            # More points = better resolution, but slower
            n_sample_points = 100 # Adjust based on your mesh size
            
            print(f"  Sampling {n_sample_points} points on CAD surface for curvature...")
            sampled_pcd = self.cad_model.mesh.sample_points_uniformly(n_sample_points)
            sampled_points = np.asarray(sampled_pcd.points)
            
            # Compute curvature at ALL sampled points
            computer = CADCurvatureComputer(self.cad_model)
            normals, point_curvatures = computer.compute_exact_properties(sampled_points)
            
            print(f"  Computed curvatures at {len(sampled_points)} sampled points")
            
            # Handle NaN values (replace with median)
            nan_mask = np.isnan(point_curvatures)
            if np.any(nan_mask):
                median_curv = np.nanmedian(point_curvatures)
                point_curvatures[nan_mask] = median_curv
                print(f"Replaced {np.sum(nan_mask)} NaN curvatures with median value {median_curv:.6f}")
            
            # Normalize curvatures for coloring
            curv_min = np.percentile(point_curvatures, 5)
            curv_max = np.percentile(point_curvatures, 95)
            
            curv_norm = np.clip(
                (point_curvatures - curv_min) / (curv_max - curv_min + 1e-10),
                0, 1
            )
            
            # Create colors for point cloud
            point_colors = self._curvature_to_rgb(curv_norm)
            
            # DIAGNOSTIC: Print what we're sending to Rerun
            print(f"\n=== SENDING TO RERUN ===")
            print(f"Sampled points: {sampled_points.shape}")
            print(f"Point colors: {point_colors.shape}, dtype: {point_colors.dtype}")
            print(f"  R: [{point_colors[:,0].min()}, {point_colors[:,0].max()}]")
            print(f"  G: [{point_colors[:,1].min()}, {point_colors[:,1].max()}]")
            print(f"  B: [{point_colors[:,2].min()}, {point_colors[:,2].max()}]")
            
            # Count white vertices
            white_count = np.sum(np.all(point_colors == 255, axis=1))
            black_count = np.sum(np.all(point_colors == 0, axis=1))
            print(f"  White points (255,255,255): {white_count}")
            print(f"  Black points (0,0,0): {black_count}")
            print(f"=========================\n")
            
            # Log as a DENSE POINT CLOUD (not mesh)
            # This will look much better than sparse mesh vertices!
            rr.log(
                "debug/cad_curvature_surface",
                rr.Points3D(
                    positions=sampled_points,
                    colors=point_colors,
                    radii=0.002  # Small radius for dense coverage
                )
            )
            
            # Also create a legend/colorbar as text
            legend_text = f"""
CURVATURE VISUALIZATION ON CAD SURFACE:

Visualization Method:
  • {n_sample_points} points uniformly sampled on surface
  • Dense point cloud (not sparse mesh vertices)
  • High resolution curvature field

Color Mapping:
  🔵 Blue   = Flat surfaces (low curvature)
  🟢 Green  = Slightly curved
  🟡 Yellow = Moderately curved  
  🔴 Red    = Highly curved (edges, corners)

Curvature Range:
  Min: {curv_min:.6f}
  Max: {curv_max:.6f}

Quality:
  Sampled points: {len(sampled_points)}
  NaN curvatures fixed: {np.sum(nan_mask)}
  {f"{100*np.sum(nan_mask)/len(sampled_points):.1f}% of points had invalid curvature!" if np.sum(nan_mask) > 0 else "✓ All curvatures computed successfully"}
  
This shows high-resolution curvature distribution
on your CAD model surface, independent of mesh poly count!
"""
            
            rr.log(
                "debug/curvature_legend",
                rr.TextDocument(legend_text)
            )
            
            print("Curvature visualized on CAD surface with dense sampling!")
            
        except Exception as e:
            print(f"Could not visualize curvature on surface: {e}")
            import traceback
            traceback.print_exc()
            print("   Falling back to point-based visualization")
    
    
    def _curvature_to_rgb(self, normalized_curvatures):
        """
        Convert normalized curvatures [0, 1] to RGB colors.
        Blue (flat) -> Cyan -> Green -> Yellow -> Red (curved)
        """
        colors = np.zeros((len(normalized_curvatures), 3), dtype=np.uint8)
        
        for i, val in enumerate(normalized_curvatures):
            if val < 0.25:
                # Blue to Cyan
                t = val / 0.25
                colors[i] = [0, int(255*t), 255]
            elif val < 0.5:
                # Cyan to Green
                t = (val - 0.25) / 0.25
                colors[i] = [0, 255, int(255*(1-t))]
            elif val < 0.75:
                # Green to Yellow
                t = (val - 0.5) / 0.25
                colors[i] = [int(255*t), 255, 0]
            else:
                # Yellow to Red
                t = (val - 0.75) / 0.25
                colors[i] = [255, int(255*(1-t)), 0]
        
        return colors

    def visualize_normals(self, TV, normals, subsample=20,iteration=0):
        """
        Visualize point cloud normals in Rerun
        
        Args:
            TV: List of point clouds [(3, N), ...]
            normals: List of (N, 3) normal arrays
            subsample: Show every Nth normal (avoid clutter)
        """
        
        rr.set_time_sequence("iteration", iteration)
        
        print(f"\n{'='*70}")
        print(f"Visualizing Normals (subsample={subsample})")
        print(f"{'='*70}")
        
        for view_id, (pc, norms) in enumerate(zip(TV, normals)):
            points = pc.T  # (N, 3)
            
            # Subsample
            indices = np.arange(0, len(points), subsample)
            vis_points = points[indices]
            vis_normals = norms[indices]
            
            n_vis = len(vis_points)
            
            # Log points
            rr.log(
                f"normals/{self.view_labels[view_id]}/points",
                rr.Points3D(
                    positions=vis_points,
                    radii=0.003,
                    colors=[150, 150, 255]
                )
            )
            
            # Log normals as arrows
            normal_length = 0.02  # 20mm arrows
            
            rr.log(
                f"normals/{self.view_labels[view_id]}/arrows",
                rr.Arrows3D(
                    origins=vis_points,
                    vectors=vis_normals * normal_length,
                    colors=[255, 150, 100],
                    radii=0.001
                )
            )
            
            print(f"  View {self.view_labels[view_id]}: {n_vis} normals visualized")
        
        print(f"\n✓ Check Rerun: normals/view_*/points and normals/view_*/arrows")
    
    def _log_statistics_text_only(self, curvatures, a_values):
        """Log statistics without matplotlib - macOS safe version."""
        
        # Compute statistics
        curv_stats = {
            'min': np.min(curvatures),
            'max': np.max(curvatures),
            'mean': np.mean(curvatures),
            'std': np.std(curvatures),
            'p10': np.percentile(curvatures, 10),
            'p50': np.percentile(curvatures, 50),
            'p90': np.percentile(curvatures, 90)
        }
        
        a_stats = {
            'min': np.min(a_values),
            'max': np.max(a_values),
            'mean': np.mean(a_values),
            'std': np.std(a_values),
            'low': np.sum(a_values < 5),
            'mid': np.sum((a_values >= 5) & (a_values <= 25)),
            'high': np.sum(a_values > 25)
        }
        
        stats_text = f"""
CURVATURE STATISTICS:
  Range: [{curv_stats['min']:.6f}, {curv_stats['max']:.6f}]
  Mean ± Std: {curv_stats['mean']:.6f} ± {curv_stats['std']:.6f}
  Percentiles [10, 50, 90]: [{curv_stats['p10']:.6f}, {curv_stats['p50']:.6f}, {curv_stats['p90']:.6f}]

A_VALUES STATISTICS:
  Range: [{a_stats['min']:.1f}, {a_stats['max']:.1f}]
  Mean ± Std: {a_stats['mean']:.1f} ± {a_stats['std']:.1f}
  
  Distribution:
    Low (0-5, highly curved): {a_stats['low']} centers ({100*a_stats['low']/len(a_values):.1f}%)
    Mid (5-25, moderate): {a_stats['mid']} centers ({100*a_stats['mid']/len(a_values):.1f}%)
    High (25-30, flat): {a_stats['high']} centers ({100*a_stats['high']/len(a_values):.1f}%)

DIAGNOSTIC:
  Curvature variance: {curv_stats['std']**2:.6f}
  A_values variance: {a_stats['std']**2:.2f}
  
  {"WARNING: Low curvature variation!" if curv_stats['std'] < 0.01 else "✓ Good curvature variation"}
  {"WARNING: Uniform a_values!" if a_stats['std'] < 2.0 else "✓ Good a_value distribution"}
"""
        
        rr.log("debug/statistics", rr.TextDocument(stats_text))
    
    def _log_statistics_text(self, curvatures, a_values):
        """Log statistics without matplotlib."""
        
        # Compute statistics
        curv_stats = {
            'min': np.min(curvatures),
            'max': np.max(curvatures),
            'mean': np.mean(curvatures),
            'std': np.std(curvatures),
            'p10': np.percentile(curvatures, 10),
            'p50': np.percentile(curvatures, 50),
            'p90': np.percentile(curvatures, 90)
        }
        
        a_stats = {
            'min': np.min(a_values),
            'max': np.max(a_values),
            'mean': np.mean(a_values),
            'std': np.std(a_values),
            'low': np.sum(a_values < 5),
            'mid': np.sum((a_values >= 5) & (a_values <= 25)),
            'high': np.sum(a_values > 25)
        }
        
        stats_text = f"""
CURVATURE STATISTICS:
  Range: [{curv_stats['min']:.6f}, {curv_stats['max']:.6f}]
  Mean ± Std: {curv_stats['mean']:.6f} ± {curv_stats['std']:.6f}
  Percentiles [10, 50, 90]: [{curv_stats['p10']:.6f}, {curv_stats['p50']:.6f}, {curv_stats['p90']:.6f}]

A_VALUES STATISTICS:
  Range: [{a_stats['min']:.1f}, {a_stats['max']:.1f}]
  Mean ± Std: {a_stats['mean']:.1f} ± {a_stats['std']:.1f}
  
  Distribution:
    Low (0-5, highly curved): {a_stats['low']} centers ({100*a_stats['low']/len(a_values):.1f}%)
    Mid (5-25, moderate): {a_stats['mid']} centers ({100*a_stats['mid']/len(a_values):.1f}%)
    High (25-30, flat): {a_stats['high']} centers ({100*a_stats['high']/len(a_values):.1f}%)

DIAGNOSTIC:
  Curvature variance: {curv_stats['std']**2:.6f}
  A_values variance: {a_stats['std']**2:.2f}
  
  {"WARNING: Low curvature variation!" if curv_stats['std'] < 0.01 else "✓ Good curvature variation"}
  {"WARNING: Uniform a_values!" if a_stats['std'] < 2.0 else "✓ Good a_value distribution"}
"""
        
        rr.log("debug/statistics", rr.TextDocument(stats_text))
    
    def _visualize_sample_ellipsoids(self, centers, normals, a_values, curvatures, sigma2=0.01):
        """Visualize ellipsoids for extreme values."""
        
        # Find extremes
        indices = {
            'min_curv': np.argmin(curvatures),
            'max_curv': np.argmax(curvatures),
            'min_a': np.argmin(a_values),
            'max_a': np.argmax(a_values),
            'median': len(centers) // 2
        }
        
        colors = {
            'min_curv': [0, 0, 1],  # Blue
            'max_curv': [1, 0, 0],  # Red
            'min_a': [1, 0, 1],     # Magenta
            'max_a': [0, 1, 0],     # Green
            'median': [0.5, 0.5, 0.5]  # Gray
        }
        
        for name, idx in indices.items():
            center = centers[idx]
            normal = normals[idx]
            a = a_values[idx]
            
            # Build covariance
            n = normal.reshape(3, 1)
            nn_T = n @ n.T
            inv_cov = (a * nn_T + np.eye(3)) / sigma2
            cov = np.linalg.inv(inv_cov)
            
            # Get principal axes
            eigvals, eigvecs = np.linalg.eigh(cov)
            
            # Create ellipsoid as stretched sphere
            u = np.linspace(0, 2*np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Scale by eigenvalues (3-sigma)
            scale = 3 * np.sqrt(eigvals)
            points = np.stack([
                x.flatten() * scale[0],
                y.flatten() * scale[1],
                z.flatten() * scale[2]
            ], axis=1)
            
            # Rotate and translate
            points = (eigvecs @ points.T).T + center
            
            rr.log(
                f"debug/ellipsoids/{name}",
                rr.Points3D(
                    positions=points,
                    colors=[colors[name]] * len(points),
                    radii=0.0005
                )
            )
            
            # Add label
            rr.log(
                f"debug/ellipsoids/{name}_label",
                rr.TextDocument(
                    f"{name}:\n"
                    f"  Curvature: {curvatures[idx]:.4f}\n"
                    f"  a_value: {a:.1f}\n"
                    f"  Eigenvalue ratio: {eigvals[2]/eigvals[0]:.1f}:1"
                )
            )
    
    def _detect_interior_points(self, centers, projected_centers, normals):
        """
        Detect which centers are on the interior of the surface.
        
        Method: Check if the vector from projected point to center
        points in the opposite direction of the surface normal.
        
        Args:
            centers: Original centers (K, 3)
            projected_centers: Projected centers on surface (K, 3)
            normals: Surface normals at projected points (K, 3)
        
        Returns:
            interior_mask: Boolean array (K,) - True if interior
        """
        # Vector from surface to center
        to_center = centers - projected_centers  # (K, 3)
        
        # Normalize
        to_center_norm = to_center / (np.linalg.norm(to_center, axis=1, keepdims=True) + 1e-10)
        
        # Dot product with normal
        # If negative, center is on interior side (opposite to normal)
        dot_products = np.sum(to_center_norm * normals, axis=1)  # (K,)
        
        # Interior if dot product < 0 (pointing inward)
        interior_mask = dot_products < -0.1  # Small threshold for numerical stability
        
        return interior_mask
 
    def log_pointcloud_surface_distances(self, TV, cad_model, iteration):
        """
        Log distance of all point cloud points from CAD surface.
        
        This is the KEY metric for calibration quality:
        - If calibration is good, points should be close to surface
        - Mean distance should decrease over iterations
        - Final distance indicates calibration accuracy
        
        Args:
            TV: List of transformed point clouds, each (3, N)
            iteration: Current iteration number
        """
        if cad_model is None:
            return

        
        # Collect all points from all views
        all_points = []
        view_points = []
        view_names = []
        
        for i, tv in enumerate(TV):
            points = tv.T  # (N, 3)
            
            all_points.append(points)
            view_points.append(len(points))
            view_names.append(f"view_{i}")
        
        all_points = np.vstack(all_points)  # (N_total, 3)
        N_total = len(all_points)
        
        # Compute distances to surface
        _, distances = cad_model.project_points_to_surface_exact(all_points)
        distances_mm = distances * 1000
        
        # Overall statistics
        mean_dist = np.mean(distances_mm)
        median_dist = np.median(distances_mm)
        max_dist = np.max(distances_mm)
        std_dist = np.std(distances_mm)
        percentile_95 = np.percentile(distances_mm, 95)
        
        # Distribution bins
        within_5mm = np.sum(distances_mm < 5)
        within_10mm = np.sum(distances_mm < 10)
        within_20mm = np.sum(distances_mm < 20)
        over_20mm = np.sum(distances_mm >= 20)
        
        # Per-view statistics
        per_view_stats = []
        start_idx = 0
        for i, n_points in enumerate(view_points):
            end_idx = start_idx + n_points
            view_distances = distances_mm[start_idx:end_idx]
            
            per_view_stats.append({
                'view': i,
                'n_points': n_points,
                'mean': np.mean(view_distances),
                'median': np.median(view_distances),
                'max': np.max(view_distances),
                'within_10mm': np.sum(view_distances < 10)
            })
            
            start_idx = end_idx
        
        # Rerun logging (every iteration)
        rr.set_time_sequence("iteration", iteration)
        
        # Overall statistics
        rr.log("pointcloud_fit/mean_mm", rr.Scalars(mean_dist))
        rr.log("pointcloud_fit/median_mm", rr.Scalars(median_dist))
        rr.log("pointcloud_fit/max_mm", rr.Scalars(max_dist))
        rr.log("pointcloud_fit/std_mm", rr.Scalars(std_dist))
        rr.log("pointcloud_fit/percentile_95_mm", rr.Scalars(percentile_95))
        
        # Quality metrics (percentage within tolerance)
        rr.log("pointcloud_fit/within_5mm_percent", 
            rr.Scalars(within_5mm/N_total*100))
        rr.log("pointcloud_fit/within_10mm_percent", 
            rr.Scalars(within_10mm/N_total*100))
        rr.log("pointcloud_fit/within_20mm_percent", 
            rr.Scalars(within_20mm/N_total*100))
        
        # Per-view metrics
        for stat in per_view_stats:
            rr.log(f"pointcloud_fit/per_view/view_{stat['view']}/mean_mm", 
                rr.Scalars(stat['mean']))
            rr.log(f"pointcloud_fit/per_view/view_{stat['view']}/within_10mm_percent",
                rr.Scalars(stat['within_10mm']/stat['n_points']*100))
        
        # Detailed text summary
        summary = f"""Point Cloud Surface Fit (Iteration {iteration})

Overall Quality:
Total points:    {N_total}
Mean distance:   {mean_dist:.2f}mm
Median distance: {median_dist:.2f}mm
Max distance:    {max_dist:.2f}mm
StdDev:          {std_dist:.2f}mm
95th percentile: {percentile_95:.2f}mm

Fit Quality:
Within  5mm: {within_5mm:4d}/{N_total} ({within_5mm/N_total*100:.1f}%)
Within 10mm: {within_10mm:4d}/{N_total} ({within_10mm/N_total*100:.1f}%)
Within 20mm: {within_20mm:4d}/{N_total} ({within_20mm/N_total*100:.1f}%)
Over   20mm: {over_20mm:4d}/{N_total} ({over_20mm/N_total*100:.1f}%)

Per-View Breakdown:
"""
        for stat in per_view_stats:
            summary += f"""  View {stat['view']} ({stat['n_points']} points):
    Mean: {stat['mean']:.2f}mm, Median: {stat['median']:.2f}mm, Max: {stat['max']:.2f}mm
    Within 10mm: {stat['within_10mm']}/{stat['n_points']} ({stat['within_10mm']/stat['n_points']*100:.1f}%)
"""
        
        summary += f"""
Target Quality Metrics:
- Mean < 10mm: {'✓' if mean_dist < 10 else '✗'}
- 90% within 10mm: {'✓' if within_10mm/N_total > 0.9 else '✗'}
- Max < 50mm: {'✓' if max_dist < 50 else '✗'}
"""
        
        rr.log("pointcloud_fit/summary", rr.TextDocument(summary))

    def log_pointcloud_distances_visual(self, TV, cad_model, iteration):
        """
        Visual representation of point cloud fit in Rerun.
        
        Colors points by distance from surface:
        - Green:  < 5mm (excellent)
        - Yellow: 5-10mm (good)
        - Orange: 10-20mm (acceptable)
        - Red:    > 20mm (poor)
        """
        if cad_model is None:
            return
        
        
        rr.set_time_sequence("iteration", iteration)
        
        # Process each view
        for i, tv in enumerate(TV):
            points = tv.T  # (N, 3)
            
            # Get distances
            _, distances = cad_model.project_points_to_surface_exact(points)
            distances_mm = distances * 1000
            
            # Assign colors
            colors = np.zeros((len(points), 3), dtype=np.uint8)
            colors[distances_mm < 5] = [0, 255, 0]      # Green
            colors[(distances_mm >= 5) & (distances_mm < 10)] = [255, 255, 0]   # Yellow
            colors[(distances_mm >= 10) & (distances_mm < 20)] = [255, 165, 0]  # Orange
            colors[distances_mm >= 20] = [255, 0, 0]    # Red
            
            # Log colored points
            rr.log(
                f"point_clouds/distance_overlay/view_{i}_colored_by_distance",
                rr.Points3D(
                    points,
                    colors=colors,
                    radii=0.005
                )
            )
            
    def log_final_calibrated_scans(self, pcd_loader, R, t, scan_limit=3):
        """
        Log full calibrated point clouds after optimization completes.
        Shows the final registration quality across all views.
        """
        if pcd_loader is None:
            return
        
        self._debug("\n=== Logging Final Calibrated Scans ===")
        
        # Log each sensor's full scans with final calibration
        for sensor_idx in range(2):
            all_scans = pcd_loader.get_sensor_scans_full(sensor_idx)
            n_scans = len(all_scans)
            
            for scan_idx, scan in enumerate(all_scans):
                view_idx = sensor_idx * n_scans + scan_idx  # stacked layout
                
                if view_idx >= len(R):
                    self._warn(f"view_idx {view_idx} out of range, skipping")
                    continue
                
                if scan_idx >= scan_limit:
                    continue

                points = np.asarray(scan.points)
                R_view = R[view_idx]
                t_view = t[view_idx].flatten()
                points_transformed = (R_view @ points.T).T + t_view
                
                color = [255, 100, 100] if sensor_idx == 0 else [100, 100, 255]
                rr.log(
                    f"calibrated_scans/sensor_{sensor_idx}/scan_{scan_idx:03d}",
                    rr.Points3D(points_transformed, colors=[color], radii=0.005)
                )

            self._debug(f"  Sensor {sensor_idx}: Logged {scan_limit} calibrated scans")


def _safe_normalize(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, np.maximum(n, eps))

def compute_squared_dists(tv, X):
        """Helper: compute squared distances for SSE."""
        K = X.shape[1]
        N = tv.shape[1]
        squared_dists = np.zeros((N, K))
        for k in range(K):
            diff = tv - X[:, k:k+1]
            squared_dists[:, k] = np.sum(diff ** 2, axis=0)
        return squared_dists


def get_anisotropy_color(a_k, a_min=0, a_max=30):
    """
    Color by anisotropy value:
    - Blue: isotropic (a_k ≈ 0) - sphere-like, edges/corners
    - Green: moderate (a_k ≈ 10-15)
    - Yellow: high (a_k ≈ 20-25)
    - Red: maximum (a_k ≈ 30) - pancake-like, flat faces
    """
    t = np.clip((a_k - a_min) / (a_max - a_min), 0, 1)
    
    if t < 0.33:
        # Blue → Green
        s = t / 0.33
        return [0, int(255 * s), int(255 * (1 - s)), 180]
    elif t < 0.66:
        # Green → Yellow
        s = (t - 0.33) / 0.33
        return [int(255 * s), 255, 0, 180]
    else:
        # Yellow → Red
        s = (t - 0.66) / 0.34
        return [255, int(255 * (1 - s)), 0, 180]
    
def get_anisotropy_color(a_k, a_min=0, a_max=30):
    import matplotlib.pyplot as plt
    t = float(np.clip((a_k - a_min) / (a_max - a_min), 0, 1))
    r, g, b, _ = plt.cm.turbo(t)
    return [int(r*255), int(g*255), int(b*255), 180]