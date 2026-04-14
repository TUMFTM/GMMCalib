"""
GMM center initialization strategies.

Provides multiple approaches for initializing GMM component centers,
including CAD-based sampling and data-driven methods.
"""

import numpy as np
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import open3d as o3d

class CountStrategy(Enum):
    """Strategies for determining number of GMM centers."""
    FIXED = "fixed"                    # Use predetermined count
    MEDIAN_FRACTION = "median_fraction"  # Fraction of median point count
    MIN_FRACTION = "min_fraction"      # Fraction of minimum point count  
    POINTS_RATIO = "points_ratio"      # Fraction of total points
    SURFACE_AREA = "surface_area"      # Based on CAD surface area
    AUTO = "auto"                      # Heuristic combining factors

class SamplingMethod(Enum):
    """Methods for sampling center positions."""
    UNIFORM = "uniform"                # Uniform random on surface
    POISSON_DISK = "poisson_disk"      # Blue noise sampling
    CURVATURE_WEIGHTED = "curvature"   # More centers in curved regions
    GRID = "grid"                      # Regular grid projected to surface
    FROM_POINTS = "from_points"        # Sample from point cloud data
    STRATIFIED = "stratified"

@dataclass
class InitConfig:
    """Configuration for GMM initialization."""
    count_strategy: CountStrategy = CountStrategy.FIXED
    sampling_method: SamplingMethod = SamplingMethod.UNIFORM
    n_centers: int = 400
    fraction: float = 0.5
    centers_per_m2: float = 1000.0
    seed: Optional[int] = None
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'InitConfig':
        """Create from configuration dictionary."""
        return cls(
            count_strategy=CountStrategy(config.get('count_strategy', 'fixed')),
            sampling_method=SamplingMethod(config.get('sampling_method', 'uniform')),
            n_centers=config.get('n_centers', 400),
            fraction=config.get('fraction', 0.5),
            centers_per_m2=config.get('centers_per_m2', 1000.0),
            seed=config.get('seed', None)
        )


def initialize_centers(
    cad_model: Optional[Any],
    point_clouds: List[np.ndarray],
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Initialize GMM centers based on configuration.
    
    Args:
        cad_model: CAD model instance (optional)
        point_clouds: List of point clouds, each (3, N) or (N, 3)
        config: Configuration dictionary with 'initialization' section
        
    Returns:
        centers: (K, 3) array of GMM center positions
    """
    init_config = config.get('initialization', {})

    cfg = InitConfig.from_dict(init_config)
    
    #if cfg.seed is not None:
    #    np.random.seed(cfg.seed)
    #    o3d.utility.random.seed(cfg.seed)
    
    # Determine number of centers
    n_centers = _compute_n_centers(cfg, point_clouds, cad_model)
    
    # Sample center positions
    if cad_model is not None:
        centers = _sample_from_cad(cad_model, n_centers, cfg.sampling_method)
    else:
        centers = _sample_from_points(point_clouds, n_centers, cfg.sampling_method)
    
    print(f"Initialized {len(centers)} GMM centers using {cfg.sampling_method.value} sampling")
    return centers


def _compute_n_centers(
    cfg: InitConfig,
    point_clouds: List[np.ndarray],
    cad_model: Optional[Any]
) -> int:
    """Compute number of centers based on strategy."""
    
    # Normalize point clouds to (N, 3) format
    clouds = []
    for pc in point_clouds:
        if pc.shape[0] == 3:
            clouds.append(pc.T)
        else:
            clouds.append(pc)
    
    point_counts = [len(c) for c in clouds]
    
    if cfg.count_strategy == CountStrategy.FIXED:
        return cfg.n_centers
    
    elif cfg.count_strategy == CountStrategy.MEDIAN_FRACTION:
        median_count = int(np.median(point_counts))
        return max(50, int(median_count * cfg.fraction))
    
    elif cfg.count_strategy == CountStrategy.MIN_FRACTION:
        min_count = min(point_counts)
        return max(50, int(min_count * cfg.fraction))
    
    elif cfg.count_strategy == CountStrategy.POINTS_RATIO:
        total_points = sum(point_counts)
        return max(50, int(total_points * cfg.fraction / len(clouds)))
    
    elif cfg.count_strategy == CountStrategy.SURFACE_AREA:
        if cad_model is None:
            raise ValueError("SURFACE_AREA strategy requires CAD model")
        area = cad_model.mesh.get_surface_area()
        return max(50, int(area * cfg.centers_per_m2))
    
    elif cfg.count_strategy == CountStrategy.AUTO:
        # Heuristic: balance data coverage with computational cost
        median_count = int(np.median(point_counts))
        base = int(median_count * 0.4)
        
        # Adjust for number of views
        view_factor = min(2.0, len(clouds) / 4)
        
        # Clamp to reasonable range
        return max(100, min(1000, int(base * view_factor)))
    
    else:
        raise ValueError(f"Unknown count strategy: {cfg.count_strategy}")


def _sample_from_cad(
    cad_model: Any,
    n_centers: int,
    method: SamplingMethod
) -> np.ndarray:
    """Sample centers from CAD model surface."""
    
    mesh = cad_model.mesh
    
    if method == SamplingMethod.UNIFORM:
        pcd = mesh.sample_points_uniformly(n_centers)
        return np.asarray(pcd.points)
    
    elif method == SamplingMethod.POISSON_DISK:
        pcd = mesh.sample_points_poisson_disk(n_centers)
        return np.asarray(pcd.points)
    
    elif method == SamplingMethod.CURVATURE_WEIGHTED:
        # Sample more points, weight by curvature, subsample
        pcd = mesh.sample_points_uniformly(n_centers * 10)
        pcd.estimate_normals()
        
        # Compute curvature proxy via normal variation
        pcd_tree = _build_kdtree(pcd)
        curvatures = _estimate_curvatures(pcd, pcd_tree)
        
        # Weighted sampling
        probs = curvatures / curvatures.sum()
        indices = np.random.choice(len(pcd.points), size=n_centers, replace=False, p=probs)
        
        return np.asarray(pcd.points)[indices]
    
    elif method == SamplingMethod.STRATIFIED:
        z_boost = 1.5
        """Explicitly balance normal directions with optional Z boost."""
        pcd = mesh.sample_points_uniformly(n_centers * 10)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        # Classify by dominant axis
        dominant = np.argmax(np.abs(normals), axis=1)
        x_mask = dominant == 0
        y_mask = dominant == 1
        z_mask = dominant == 2
        
        # Weighted allocation
        total = 1.0 + 1.0 + z_boost
        n_x = int(n_centers * 1.0 / total)
        n_y = int(n_centers * 1.0 / total)
        n_z = n_centers - n_x - n_y
        
        # Sample from each group
        x_idx = np.random.choice(np.where(x_mask)[0], size=min(n_x, x_mask.sum()), replace=False)
        y_idx = np.random.choice(np.where(y_mask)[0], size=min(n_y, y_mask.sum()), replace=False)
        z_idx = np.random.choice(np.where(z_mask)[0], size=min(n_z, z_mask.sum()), replace=False)
        
        selected = np.concatenate([x_idx, y_idx, z_idx])
        
        print(f"Stratified: X={len(x_idx)}, Y={len(y_idx)}, Z={len(z_idx)}")
        
        return points[selected]
    elif method == SamplingMethod.GRID:
        # Create 3D grid, project to surface
        bbox = mesh.get_axis_aligned_bounding_box()
        min_b, max_b = bbox.min_bound, bbox.max_bound
        
        # Estimate grid resolution
        volume = np.prod(max_b - min_b)
        spacing = (volume / n_centers) ** (1/3)
        
        x = np.arange(min_b[0], max_b[0], spacing)
        y = np.arange(min_b[1], max_b[1], spacing)
        z = np.arange(min_b[2], max_b[2], spacing)
        
        grid = np.array(np.meshgrid(x, y, z)).reshape(3, -1).T
        
        # Project to surface (using CAD model's method if available)
        if hasattr(cad_model, 'project_points_to_surface'):
            projected, _ = cad_model.project_points_to_surface(grid)
            return projected[:n_centers]
        else:
            # Fallback: use closest points on surface
            scene = cad_model.create_scene()
            closest = scene.compute_closest_points(grid)
            return closest['points'][:n_centers]
    
    else:
        raise ValueError(f"Unknown sampling method: {method}")


def _sample_from_points(
    point_clouds: List[np.ndarray],
    n_centers: int,
    method: SamplingMethod
) -> np.ndarray:
    """Sample centers from point cloud data."""
    
    # Combine all point clouds
    clouds = []
    for pc in point_clouds:
        if pc.shape[0] == 3:
            clouds.append(pc.T)
        else:
            clouds.append(pc)
    
    all_points = np.vstack(clouds)
    
    if method == SamplingMethod.FROM_POINTS or method == SamplingMethod.UNIFORM:
        # Random subset
        indices = np.random.choice(len(all_points), size=min(n_centers, len(all_points)), replace=False)
        return all_points[indices]
    
    elif method == SamplingMethod.POISSON_DISK:
        import open3d as o3d
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        
        # Estimate radius for desired point count
        bbox = pcd.get_axis_aligned_bounding_box()
        volume = np.prod(bbox.max_bound - bbox.min_bound)
        radius = (volume / n_centers) ** (1/3) * 0.5
        
        downsampled = pcd.voxel_down_sample(radius)
        points = np.asarray(downsampled.points)
        
        if len(points) > n_centers:
            indices = np.random.choice(len(points), n_centers, replace=False)
            return points[indices]
        return points
    
    else:
        # Fallback to random sampling
        indices = np.random.choice(len(all_points), size=min(n_centers, len(all_points)), replace=False)
        return all_points[indices]


def _build_kdtree(pcd):
    """Build KD-tree for point cloud."""
    import open3d as o3d
    return o3d.geometry.KDTreeFlann(pcd)


def _estimate_curvatures(pcd, tree, k: int = 30) -> np.ndarray:
    """Estimate curvature at each point via normal variation."""
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    curvatures = np.zeros(len(points))
    
    for i in range(len(points)):
        _, idx, _ = tree.search_knn_vector_3d(points[i], k)
        neighbor_normals = normals[idx[1:]]
        
        # Curvature proxy: variance of neighbor normals
        curvatures[i] = np.var(neighbor_normals, axis=0).sum()
    
    # Normalize and add small epsilon for numerical stability
    curvatures = curvatures / (curvatures.max() + 1e-8) + 0.1
    return curvatures