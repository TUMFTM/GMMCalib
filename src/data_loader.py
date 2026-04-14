from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import yaml

from transformPCDs import compute_global_transform


class PCDLoader:
    def __init__(
        self,
        data_path: str,
        config_file_path: str,
        sequence: list[int],
        min_points_in_roi: int = 100,
        require_paired: bool = True,
    ) -> None:
        self.data_path = Path(data_path)
        self.config_path = Path(config_file_path)
        self.sequence = sequence
        self.min_points_in_roi = min_points_in_roi
        self.require_paired = require_paired

        self._config = self._load_config()
        self.global_transforms = self._compute_transforms()

        self.roi = self._get_roi()
        self.roi_per_sensor = [self._get_roi(0), self._get_roi(1)]

        self.pcds_full: list[o3d.geometry.PointCloud] = []
        self.pcds_overlap: list[o3d.geometry.PointCloud] = []
        self.view_labels: list[str] = []
        self.valid_frames: list[int] = []

        self.quality_stats: dict[str, Any] = {
            "frames_requested": len(sequence),
            "frames_loaded": 0,
            "frames_skipped": 0,
            "skip_reasons": [],
        }

        self._load_default_format()

        self.num_sensors = 2
        self.num_views_per_sensor = len(self.pcds_full) // 2

        self._print_summary()

    def _load_config(self) -> dict[str, Any]:
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _compute_transforms(self) -> tuple[np.ndarray, np.ndarray]:
        t1 = self._config.get("transform_sensor_1", [[0, 0, 0, 0, 0, 0]])[0]
        t2 = self._config.get("transform_sensor_2", [[0, 0, 0, 0, 0, 0]])[0]

        T_1 = compute_global_transform(t1[3:], t1[:3])
        T_2 = compute_global_transform(t2[3:], t2[:3])
        return T_1, T_2

    def _get_roi(
        self, sensor_idx: int | None = None
    ) -> o3d.geometry.AxisAlignedBoundingBox:
        if sensor_idx is not None:
            min_key = f"min_bound_sensor_{sensor_idx + 1}"
            max_key = f"max_bound_sensor_{sensor_idx + 1}"
            if min_key in self._config and max_key in self._config:
                min_bound = self._config[min_key][0]
                max_bound = self._config[max_key][0]
                return o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        min_bound = self._config.get("min_bound", [[-10, -10, -10]])[0]
        max_bound = self._config.get("max_bound", [[10, 10, 10]])[0]
        return o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    def _sensor_file(self, sensor_idx: int, frame: int) -> Path:
        return self.data_path / f"sensor_{sensor_idx + 1}" / f"{frame}.pcd"

    def _try_load(
        self,
        file_path: Path,
        transform: np.ndarray,
    ) -> o3d.geometry.PointCloud | None:
        if not file_path.exists():
            return None

        try:
            pcd = o3d.io.read_point_cloud(str(file_path))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

        if len(pcd.points) == 0:
            return None

        pcd.transform(transform)
        return pcd

    def _record_skip(self, frame: int, reason: str) -> None:
        self.quality_stats["frames_skipped"] += 1
        self.quality_stats["skip_reasons"].append({"frame": frame, "reason": reason})

    def _add_point_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        label: str,
        sensor_idx: int,
    ) -> None:
        cropped = pcd.crop(self.roi_per_sensor[sensor_idx])
        self.pcds_full.append(pcd)
        self.pcds_overlap.append(cropped)
        self.view_labels.append(label)
        self.quality_stats["frames_loaded"] += 1

    def _load_default_format(self) -> None:
        T_1, T_2 = self.global_transforms
        transforms = [T_1, T_2]

        valid_frames: list[int] = []

        # First pass: determine valid paired frames
        for frame in self.sequence:
            frame_ok = True
            roi_counts: list[int] = []

            for sensor_idx in range(2):
                path = self._sensor_file(sensor_idx, frame)
                pcd = self._try_load(path, transforms[sensor_idx])

                if pcd is None:
                    frame_ok = False
                    self._record_skip(
                        frame, f"sensor_{sensor_idx + 1} missing/empty: {path}"
                    )
                    break

                n_roi = len(pcd.crop(self.roi_per_sensor[sensor_idx]).points)
                roi_counts.append(n_roi)

                if n_roi < self.min_points_in_roi:
                    frame_ok = False
                    self._record_skip(
                        frame,
                        f"sensor_{sensor_idx + 1} only {n_roi} points in ROI",
                    )
                    if self.require_paired:
                        break

            if frame_ok or (
                not self.require_paired
                and any(c >= self.min_points_in_roi for c in roi_counts)
            ):
                valid_frames.append(frame)

        self.valid_frames = valid_frames

        # Second pass: load valid frames, sensor 1 first then sensor 2
        for sensor_idx in range(2):
            for frame in valid_frames:
                path = self._sensor_file(sensor_idx, frame)
                pcd = self._try_load(path, transforms[sensor_idx])
                if pcd is not None:
                    self._add_point_cloud(
                        pcd,
                        label=f"sensor_{sensor_idx + 1}_{frame}",
                        sensor_idx=sensor_idx,
                    )

    def _print_summary(self) -> None:
        n_requested = self.quality_stats["frames_requested"]
        n_valid = len(self.valid_frames)
        n_skipped = n_requested - n_valid

        print("\n" + "=" * 50)
        print("Data Loading Summary")
        print("=" * 50)
        print(f"Frames requested: {n_requested}")
        print(
            f"Frames loaded:    {n_valid} ({100 * n_valid / max(1, n_requested):.1f}%)"
        )
        print(f"Frames skipped:   {n_skipped}")
        print(f"Total point clouds: {len(self.pcds_full)}")

        if n_skipped > 0:
            print("\nSkipped frames:")
            for info in self.quality_stats["skip_reasons"][:10]:
                print(f"  Frame {info['frame']}: {info['reason']}")

        if self.pcds_overlap:
            counts = [len(pcd.points) for pcd in self.pcds_overlap]
            print("\nPoints per view (in ROI):")
            print(
                f"  Min: {min(counts)}, Max: {max(counts)}, "
                f"Mean: {np.mean(counts):.0f}, Median: {np.median(counts):.0f}"
            )

        print("=" * 50 + "\n")

    def get_sensor_scans_full(self, sensor_idx: int) -> list[o3d.geometry.PointCloud]:
        start = sensor_idx * self.num_views_per_sensor
        end = start + self.num_views_per_sensor
        return self.pcds_full[start:end]

    def get_sensor_scans_overlap(
        self, sensor_idx: int
    ) -> list[o3d.geometry.PointCloud]:
        start = sensor_idx * self.num_views_per_sensor
        end = start + self.num_views_per_sensor
        return self.pcds_overlap[start:end]

    def get_all_points(self, use_overlap: bool = True) -> np.ndarray:
        pcds = self.pcds_overlap if use_overlap else self.pcds_full
        if not pcds:
            return np.empty((0, 3), dtype=np.float64)
        return np.vstack([np.asarray(p.points) for p in pcds])

    def get_quality_report(self) -> dict[str, Any]:
        counts = [len(pcd.points) for pcd in self.pcds_overlap]
        return {
            "frames_requested": self.quality_stats["frames_requested"],
            "frames_valid": len(self.valid_frames),
            "valid_frame_indices": self.valid_frames,
            "skip_reasons": self.quality_stats["skip_reasons"],
            "point_statistics": {
                "min": min(counts) if counts else 0,
                "max": max(counts) if counts else 0,
                "mean": float(np.mean(counts)) if counts else 0.0,
                "median": float(np.median(counts)) if counts else 0.0,
                "total": int(sum(counts)),
            },
        }
