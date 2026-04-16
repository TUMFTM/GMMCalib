import argparse
import os
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as R

import create_gt
import transformPCDs
from cad_model import CADModel
from data_loader import PCDLoader
from gmm_initialization import initialize_centers

GMM_VARIANTS = {
    "default": ("gmm", "GMM"),
    "gmm": ("gmm", "GMM"),
    "gmm_calib": ("gmm", "GMM"),
    "cad_calib": ("cad_calib", "CADCalib"),
}


def get_gmm_class(name: str):
    """Lazy import GMM variant."""
    if name not in GMM_VARIANTS:
        raise ValueError(
            f"Unknown GMM type: {name}. Available: {list(GMM_VARIANTS.keys())}"
        )
    module_name, class_name = GMM_VARIANTS[name]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def calibrate(
    pcds,
    config,
    max_num_iter: int = 100,
    cad_model=None,
    method: str = "gmm_calib",
    debug_config=None,
    pcd_loader=None,
):
    V = [np.asarray(cloud.points) for cloud in pcds]
    nObs = len(V)

    if method == "cad_calib" and cad_model is None:
        raise ValueError("CADCalib requires cad_model but none was provided")

    if cad_model is not None:
        centers = initialize_centers(cad_model, V, config)
    else:
        centers = create_gt.create_init_pc(
            box_size=(0.5, 0.5, 0.5), num_points=400
        ) + np.array([9.8, 4.75, 0.38], dtype=np.float64)

    GMM_Class = get_gmm_class(method)

    gmm = GMM_Class(
        config=config,
        V=V,
        Xin=centers,
        cad_model=cad_model,
        debug_config=debug_config,
        pcd_loader=pcd_loader,
    )
    X, TV, AllT, pk = gmm.run_optimization(max_num_iter)

    if not AllT:
        raise RuntimeError("No transforms were produced during optimization")

    T_1 = [
        transformPCDs.homogeneous_transform(AllT[-1][0][i], AllT[-1][1][i].reshape(-1))
        for i in range(nObs // 2)
    ]
    T_2 = [
        transformPCDs.homogeneous_transform(AllT[-1][0][i], AllT[-1][1][i].reshape(-1))
        for i in range(nObs // 2, nObs)
    ]

    T_calib = [np.linalg.inv(T_2[i]) @ T_1[i] for i in range(len(T_1))]
    T_final = transformPCDs.mean_transform(T_calib)

    R_final = T_final[:3, :3]
    t_final = T_final[:3, 3]
    rpy = R.from_matrix(R_final).as_euler("xyz")

    print(f"Calibration result:\n{T_final}")
    print(
        f"Roll: {rpy[0]:.6f} | Pitch: {rpy[1]:.6f} | Yaw: {rpy[2]:.6f} rad | "
        f"X: {t_final[0]:.6f} | Y: {t_final[1]:.6f} | Z: {t_final[2]:.6f} m"
    )

    os.makedirs("./output", exist_ok=True)
    with open("./output/gmmcalib_result.pkl", "wb") as f:
        pickle.dump([T_final, X], f)

    with open("metrics.txt", "w") as f:
        f.write(f"roll_rad {abs(rpy[0]):.6f}\n")
        f.write(f"pitch_rad {abs(rpy[1]):.6f}\n")
        f.write(f"yaw_rad {abs(rpy[2]):.6f}\n")
        f.write(f"x_m {abs(t_final[0]):.6f}\n")
        f.write(f"y_m {abs(t_final[1]):.6f}\n")
        f.write(f"z_m {abs(t_final[2]):.6f}\n")

    return T_final, X, TV


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run calibration")
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--config_file_path", type=str, default="../config/config.yaml")
    parser.add_argument("--sequence", nargs="+", type=int)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--method", type=str, default="gmm_calib")
    parser.add_argument("--robust", type=bool, default=False)
    parser.add_argument("--max_num_iter", type=int, default=100)

    args = parser.parse_args()

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.data_path))
    config_file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), args.config_file_path)
    )
    method = args.method

    model_path = (
        os.path.abspath(os.path.join(os.path.dirname(__file__), args.model_path))
        if args.model_path is not None
        else None
    )

    cad_model = None
    if model_path is not None:
        cad_model = CADModel(model_path)

    if args.sequence is None:
        sensor1_dir = os.path.join(data_path, "sensor_1")
        sequence = list(range(1, len(os.listdir(sensor1_dir)) + 1))
    else:
        sequence = args.sequence

    pcd_loader = PCDLoader(data_path, config_file_path, sequence)

    config = {}

    # algo hyper params
    debug_config = {}
    debug_config["verbose"] = False

    # set some reasonable defaults
    if method == "cad_calib":
        config["initialization"] = {
            "count_strategy": "median_fraction",
            "fraction": 1.75,
            "sampling_method": "poisson_disk",
        }
        if args.robust:
            debug_config["alimit"] = 10  # reduce anisotropy with high noise
        else:
            debug_config["alimit"] = 30  # highly anisotropic

    calibrate(
        pcd_loader.pcds_overlap,
        config=config,
        max_num_iter=args.max_num_iter,
        debug_config=debug_config,
        method=method,
        cad_model=cad_model,
        pcd_loader=pcd_loader,
    )
