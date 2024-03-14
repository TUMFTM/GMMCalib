import numpy as np
import open3d as o3d
import pickle
from transformPCDs import compute_global_transform
import yaml


def generate_data(data_path, config_file_path, sequence):
    # Read the parameters from the YAML file
    with open(config_file_path, 'r') as file:
        config_data = yaml.safe_load(file)

    transform_sensor_1 = config_data.get("transform_sensor_1", "")[0]
    transform_sensor_2 = config_data.get("transform_sensor_2", "")[0]
    min_bound = config_data.get("min_bound", "")[0]
    max_bound = config_data.get("max_bound", "")[0]
    number_of_sensors = config_data.get("number_of_sensors", "")
    
    sensors = [data_path + "/sensor_" + str(i+1) + "/" for i in range(number_of_sensors)]

    pcds = []
    for sensor in sensors:
        for frame in range(sequence[0], sequence[-1]+1):
            pcd = o3d.io.read_point_cloud(sensor + str(frame) + ".pcd")

            if sensor == sensors[0]:
                T_g = compute_global_transform(transform_sensor_1[3:], transform_sensor_1[:3])
            else: 
                T_g = compute_global_transform(transform_sensor_2[3:], transform_sensor_2[:3])

            pcd.transform(T_g)
            # Crop 
            roi = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            pcds.append(pcd.crop(roi))
    return pcds