import os
import pickle 
from functools import reduce
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from ..registry import PIPELINES

def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]

def read_file(path, tries=2, num_point_feature=4, virtual=False, modality="lidar"):
    if virtual:
            raise NotImplementedError()
    
    if modality == "lidar":
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]
    elif modality == "radar":
        pc = RadarPointCloud.from_file(path)
        return pc.points.T

    return points


def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def read_sweep(sweep, virtual=False, modality="lidar"):
    min_distance = 1.0
    if modality == "lidar":
        points_sweep = read_file(str(sweep["lidar_path"]), virtual=virtual, modality="lidar").T
        points_sweep = remove_close(points_sweep, min_distance)
        nbr_points = points_sweep.shape[1]
    
        if sweep["transform_matrix"] is not None:
            points_sweep[:3, :] = sweep["transform_matrix"].dot(
                np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
            )[:3, :]
        curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    if modality == "radar":
        points_sweep = np.empty([18, 0])
        for path, tm in zip(sweep["radar_paths"], sweep["radar_transfrom_matrices"]):
            if path is None or tm is None:
                continue
            _points_sweep = read_file(path, virtual=virtual, modality="radar").T
            _points_sweep = remove_close(_points_sweep, min_distance)
            _nbr_points = _points_sweep.shape[1]
            _points_sweep[:3, :] = tm.dot(np.vstack((_points_sweep[:3, :], np.ones(_nbr_points))))[:3, :]
            points_sweep = np.hstack([points_sweep, _points_sweep])

        curr_times = sweep["radar_time_lags"][0] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T



def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 


@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    def __init__(self, dataset="NuScenesDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)
        self.modality = kwargs.get("modality", "lidar")

        assert self.modality in ["lidar", "radar"], "Invalid modality"

    def __call__(self, res, info):

        res["type"] = self.type

        if self.type != "NuScenesDataset":
            raise NotImplementedError

        nsweeps = res[self.modality]["nsweeps"]

        if self.modality == "radar":
            points = np.empty([0, 18])
            for radar_path in info["radar_paths"]:
                radar_points = read_file(radar_path, virtual=False, modality="radar")
                points = np.concatenate([points, radar_points])
        else:
            sensor_path = Path(info["lidar_path"])
            points = read_file(str(sensor_path), virtual=res["virtual"], modality="lidar")

        
        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        assert (nsweeps - 1) == len(
            info["sweeps"]
        ), "nsweeps {} should equal to list length {}.".format(
            nsweeps, len(info["sweeps"])
        )

        for i in np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
            sweep = info["sweeps"][i]
            if self.modality == "radar" and not sweep.get("radar_paths"):
                continue
            points_sweep, times_sweep = read_sweep(sweep, virtual=res["virtual"], modality=self.modality)
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        res[self.modality]["points"] = points
        res[self.modality]["times"] = times
        res[self.modality]["combined"] = np.hstack([points, times])
    
        return res, info


@PIPELINES.register_module
class LoadPointCloudAnnotations(object):
    def __init__(self, with_bbox=True, **kwargs):
        pass

    def __call__(self, res, info):

        if res["type"] in ["NuScenesDataset"] and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes[np.isnan(gt_boxes)] = 0
            res["lidar"]["annotations"] = {
                "boxes": gt_boxes,
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
            }
        else:
            pass 

        return res, info
