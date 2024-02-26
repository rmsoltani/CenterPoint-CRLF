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
        rm = Quaternion(axis=(0,0,1), degrees=90).rotation_matrix
        pc = RadarPointCloud.from_file(path)
        pc.rotate(rm)
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


def read_sweep(sweep, virtual=False, modality="lidar", extend=False):
    min_distance = 1.0
    points_sweep = read_file(str(sweep[f"{modality}_path"]), virtual=virtual, modality=modality).T
    points_sweep = remove_close(points_sweep, min_distance)
    nbr_points = points_sweep.shape[1]
    
    postfix = "_radar" if modality == "radar" else ""
    
    if sweep[f"transform_matrix{postfix}"] is not None:
        points_sweep[:3, :] = sweep[f"transform_matrix{postfix}"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]

    if modality == "radar" and extend:
        points_sweep = np.concatenate([
            points_sweep.T,
            _get_extra_radar(
                sweep["extra_paths_radar"],
                sweep["extra_cs_recs_radar"],
                sweep["global_from_car_radar"]
            )
        ]).T
    curr_times = sweep[f"time_lag{postfix}"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T


def _get_extra_radar(paths, cs_recs, pose_recs, ref_from_car=None, car_from_global=None):
    all_points = np.empty([0, 18])
    for path, cs_rec, pose_rec in zip(paths, cs_recs, pose_recs):
        min_distance = 1.0
        car_from_current = transform_matrix(cs_rec["translation"], Quaternion(cs_rec["rotation"])) if type(cs_rec) is dict else cs_rec
        global_from_car = transform_matrix(pose_rec["translation"], Quaternion(pose_rec["rotation"])) if type(pose_rec) is dict else pose_rec

        if ref_from_car is None or car_from_global is None:
            tm = car_from_current
        else:
            tm = reduce(
                np.matmul,
                [ref_from_car, car_from_global, global_from_car, car_from_current],
            )
        points = read_file(path, modality="radar").T
        points = remove_close(points, min_distance)
        points[:3, :] = tm.dot(
            np.vstack((points[:3, :], np.ones(points.shape[1])))
        )[:3, :]

        all_points = np.concatenate([all_points, points.T])

    return all_points

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
        self.extend = kwargs.get("extend", False)

        assert self.modality in ["lidar", "radar"], "Invalid modality"

    def __call__(self, res, info):

        res["type"] = self.type

        if self.type != "NuScenesDataset":
            raise NotImplementedError

        nsweeps = res[self.modality]["nsweeps"]

        sensor_path = Path(info[f"{self.modality}_path"])
        points = read_file(str(sensor_path), virtual=res["virtual"], modality=self.modality)

        if self.modality == "radar" and self.extend:
            points = np.concatenate([
                points, 
                _get_extra_radar(
                    info["extra_paths_radar"],
                    info["extra_cs_recs_radar"],
                    info["extra_pose_recs_radar"],
                    ref_from_car=info["ref_from_car_radar"], 
                    car_from_global=info["car_from_global_radar"]
                )
            ])
        
        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        assert (nsweeps - 1) == len(
            info["sweeps"]
        ), "nsweeps {} should equal to list length {}.".format(
            nsweeps, len(info["sweeps"])
        )

        for i in np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
            sweep = info["sweeps"][i]
            points_sweep, times_sweep = read_sweep(sweep, virtual=res["virtual"], modality=self.modality, extend=self.extend)
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
