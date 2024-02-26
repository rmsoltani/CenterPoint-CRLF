import numpy as np
import pickle

from pathlib import Path
from functools import reduce
from typing import List

from tqdm import tqdm
from pyquaternion import Quaternion

try:
    from nuscenes import NuScenes
    from nuscenes.utils import splits
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import transform_matrix
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
except:
    print("nuScenes devkit not Found!")

general_to_detection = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}

cls_attr_dist = {
    "barrier": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "bicycle": {
        "cycle.with_rider": 2791,
        "cycle.without_rider": 8946,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "bus": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 9092,
        "vehicle.parked": 3294,
        "vehicle.stopped": 3881,
    },
    "car": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 114304,
        "vehicle.parked": 330133,
        "vehicle.stopped": 46898,
    },
    "construction_vehicle": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 882,
        "vehicle.parked": 11549,
        "vehicle.stopped": 2102,
    },
    "ignore": {
        "cycle.with_rider": 307,
        "cycle.without_rider": 73,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 165,
        "vehicle.parked": 400,
        "vehicle.stopped": 102,
    },
    "motorcycle": {
        "cycle.with_rider": 4233,
        "cycle.without_rider": 8326,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "pedestrian": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 157444,
        "pedestrian.sitting_lying_down": 13939,
        "pedestrian.standing": 46530,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "traffic_cone": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "trailer": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 3421,
        "vehicle.parked": 19224,
        "vehicle.stopped": 1895,
    },
    "truck": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 21339,
        "vehicle.parked": 55626,
        "vehicle.stopped": 11097,
    },
}

def _second_det_to_nusc_box(detection):
    box3d = detection["box3d_lidar"].detach().cpu().numpy()
    scores = detection["scores"].detach().cpu().numpy()
    labels = detection["label_preds"].detach().cpu().numpy()
    box3d[:, -1] = -box3d[:, -1] - np.pi / 2
    box_list = []
    for i in range(box3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=box3d[i, -1])
        velocity = (*box3d[i, 6:8], 0.0)
        box = Box(
            box3d[i, :3],
            box3d[i, 3:6],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        box_list.append(box)
    return box_list


def _lidar_nusc_box_to_global(nusc, boxes, sample_token):
    try:
        s_record = nusc.get("sample", sample_token)
        sample_data_token = s_record["data"]["LIDAR_TOP"]
    except:
        sample_data_token = sample_token

    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(cs_record["rotation"]))
        box.translate(np.array(cs_record["translation"]))
        # Move box to global coord system
        box.rotate(Quaternion(pose_record["rotation"]))
        box.translate(np.array(pose_record["translation"]))
        box_list.append(box)
    return box_list


def _get_available_scenes(nusc):
    available_scenes = []
    print("total scene num:", len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num:", len(available_scenes))
    return available_scenes


def get_sample_data(
    nusc, sample_data_token: str, selected_anntokens: List[str] = None
):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param selected_anntokens: If provided only return the selected annotation.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = nusc.get("sensor", cs_record["sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record["modality"] == "camera":
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])
    else:
        cam_intrinsic = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        box.velocity = nusc.box_velocity(box.token)
        # Move box to ego vehicle coord system
        box.translate(-np.array(pose_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record["translation"]))
        box.rotate(Quaternion(cs_record["rotation"]).inverse)

        box_list.append(box)

    return data_path, box_list, cam_intrinsic

CAM_CHANS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
RADAR_CHANS = ['RADAR_FRONT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_RIGHT', 'RADAR_FRONT_LEFT', 'RADAR_BACK_LEFT']


def get_pointsensor_to_image_transform(nusc, pointsensor,  camera_sensor):
    tms = []
    intrinsics = []  
    cam_paths = [] 
    for chan in CAM_CHANS:
        cam = camera_sensor[chan]

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
        ps_cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        car_from_lidar = transform_matrix(
            ps_cs_record["translation"], Quaternion(ps_cs_record["rotation"]), inverse=False
        )

        # Second step: transform to the global frame.
        ps_poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
        global_from_car = transform_matrix(
            ps_poserecord["translation"],  Quaternion(ps_poserecord["rotation"]), inverse=False,
        )

        # Third step: transform into the ego vehicle frame for the timestamp of the image.
        cam_poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
        car_from_global = transform_matrix(
            cam_poserecord["translation"],
            Quaternion(cam_poserecord["rotation"]),
            inverse=True,
        )

        # Fourth step: transform into the camera.
        cam_cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        cam_from_car = transform_matrix(
            cam_cs_record["translation"], Quaternion(cam_cs_record["rotation"]), inverse=True
        )

        tm = reduce(
            np.dot,
            [cam_from_car, car_from_global, global_from_car, car_from_lidar],
        )

        cam_path, _, intrinsic = nusc.get_sample_data(cam['token'])

        tms.append(tm)
        intrinsics.append(intrinsic)
        cam_paths.append(cam_path)

    return tms, intrinsics, cam_paths  

def find_closet_camera_tokens(nusc, pointsensor, ref_sample):
    lidar_timestamp = pointsensor["timestamp"]

    min_cams = {} 

    for chan in CAM_CHANS:
        camera_token = ref_sample['data'][chan]

        cam = nusc.get('sample_data', camera_token)
        min_diff = abs(lidar_timestamp - cam['timestamp'])
        min_cam = cam

        for i in range(6):  # nusc allows at most 6 previous camera frames 
            if cam['prev'] == "":
                break 

            cam = nusc.get('sample_data', cam['prev'])
            cam_timestamp = cam['timestamp']

            diff = abs(lidar_timestamp-cam_timestamp)

            if (diff < min_diff):
                min_diff = diff 
                min_cam = cam 
            
        min_cams[chan] = min_cam 

    return min_cams     


def _fill_trainval_infos(nusc, train_scenes, val_scenes, test=False, nsweeps=10, filter_zero=True, modalities=["lidar"]):
    from nuscenes.utils.geometry_utils import transform_matrix

    train_nusc_infos = []
    val_nusc_infos = []

    REF_CHAN = "LIDAR_TOP"  # The lidar channel from which we track back n sweeps to aggregate the point cloud.
    CHAN = "LIDAR_TOP"  # The reference channel of the current sample_rec that the point clouds are mapped to.
    RAD_REF_CHAN = "RADAR_FRONT"  # The radar channel from which we track back n sweeps to aggregate the point cloud.
    RAD_CHAN = "RADAR_FRONT"  # The reference channel of the current sample_rec that the point clouds are mapped to.

    for sample in tqdm(nusc.sample):
        """ Manual save info["sweeps"] """        

        info = {}
        if "lidar" in modalities:
            lidar_info, ref_boxes = _get_sensor_info(nusc, sample, REF_CHAN)
            info.update(lidar_info)
        
        if "radar" in modalities:
            radar_info, rad_ref_boxes = _get_sensor_info(nusc, sample, RAD_REF_CHAN)
            info.update(radar_info)
        
        if info == {}:
            raise ValueError("Invalid modailties!")

        sample_data_token = sample["data"][CHAN]
        rad_sample_data_token = sample["data"][RAD_CHAN]
        curr_sd_rec = nusc.get("sample_data", sample_data_token)
        rad_curr_sd_rec = nusc.get("sample_data", rad_sample_data_token)
        sweeps = []
        while len(sweeps) < nsweeps - 1:
            if "lidar" in modalities:
                if curr_sd_rec["prev"] == "":
                    sweep = {
                        "lidar_path": info["lidar_path"],
                        "sample_data_token": curr_sd_rec["token"],
                        "transform_matrix": None,
                        "time_lag": curr_sd_rec["timestamp"] * 0,
                        "all_cams_from_lidar": info["all_cams_from_lidar"],
                        "all_cams_intrinsic": info["all_cams_intrinsic"],
                        "all_cams_path": info["all_cams_path"],
                    }

                    if len(sweeps) != 0:
                        sweep = {k: v for k,v in sweeps[-1].items() if k in sweep}
                else:
                    curr_sd_rec = nusc.get("sample_data", curr_sd_rec["prev"])

                    sweep = _get_sweep_info(
                        nusc, 
                        curr_sd_rec, 
                        info["timestamp"], 
                        sample=sample, 
                        ref_from_car=info["ref_from_car"], 
                        car_from_global=info["car_from_global"]
                    )
            else:
                sweep = {}
            
            if "radar" in modalities:
                if rad_curr_sd_rec["prev"] == "":
                    sample = nusc.get("sample", rad_curr_sd_rec["sample_token"])
                    rad_extra_sample_data_tokens = [sample["data"][chan] for chan in RADAR_CHANS if chan != rad_curr_sd_rec["channel"]]
                    rad_extra_paths = [nusc.get_sample_data_path(token) for token in rad_extra_sample_data_tokens]
                    rad_extra_sd_recs = [nusc.get("sample_data", token) for token in rad_extra_sample_data_tokens]
                    rad_extra_cs_recs = [nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"]) for sd_rec in rad_extra_sd_recs]
                    rad_sweep = {
                        "radar_path": info["radar_path"],
                        "extra_paths_radar": rad_extra_paths,
                        "sample_data_token_radar": rad_curr_sd_rec["token"],
                        "extra_sample_data_tokens_radar": rad_extra_sample_data_tokens,
                        "transform_matrix_radar": None,
                        "extra_transform_matrices_radar": [None] * len(rad_extra_sample_data_tokens),
                        "time_lag_radar": 0,
                        "cams_from_radars_radar": info["all_cams_from_radar"],
                        "all_cams_intrinsics_radar": info["all_cams_intrinsic_radar"],
                        "all_cams_paths_radar": info["all_cams_path_radar"],
                        "global_from_car_radar": nusc.get("ego_pose", rad_curr_sd_rec["ego_pose_token"]),
                        "extra_cs_recs_radar": rad_extra_cs_recs,
                    }
                    
                    if len(sweeps) != 0:
                        rad_sweep = {k: v for k,v in sweeps[-1].items() if k in rad_sweep}
                else:
                    prev = nusc.get("sample_data", rad_curr_sd_rec["prev"])
                    if prev == "" or prev["sample_token"] != rad_curr_sd_rec["sample_token"]:
                        prev = rad_curr_sd_rec
                    rad_curr_sd_rec = prev

                    rad_sweep = _get_sweep_info(
                        nusc,
                        prev,
                        info["timestamp_radar"],
                        sample=sample, 
                        ref_from_car=info["ref_from_car_radar"], 
                        car_from_global=info["car_from_global_radar"]
                    )

            else:
                rad_sweep = {}
            
            sweeps.append({**sweep, **rad_sweep})
        info["sweeps"] = sweeps

        assert (
            len(info["sweeps"]) == nsweeps - 1
        )
        
        if not test:
            annotations = [
                nusc.get("sample_annotation", token) for token in sample["anns"]
            ]

            mask = np.array([(anno['num_lidar_pts'] + anno['num_radar_pts'])>0 for anno in annotations], dtype=bool).reshape(-1)

            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)
            # rots = np.array([b.orientation.yaw_pitch_roll[0] for b in ref_boxes]).reshape(-1, 1)
            velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
            rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(
                -1, 1
            )
            names = np.array([b.name for b in ref_boxes])
            tokens = np.array([b.token for b in ref_boxes])
            gt_boxes = np.concatenate(
                [locs, dims, velocity[:, :2], -rots - np.pi / 2], axis=1
            )
            # gt_boxes = np.concatenate([locs, dims, rots], axis=1)

            assert len(annotations) == len(gt_boxes) == len(velocity)

            if not filter_zero:
                info["gt_boxes"] = gt_boxes
                info["gt_boxes_velocity"] = velocity
                info["gt_names"] = np.array([general_to_detection[name] for name in names])
                info["gt_boxes_token"] = tokens
            else:
                info["gt_boxes"] = gt_boxes[mask, :]
                info["gt_boxes_velocity"] = velocity[mask, :]
                info["gt_names"] = np.array([general_to_detection[name] for name in names])[mask]
                info["gt_boxes_token"] = tokens[mask]

        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def _get_sensor_info(nusc, sample, ref_chan):
     # Get reference pose and timestamp
        ref_sd_token = sample["data"][ref_chan]
        ref_sd_rec = nusc.get("sample_data", ref_sd_token)
        ref_cs_rec = nusc.get(
            "calibrated_sensor", ref_sd_rec["calibrated_sensor_token"]
        )
        ref_pose_rec = nusc.get("ego_pose", ref_sd_rec["ego_pose_token"])
        ref_time = 1e-6 * ref_sd_rec["timestamp"]

        ref_sensor_path, ref_boxes, _ = get_sample_data(nusc, ref_sd_token)

        ref_cam_front_token = sample["data"]["CAM_FRONT"]
        ref_cam_path, _, ref_cam_intrinsic = nusc.get_sample_data(ref_cam_front_token)

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(
            ref_cs_rec["translation"], Quaternion(ref_cs_rec["rotation"]), inverse=True
        )

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(
            ref_pose_rec["translation"],
            Quaternion(ref_pose_rec["rotation"]),
            inverse=True,
        )

        ref_cams = {}
        # get all camera sensor data
        for cam_chan in CAM_CHANS:
            camera_token = sample['data'][cam_chan]
            cam = nusc.get('sample_data', camera_token)

            ref_cams[cam_chan] = cam 

        # get camera info for point painting 
        all_cams_from_sensor, all_cams_intrinsic, all_cams_path = get_pointsensor_to_image_transform(nusc, pointsensor=ref_sd_rec, camera_sensor=ref_cams)    

        modality = "radar" if "RADAR" in ref_chan else "lidar"
        postfix = "_radar" if modality == "radar" else ""
        info = {
            f"{modality}_path": ref_sensor_path,
            "cam_front_path": ref_cam_path,
            "cam_intrinsic": ref_cam_intrinsic,
            "token": sample["token"],
            "sweeps": [],
            f"ref_from_car{postfix}": ref_from_car,
            f"car_from_global{postfix}": car_from_global,
            f"timestamp{postfix}": ref_time,
            f"all_cams_from_{modality}": all_cams_from_sensor,
            f"all_cams_intrinsic{postfix}": all_cams_intrinsic,
            f"all_cams_path{postfix}": all_cams_path
        }
        return info, ref_boxes


def _get_sweep_info(nusc, curr_sd_rec, ref_time, sample=None, ref_from_car=None, car_from_global=None):

    # get nearest camera frame data 
    cam_data = find_closet_camera_tokens(nusc, curr_sd_rec, ref_sample=sample)
    cur_cams_from_sensor, cur_cams_intrinsic, cur_cams_path = get_pointsensor_to_image_transform(nusc, pointsensor=curr_sd_rec, camera_sensor=cam_data)   

    # Get past pose
    current_pose_rec = nusc.get("ego_pose", curr_sd_rec["ego_pose_token"])
    global_from_car = transform_matrix(
        current_pose_rec["translation"],
        Quaternion(current_pose_rec["rotation"]),
        inverse=False,
    )

    # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
    current_cs_rec = nusc.get(
        "calibrated_sensor", curr_sd_rec["calibrated_sensor_token"]
    )
    car_from_current = transform_matrix(
        current_cs_rec["translation"],
        Quaternion(current_cs_rec["rotation"]),
        inverse=False,
    )

    tm = reduce(
        np.matmul,
        [ref_from_car, car_from_global, global_from_car, car_from_current],
    )

    modality = curr_sd_rec["sensor_modality"]

    data_path = nusc.get_sample_data_path(curr_sd_rec["token"])

    time_lag = ref_time - 1e-6 * curr_sd_rec["timestamp"]
    postfix = "_radar" if modality == "radar" else ""
    sweep_info =  {
        f"{modality}_path": data_path,
        f"sample_data_token{postfix}": curr_sd_rec["token"],
        f"transform_matrix{postfix}": tm,
        f"global_from_car{postfix}": global_from_car,
        f"car_from_current{postfix}": car_from_current,
        f"time_lag{postfix}": time_lag,
        f"all_cams_intrinsic{postfix}": cur_cams_intrinsic,
        f"all_cams_path{postfix}": cur_cams_path,
        f"all_cams_from_{modality}": cur_cams_from_sensor,
    }

    if modality == "radar":
        sample = nusc.get("sample", curr_sd_rec["sample_token"])
        rad_extra_sample_data_tokens = [sample["data"][chan] for chan in RADAR_CHANS if chan != curr_sd_rec["channel"]]
        rad_extra_paths = [nusc.get_sample_data_path(token) for token in rad_extra_sample_data_tokens]
        rad_extra_sd_recs = [nusc.get("sample_data", token) for token in rad_extra_sample_data_tokens]
        rad_extra_cs_recs = [nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"]) for sd_rec in rad_extra_sd_recs]

        sweep_info["extra_sample_data_tokens_radar"] = rad_extra_sample_data_tokens
        sweep_info["extra_paths_radar"] = rad_extra_paths
        sweep_info["extra_cs_recs_radar"] = rad_extra_cs_recs

    return sweep_info

def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw
    

def create_nuscenes_infos(root_path, version="v1.0-trainval", nsweeps=10, filter_zero=True, modalities=["lidar"], base_suffix=""):
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers
    if version == "v1.0-trainval":
        train_scenes = splits.train
        # random.shuffle(train_scenes)
        # train_scenes = train_scenes[:int(len(train_scenes)*0.2)]
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")
    test = "test" in version
    root_path = Path(root_path)
    # filter exist scenes. you may only download part of dataset.
    available_scenes = _get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set(
        [
            available_scenes[available_scene_names.index(s)]["token"]
            for s in train_scenes
        ]
    )
    val_scenes = set(
        [available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes]
    )
    if test:
        print(f"test scene: {len(train_scenes)}")
    else:
        print(f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, nsweeps=nsweeps, filter_zero=filter_zero, modalities=modalities
    )

    suffix = base_suffix
    suffix += "velo" if "lidar" in modalities else ""
    suffix += "_radar" if "radar" in modalities else ""

    if test:
        print(f"test sample: {len(train_nusc_infos)}")
        with open(
            root_path / f"infos_test_{nsweeps}sweeps_with{suffix}.pkl", "wb"
        ) as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print(
            f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}"
        )
        with open(
            root_path / f"infos_train_{nsweeps}sweeps_with{suffix}_filter_{filter_zero}.pkl", "wb"
        ) as f:
            pickle.dump(train_nusc_infos, f)
        with open(
            root_path / f"infos_val_{nsweeps}sweeps_with{suffix}_filter_{filter_zero}.pkl", "wb"
        ) as f:
            pickle.dump(val_nusc_infos, f)


def eval_main(nusc, eval_version, res_path, eval_set, output_dir):
    # nusc = NuScenes(version=version, dataroot=str(root_path), verbose=True)
    cfg = config_factory(eval_version)

    nusc_eval = NuScenesEval(
        nusc,
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
    )
    metrics_summary = nusc_eval.main(plot_examples=10,)
