import numpy as np
from ..registry import PIPELINES


@PIPELINES.register_module
class LidarPlusRadarFusion(object):
    def __init__(self, radar_feature_mask=None) -> None:
        """
        :param radar_feature_mask: Either a list/tuple contain the indexes of the used features or a boolean mask.
        If excluded, no features will be used. (x,y,z coors do are not counted as 'features')
        The features used must align with those used when creating the gt_database
        
        Feature indexes:
        0        1  2   3  4  5       6       7                8           9    10     11            12   13     14
        dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
        """

        self.radar_feature_mask = np.array([False, False, False], dtype=bool)

        if radar_feature_mask is not None:
            radar_feature_mask = np.array(radar_feature_mask)

            if radar_feature_mask.dtype == bool: 
                assert len(radar_feature_mask == 15), "Feature mask contain map to all 15 features"
                self.radar_feature_mask = np.concatenate([
                    self.radar_feature_mask, 
                    np.array(radar_feature_mask, dtype=bool)
                ])
            elif radar_feature_mask.dtype == int:
                self.radar_feature_mask = np.concatenate([
                    self.radar_feature_mask,
                    np.array([i in radar_feature_mask for i in range(15)], dtype=bool),
                ])
            else:
                raise ValueError("Invalid radar_feature_mask")
            

    def __call__(self, res, info):
        lidar_points = res["lidar"]["points"]
        lidar_times = res["lidar"]["times"]
        radar_points = res["radar"]["points"]
        radar_times = res["radar"]["times"]

        fused_points = self._get_fused_points(lidar_points, radar_points)
        fused_times = np.concatenate([lidar_times, radar_times], dtype=np.float32)
        
        res["lidar"]["points"] = fused_points
        res["lidar"]["times"] = fused_times
        res["lidar"]["combined"] = np.hstack([fused_points, fused_times])

        return res, info

    def _get_fused_points(self, lidar_points, radar_points):
        extended_lidar_points = np.hstack((lidar_points, np.zeros([lidar_points.shape[0], 3], dtype=np.float32)), dtype=np.float32)

        radar_xyz = radar_points[:, :3]
        radar_features = radar_points[:, self.radar_feature_mask]
        extended_radar_points = np.hstack((radar_xyz, np.zeros([radar_points.shape[0], 1], dtype=np.float32), radar_features), dtype=np.float32)


        fused_points = np.concatenate([extended_lidar_points, extended_radar_points])

        return fused_points


