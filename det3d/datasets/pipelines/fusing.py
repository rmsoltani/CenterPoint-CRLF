import numpy as np
from scipy.spatial import KDTree
from ..registry import PIPELINES


@PIPELINES.register_module
class LidarPlusRadarFusion(object):
    def __init__(self, radar_feature_mask=None, max_fusion_radius=None, fusion_workers=-1, filter_unique_radar=False) -> None:
        """
        :param radar_feature_mask: Either a list/tuple contain the indexes of the used features or a boolean mask.
        If excluded, no features will be used. (x,y,z coors do are not counted as 'features')
        The features used must align with those used when creating the gt_database
        
        Feature indexes:
        0        1  2   3  4  5       6       7                8           9    10     11            12   13     14
        dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
        """

        self.radar_feature_mask = np.array([False, False, False], dtype=bool)
        self.max_fusion_radius = max_fusion_radius
        self.fusion_workers = fusion_workers
        self.filter_unique_radar = filter_unique_radar

        print("Filter unique:", self.filter_unique_radar)

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

        if self.filter_unique_radar:
            radar_points, radar_times = self._get_unique_radar(radar_points, radar_times)

        fused_points = self._get_fused_points(lidar_points, radar_points)
        fused_times = np.concatenate([lidar_times, radar_times], dtype=np.float32)
        
        res["lidar"]["points"] = fused_points
        res["lidar"]["times"] = fused_times
        res["lidar"]["combined"] = np.hstack([fused_points, fused_times])

        return res, info

    def _get_fused_points(self, lidar_points, radar_points):

        radar_coords = radar_points[:, :3]
        lidar_coords = lidar_points[:, :3]

        radar_features = radar_points[:, self.radar_feature_mask]

        extended_lidar_points = np.hstack((lidar_points, np.zeros([lidar_points.shape[0], radar_features.shape[1]], dtype=np.float32)), dtype=np.float32)
        extended_radar_points = np.hstack((radar_coords, np.zeros([radar_points.shape[0], 1], dtype=np.float32), radar_features), dtype=np.float32)

        for curr, closest in self._get_closest_index(lidar_coords, radar_coords):
            extended_lidar_points[curr, lidar_points.shape[1]:] = radar_features[closest]

        fused_points = np.concatenate([extended_lidar_points, extended_radar_points])

        return fused_points
    
    def _get_closest_index(self, lidar_coords, radar_coords):
        if self.max_fusion_radius is None:
            return

        tree = KDTree(radar_coords)
        curr = 0
        _, indecies = tree.query(lidar_coords, k=1, distance_upper_bound=self.max_fusion_radius, workers=self.fusion_workers)
        for i in indecies:
            if i != tree.n:
                yield curr, i
            curr += 1
        
    def _get_unique_radar(self, radar_points, radar_times):
        points, indexes =  np.unique(radar_points, return_index=True, axis=0)
        return points, radar_times[indexes]


