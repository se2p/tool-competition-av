from typing import Tuple

import numpy as np

from road_points import RoadPoints
from vehicle_state_reader import VehicleStateReader


class RoadVehicle:
    def __init__(self, vehicle_state_reader: VehicleStateReader, road: RoadPoints):
        self.road = road
        self.vehicle_state_reader = vehicle_state_reader

    def road_start_pose(self, meters_from_road_start=2.5, road_point_index=0) \
            -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        pose = self.road.vehicle_start_pose(meters_from_road_start, road_point_index)
        return pose.pos, pose.rot


if __name__ == '__main__':
    print('ok')
    import os

    print(os.getcwd())
    os.chdir('../udacity_integration')
    from road_storage import RoadStorage

    rs = RoadStorage()
    r1 = [[0.0, 0.0, -28.0, 8.0], [0.0, 2.0, -28.0, 8.0], [4.0, 0.0, -28.0, 8.0], [6.0, 0.0, -28.0, 8.0]]
    rp1 = RoadPoints().add_middle_nodes(r1)
    rv = RoadVehicle(None, rp1)
    print('vehicle start pose for this road:', rv.road_start_pose())
