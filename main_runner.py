import json
import os
import time
import traceback
from typing import Tuple
import numpy as np
import cv2

from self_driving.beamng_brewer import BeamNGBrewer
from self_driving.beamng_tig_maps import maps
from self_driving.beamng_waypoint import BeamNGWaypoint
from self_driving.simulation_data import SimulationDataRecord, SimulationData
from self_driving.simulation_data_collector import SimulationDataCollector
from self_driving.utils import get_node_coords, points_distance
from self_driving.vehicle_state_reader import VehicleStateReader
from beamngpy.sensors import Camera



FloatDTuple = Tuple[float, float, float, float]
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3

class BeamNGCarCameras:
    def __init__(self):
        direction = (0, 1, 0)
        fov = 120
        resolution = (320, 160)
        y, z = 1.7, 1.0

        cam_center = 'cam_center', Camera((-0.3, y, z), direction, fov, resolution, colour=True, depth=True,
                                          annotation=True)
        cam_left = 'cam_left', Camera((-1.3, y, z), direction, fov, resolution, colour=True, depth=True,
                                      annotation=True)
        cam_right = 'cam_right', Camera((0.4, y, z), direction, fov, resolution, colour=True, depth=True,
                                        annotation=True)

        self.cameras_array = [cam_center, cam_left, cam_right]


class Executor():
    def __init__(self):
        self.brewer: BeamNGBrewer = None

    def evaluate(self, road_list):
        for sample_nodes in road_list:
            counter = 20
            attempt = 0
            condition = True
            while condition:
                attempt += 1
                if attempt == counter:
                    raise Exception('Exhausted attempts')
                if attempt > 1:
                    self._close()
                if attempt > 2:
                    time.sleep(5)
                sim = self._run_simulation(sample_nodes)
                if sim.info.success:
                    condition = False
                return(sim)

    def _run_simulation(self, nodes) -> SimulationData:
        if not self.brewer:
            self.brewer = BeamNGBrewer()
            self.vehicle = self.brewer.setup_vehicle()
            self.camera = self.brewer.setup_scenario_camera()

        brewer = self.brewer
        brewer.setup_road_nodes(nodes)
        beamng = brewer.beamng
        waypoint_goal = BeamNGWaypoint('waypoint_goal', get_node_coords(nodes[-1]))
        maps.install_map_if_needed()
        maps.beamng_map.generated().write_items(brewer.decal_road.to_json() + '\n' + waypoint_goal.to_json())

        cameras = BeamNGCarCameras()
        vehicle_state_reader = VehicleStateReader(self.vehicle, beamng, additional_sensors=cameras.cameras_array)
        brewer.vehicle_start_pose = brewer.road_points.vehicle_start_pose()

        steps = brewer.params.beamng_steps
        simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
        name = 'beamng_executor/sim_$(id)'.replace('$(id)', simulation_id)
        sim_data_collector = SimulationDataCollector(self.vehicle, beamng, brewer.decal_road, brewer.params,
                                                     vehicle_state_reader=vehicle_state_reader,
                                                     camera=self.camera,
                                                     simulation_name=name)

        sim_data_collector.get_simulation_data().start()
        try:
            brewer.bring_up()
            iterations_count = 1000
            idx = 0
            while True:
                idx += 1
                if idx >= iterations_count:
                    sim_data_collector.save()
                    raise Exception('Timeout simulation ', sim_data_collector.name)

                sim_data_collector.collect_current_data(oob_bb=True)
                #sim_data_collector.collect_current_data(oob_bb=False)
                last_state: SimulationDataRecord = sim_data_collector.states[-1]

                if points_distance(last_state.pos, waypoint_goal.position) < 8.0:
                    break

                if last_state.is_oob:
                    break

                import timeit
                start = timeit.default_timer()

                beamng.step(steps)

                #brewer.vehicle.ai_set_aggression(1)
                brewer.vehicle.ai_set_speed(7.0, mode='limit')
                brewer.vehicle.ai_drive_in_lane(True)
                brewer.vehicle.ai_set_waypoint(waypoint_goal.name)


                end = timeit.default_timer()
                #print("Time: ", end - start)

            sim_data_collector.get_simulation_data().end(success=True)
        except Exception as ex:
            sim_data_collector.get_simulation_data().end(success=False, exception=ex)
            traceback.print_exception(type(ex), ex, ex.__traceback__)
        finally:
            sim_data_collector.save()
            try:
                sim_data_collector.take_car_picture_if_needed()
            except:
                pass

            self.end_iteration()

        return sim_data_collector.simulation_data



    def end_iteration(self):
        try:
            if self.brewer:
                self.brewer.beamng.stop_scenario()
        except Exception as ex:
            traceback.print_exception(type(ex), ex, ex.__traceback__)

    def _close(self):
        if self.brewer:
            try:
                self.brewer.beamng.close()
            except Exception as ex:
                traceback.print_exception(type(ex), ex, ex.__traceback__)
            self.brewer = None


if __name__ == '__main__':
    ROAD_PATH = "data/member_seeds/population_HQ1/seed0.json"
    ROAD_PATH =r"data\seed0.json"
    MAX_SPEED = 25
    MIN_SPEED = 10
    inst = Executor()

    with open(ROAD_PATH, 'r') as f:
        dict = json.loads(f.read())
    sample_nodes = [tuple(t) for t in dict['sample_nodes']]

    # nodes should be a list of (x,y) float coordinates
    nodes = [sample[:2] for sample in sample_nodes]
    nodes = [(node[0],node[1],-28.0,8.0) for node in nodes]

    sim = inst.evaluate([nodes])
    #print(sim.states)

