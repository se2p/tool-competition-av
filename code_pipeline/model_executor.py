from code_pipeline.executors import AbstractTestExecutor

import time
import traceback

from models.config import ModelsConfig
from self_driving.beamng_brewer import BeamNGBrewer
from self_driving.beamng_tig_maps import maps, LevelsFolder
from self_driving.beamng_waypoint import BeamNGWaypoint
from self_driving.simulation_data import SimulationDataRecord, SimulationData
from self_driving.simulation_data_collector import SimulationDataCollector
from self_driving.utils import get_node_coords, points_distance
from self_driving.vehicle_state_reader import VehicleStateReader
from beamngpy.sensors import Camera
from abc import abstractmethod

from shapely.geometry import Point

import logging as log
import os.path
import cv2
import numpy

DRIVER_CAMERA_NAME = 'driver_view_camera'


def preprocess_image(image, resize=None):
    image_array = numpy.asarray(image)
    # removes sky and front of car
    image_array = image_array[80:-1, :, :]
    image_array = cv2.resize(image_array, resize, cv2.INTER_AREA)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2YUV)
    return image_array


class ModelExecutor(AbstractTestExecutor):

    def __init__(self, result_folder, time_budget, map_size,
                 oob_tolerance=0.95, max_speed=70,
                 beamng_home=None, beamng_user=None, road_visualizer=None):
        super(ModelExecutor, self).__init__(result_folder, time_budget, map_size)
        self.test_time_budget = 250000
        self.MIN_SPEED = 5.0
        self.MAX_SPEED = max_speed

        self.oob_tolerance = oob_tolerance
        self.maxspeed = max_speed
        self.brewer: BeamNGBrewer = None
        self.beamng_home = beamng_home
        self.beamng_user = beamng_user
        self.config = ModelsConfig()

        if self.beamng_user is not None and not os.path.exists(os.path.join(self.beamng_user, "research.key")):
            log.warning("%s is missing but is required to use BeamNG.research", )

        if self.config.model_path is None or not os.path.exists(self.config.model_path):
            log.warning("Required model not available")

        # Runtime Monitor about relative movement of the car
        self.last_observation = None
        # Not sure how to set this... How far can a car move in 250 ms at 5Km/h
        self.min_delta_position = 1.0
        self.road_visualizer = road_visualizer

    @abstractmethod
    def load_model(self):
        print('implement custom load operation')

    @abstractmethod
    def predict(self, image):
        print('implement custom predict operation')

    def _execute(self, the_test):
        # Ensure we do not execute anything longer than the time budget
        super()._execute(the_test)

        # TODO Show name of the test?
        log.info("Executing test %s", the_test.id)

        # TODO Not sure why we need to repeat this 2 times...
        counter = 2

        attempt = 0
        sim = None
        condition = True
        while condition:
            attempt += 1
            if attempt == counter:
                test_outcome = "ERROR"
                description = 'Exhausted attempts'
                break
            if attempt > 1:
                self._close()
            if attempt > 2:
                time.sleep(5)

            sim = self._run_simulation(the_test)

            if sim.info.success:
                if sim.exception_str:
                    test_outcome = "FAIL"
                    description = sim.exception_str
                else:
                    test_outcome = "PASS"
                    description = 'Successful test'
                condition = False

        execution_data = sim.states

        # TODO: report all test outcomes
        return test_outcome, description, execution_data

    def _is_the_car_moving(self, last_state):
        """ Check if the car moved in the past 10 seconds """

        # Has the position changed
        if self.last_observation is None:
            self.last_observation = last_state
            return True

        # If the car moved since the last observation, we store the last state and move one
        if Point(self.last_observation.pos[0], self.last_observation.pos[1]).distance(
                Point(last_state.pos[0], last_state.pos[1])) > self.min_delta_position:
            self.last_observation = last_state
            return True
        else:
            # How much time has passed since the last observation?
            if last_state.timer - self.last_observation.timer > 10.0:
                return False
            else:
                return True

    def _run_simulation(self, the_test) -> SimulationData:
        if not self.brewer:
            self.brewer = BeamNGBrewer(beamng_home=self.beamng_home, beamng_user=self.beamng_user)
            self.vehicle = self.brewer.setup_vehicle()

        # For the execution we need the interpolated points
        nodes = the_test.interpolated_points

        brewer = self.brewer
        brewer.setup_road_nodes(nodes)
        beamng = brewer.beamng
        waypoint_goal = BeamNGWaypoint('waypoint_goal', get_node_coords(nodes[-1]))

        # TODO Make sure that maps points to the right folder !
        if self.beamng_user is not None:
            beamng_levels = LevelsFolder(os.path.join(self.beamng_user, 'levels'))
            maps.beamng_levels = beamng_levels
            maps.beamng_map = maps.beamng_levels.get_map('tig')

        maps.install_map_if_needed()
        maps.beamng_map.generated().write_items(brewer.decal_road.to_json() + '\n' + waypoint_goal.to_json())

        # camera settings taken from https://github.com/testingautomated-usi/DeepHyperion/blob/main/DeepHyperion-BNG
        # /udacity_integration/beamng_car_cameras.py
        cam_pos = (-0.3, 1.7, 1.0)
        cam_dir = (0, 1, 0)
        cam_fov = 120
        cam_res = (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT)
        camera = (DRIVER_CAMERA_NAME, Camera(cam_pos, cam_dir, cam_fov, cam_res, colour=True, depth=True,
                                             annotation=True))
        additional_sensors = [camera]
        vehicle_state_reader = VehicleStateReader(self.vehicle, beamng, additional_sensors=additional_sensors)
        brewer.vehicle_start_pose = brewer.road_points.vehicle_start_pose()

        steps = brewer.params.beamng_steps
        simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
        name = 'model_executor/{}/sim_{}'.format(self.config.model_name, simulation_id)
        sim_data_collector = SimulationDataCollector(self.vehicle, beamng, brewer.decal_road, brewer.params,
                                                     vehicle_state_reader=vehicle_state_reader,
                                                     simulation_name=name)

        sim_data_collector.oob_monitor.tolerance = self.oob_tolerance

        sim_data_collector.get_simulation_data().start()
        try:
            brewer.bring_up()

            while True:
                # get camera image and preprocess it
                img = self.get_driver_camera_image()

                # predict steering angle
                steering_angle = float(self.predict(img))

                # show driver view with predicted steering angle
                img_cv = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
                # draw the same text a bit thicker to create a black outline around the text
                cv2.putText(img_cv, f"{steering_angle:.9f}", (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
                            cv2.LINE_AA)
                cv2.putText(img_cv, f"{steering_angle:.9f}", (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
                cv2.imshow('Predicted Steering Angle', img_cv)
                cv2.waitKey(1)

                sim_data_collector.collect_current_data(oob_bb=False)
                last_state: SimulationDataRecord = sim_data_collector.states[-1]

                if last_state.vel_kmh > self.MAX_SPEED:
                    self.speed_limit = self.MIN_SPEED  # slow down
                else:
                    self.speed_limit = self.MAX_SPEED

                # Based on the predicted steering angle the controller sets the speed:
                # for sharper turns will slow down more than for straight segments
                throttle = 1.0 - steering_angle ** 2 - (last_state.vel_kmh / self.speed_limit) ** 2

                # pass controls to vehicle
                self.vehicle.control(steering=steering_angle, throttle=throttle)

                # Target point reached
                if points_distance(last_state.pos, waypoint_goal.position) < 8.0:
                    break

                assert self._is_the_car_moving(last_state), "Car is not moving fast enough " + str(
                    sim_data_collector.name)

                assert not last_state.is_oob, "Car drove out of the lane " + str(sim_data_collector.name)

                beamng.step(steps)

            sim_data_collector.get_simulation_data().end(success=True)
            self.total_elapsed_time = self.get_elapsed_time()
        except AssertionError as aex:
            sim_data_collector.save()
            # An assertion that trigger is still a successful test execution, otherwise it will count as ERROR
            sim_data_collector.get_simulation_data().end(success=True, exception=aex)
            traceback.print_exception(type(aex), aex, aex.__traceback__)
        except Exception as ex:
            sim_data_collector.save()
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

    def get_driver_camera_image(self):
        sensors = self.brewer.beamng.poll_sensors(self.vehicle)
        cam = sensors[DRIVER_CAMERA_NAME]
        img = cam['colour'].convert('RGB')
        return img

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
