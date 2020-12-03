from executors import AbstractTestExecutor
import time
import traceback
from typing import Tuple

from self_driving.beamng_brewer import BeamNGBrewer
from self_driving.beamng_tig_maps import maps
from self_driving.beamng_waypoint import BeamNGWaypoint
from self_driving.simulation_data import SimulationDataRecord, SimulationData
from self_driving.simulation_data_collector import SimulationDataCollector
from self_driving.utils import get_node_coords, points_distance
from self_driving.vehicle_state_reader import VehicleStateReader

from shapely.geometry import Point

FloatDTuple = Tuple[float, float, float, float]

class BeamngExecutor(AbstractTestExecutor):

    def __init__(self, beamng_home = None, time_budget=None, map_size=None):
        super().__init__(time_budget, map_size)
        self.test_time_budget = 250000
        # TODO Expose those as parameters
        self.maxspeed = 70.0
        self.risk_value = 0.7
        self.brewer: BeamNGBrewer = None
        self.beamng_home = beamng_home
        # Runtime Monitor about relative movement of the car
        self.last_observation = None
        # Not sure how to set this... How far can a car move in 250 ms at 5Km/h
        self.min_delta_position = 1.0

    def _execute(self, the_test):
        # Ensure we do not execute anything longer than the time budget
        super()._execute(the_test)

        # BeamNG requires the roads to be interpolated, as it cannot generate "squared" roads
        # This returns 4-tuple
        interpolated_test = self._interpolate(the_test)

        print("Executing the test")

        # TODO Not sure why we need to repeat this 20 times...
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

            sim = self._run_simulation(interpolated_test)

            if sim.info.success:
                if sim.exception_str:
                    test_outcome = "FAIL"
                    description = sim.exception_str
                else:
                    test_outcome = "SUCCESS"
                    description = 'Successful test'
                condition = False


        execution_data = sim.states

        # TODO: report all test outcomes
        return test_outcome, description, execution_data


    def _is_the_car_moving(self, last_state):
        """ Check if the car moved in the past 10 seconds """

        # Has the position changed
        if self.last_observation == None:
            self.last_observation = last_state
            return True

        # If the car moved since the last observation, we store the last state and move one
        if Point(self.last_observation.pos[0],self.last_observation.pos[1]).distance(Point(last_state.pos[0], last_state.pos[1])) > self.min_delta_position:
            self.last_observation = last_state
            return True
        else:
            # How much time has passed since the last observation?
            if last_state.timer - self.last_observation.timer > 10.0:
                return False
            else:
                return True

    def _run_simulation(self, nodes) -> SimulationData:
        if not self.brewer:
            self.brewer = BeamNGBrewer(self.beamng_home)
            self.vehicle = self.brewer.setup_vehicle()

        brewer = self.brewer
        brewer.setup_road_nodes(nodes)
        beamng = brewer.beamng
        waypoint_goal = BeamNGWaypoint('waypoint_goal', get_node_coords(nodes[-1]))
        maps.install_map_if_needed()
        maps.beamng_map.generated().write_items(brewer.decal_road.to_json() + '\n' + waypoint_goal.to_json())

        vehicle_state_reader = VehicleStateReader(self.vehicle, beamng, additional_sensors=None)
        brewer.vehicle_start_pose = brewer.road_points.vehicle_start_pose()

        steps = brewer.params.beamng_steps
        simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
        name = 'beamng_executor/sim_$(id)'.replace('$(id)', simulation_id)
        sim_data_collector = SimulationDataCollector(self.vehicle, beamng, brewer.decal_road, brewer.params,
                                                     vehicle_state_reader=vehicle_state_reader,
                                                     simulation_name=name)

        sim_data_collector.get_simulation_data().start()
        try:
            brewer.bring_up()
            iterations_count = int(self.test_time_budget/250)
            idx = 0

            brewer.vehicle.ai_set_aggression(self.risk_value)
            brewer.vehicle.ai_set_speed(self.maxspeed, mode='limit')
            brewer.vehicle.ai_drive_in_lane(True)
            brewer.vehicle.ai_set_waypoint(waypoint_goal.name)

            while True:
                idx += 1

                assert idx < iterations_count, "Timeout Simulation " + str(sim_data_collector.name)

                sim_data_collector.collect_current_data(oob_bb=True)
                last_state: SimulationDataRecord = sim_data_collector.states[-1]
                # Target point reached
                if points_distance(last_state.pos, waypoint_goal.position) < 8.0:
                    break

                assert self._is_the_car_moving(last_state), "Car is not moving fast enough " + str(sim_data_collector.name)

                assert not last_state.is_oob, "Car drove out of the lane " + str(sim_data_collector.name)

                beamng.step(steps)

            sim_data_collector.get_simulation_data().end(success=True)
            run_elapsed_time = float(last_state.timer)
            self.total_elapsed_time += run_elapsed_time
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