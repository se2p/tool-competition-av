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



FloatDTuple = Tuple[float, float, float, float]

class BeamngExecutor(AbstractTestExecutor):

    def __init__(self, beamng_home = None, time_budget=None, map_size=None):
        super().__init__(time_budget, map_size)
        self.test_time_budget = 250000
        self.maxspeed = 7.0
        self.brewer: BeamNGBrewer = None
        self.beamng_home = beamng_home

    def _execute(self, the_test):
        # Ensure we do not execute anything longer than the time budget
        super()._execute(the_test)

        print("Executing the test")

        counter = 20
        attempt = 0
        sim = None
        condition = True
        while condition:
            attempt += 1
            if attempt == counter:
                test_outcome = "FAIL"
                description = 'Exhausted attempts'
                break
            if attempt > 1:
                self._close()
            if attempt > 2:
                time.sleep(5)
            sim = self._run_simulation(the_test)
            if sim.info.success:
                test_outcome = "SUCCESS"
                description = 'Successful test'
                condition = False

        execution_data = sim.states

        # TODO: report all test outcomes
        return test_outcome, description, execution_data

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
            import timeit
            start = timeit.default_timer()
            brewer.bring_up()
            iterations_count = int(self.test_time_budget/250)
            idx = 0
            while True:
                idx += 1
                if idx >= iterations_count:
                    sim_data_collector.save()
                    raise Exception('Timeout simulation ', sim_data_collector.name)

                sim_data_collector.collect_current_data(oob_bb=True)
                last_state: SimulationDataRecord = sim_data_collector.states[-1]

                if points_distance(last_state.pos, waypoint_goal.position) < 8.0:
                    break

                if last_state.is_oob:
                    break

                beamng.step(steps)

                #brewer.vehicle.ai_set_aggression(1)
                brewer.vehicle.ai_set_speed(self.maxspeed, mode='limit')
                brewer.vehicle.ai_drive_in_lane(True)
                brewer.vehicle.ai_set_waypoint(waypoint_goal.name)



                #print("Time: ", end - start)

            sim_data_collector.get_simulation_data().end(success=True)
            end = timeit.default_timer()
            run_elapsed_time = end - start
            self.total_elapsed_time += run_elapsed_time
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