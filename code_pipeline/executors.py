# TODO Create an abstract class to host the logic for calling the validation and the timing
# Use this class to output tests in predefined locations after execution
import json
import logging as log
import random
import time
import sys

from abc import ABC, abstractmethod

from code_pipeline.validation import TestValidator
from code_pipeline.tests_generation import TestGenerationStatistic

from self_driving.simulation_data import SimulationDataRecord


class AbstractTestExecutor(ABC):

    start_time = None

    def __init__(self, time_budget=None, map_size=None, road_visualizer=None):

        self.stats = TestGenerationStatistic()

        self.time_budget = time_budget
        self.test_validator = TestValidator(map_size)
        self.start_time = time.monotonic()
        self.total_elapsed_time = 0

        self.road_visualizer = road_visualizer

        self.timeout_forced = False

        super().__init__()

    def is_force_timeout(self):
        return self.timeout_forced == True

    def execute_test(self, the_test):
        # Maybe we can solve this using decorators, but we need the reference to the instance, not sure how to handle
        # that cleanly
        if self.get_remaining_time() <= 0:
            self.timeout_forced = True
            log.warning("Time budget is over, cannot run more tests. FORCE EXIT")
            sys.exit(123)

        self.stats.test_generated += 1

        is_valid, validation_msg = self.validate_test(the_test)

        # This might be placed inside validate_test
        the_test.set_validity(is_valid, validation_msg)

        # Visualize the road if a road visualizer is defined. Also includes results for the validation
        if self.road_visualizer:
            self.road_visualizer.visualize_road_test(the_test)

        if is_valid:
            self.stats.test_valid += 1
            start_execution_real_time = time.monotonic()

            test_outcome, description, execution_data = self._execute(the_test)

            end_execution_real_time = time.monotonic()
            self.stats.test_execution_real_times.append(end_execution_real_time - start_execution_real_time)
            # Check that at least one element is there
            if execution_data and len(execution_data) > 0:
                self.stats.test_execution_simulation_times.append(execution_data[-1].timer)

            if test_outcome == "ERROR":
                self.stats.test_in_error += 1
                # This indicates a generic error during the execution, usually caused by a malformed test that the
                # validation logic was not able to catch.
                return "ERROR", description, []
            elif test_outcome == "PASS":
                self.stats.test_passed += 1
                return test_outcome, description, execution_data
            else:
                self.stats.test_failed += 1
                # Valid, either pass or fail
                if description.startswith("Car drove out of the lane "):
                    self.stats.obes += 1
                return test_outcome, description, execution_data
        else:
            self.stats.test_invalid += 1
            return "INVALID", validation_msg, []

    def validate_test(self, the_test):
        log.debug("Validating test")
        return self.test_validator.validate_test(the_test)

    def get_elapsed_time(self):
        return self.total_elapsed_time

    def get_remaining_time(self):
        return self.time_budget - (self.get_elapsed_time())

    def get_stats(self):
        return self.stats

    def close(self):
        log.info("CLOSING EXECUTOR")
        self._close()

    @abstractmethod
    def _close(self):
        if self.get_remaining_time() > 0:
            log.warning("Despite the time budget is not over executor is exiting!")

    @abstractmethod
    def _execute(self, the_test):
        # This should not be necessary, but better safe than sorry...
        if self.get_remaining_time() <= 0:
            self.timeout_forced = True
            log.warning("Time budget is over, cannot run more tests. FORCE EXIT")
            sys.exit(123)

class MockExecutor(AbstractTestExecutor):

    def _execute(self, the_test):
        # Ensure we do not execute anything longer than the time budget
        super()._execute(the_test)

        test_outcome = random.choice(["FAIL", "FAIL", "FAIL", "PASS", "PASS", "PASS", "PASS", "PASS", "ERROR"])
        description = "Mocked test results"


        sim_state = SimulationDataRecord(
            timer=3.0,
            pos= 0.0,
            dir= 0.0,
            vel= 0.0,
            steering= 0.0,
            steering_input= 0.0,
            brake= 0.0,
            brake_input= 0.0,
            throttle= 0.0,
            throttle_input= 0.0,
            wheelspeed= 0.0,
            vel_kmh = 0.0,
            is_oob= False,
            oob_counter = 0,
            max_oob_percentage = 0.0,
            oob_distance= 0.0,
        )

        execution_data = [sim_state]

        log.info("Pretend test is executing")
        time.sleep(5)
        self.total_elapsed_time += 5

        return test_outcome, description, execution_data

    def _close(self):
        super()._close()
        print("Closing Mock Executor")

if __name__ == '__main__':
    # TODO Remove this code and create an unit test instead
    from code_pipeline.beamng_executor import BeamngExecutor
    executor = BeamngExecutor(time_budget=250000, map_size=250, beamng_home=r"C:\Users\vinni\bng_competition\BeamNG.research.v1.7.0.0")
    ROAD_PATH = r"data\seed0.json"
    with open(ROAD_PATH, 'r') as f:
        dict = json.loads(f.read())
    sample_nodes = [tuple(t) for t in dict['sample_nodes']]

    # nodes should be a list of (x,y) float coordinates
    nodes = [sample[:2] for sample in sample_nodes]
    nodes = [(node[0], node[1], -28.0, 8.0) for node in nodes]

    tc = nodes
    test_outcome, description, execution_data= executor.execute_test(tc)
