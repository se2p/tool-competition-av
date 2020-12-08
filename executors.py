# TODO Create an abstract class to host the logic for calling the validation and the timing
# Use this class to output tests in predefined locations after execution
import json


from validation import TestValidator
from abc import ABC, abstractmethod

from scipy.interpolate import splev, splprep
from numpy.ma import arange
from shapely.geometry import LineString

from self_driving.simulation_data import SimulationDataRecord

import random

import time


class TestGenerationStatistic:
    """
        Store statistics about test generation
        TODO: Refactor using a RoadTest and RoadTestExecution
    """

    def __init__(self):
        self.test_generated = 0
        self.test_valid = 0
        self.test_invalid = 0
        self.test_passed = 0
        self.test_failed = 0
        self.test_in_error = 0

        self.test_execution_real_times = []
        self.test_execution_simulation_times = []

        # TODO Capturing this is not that easy. We might approximate it as the time between consecutive
        #  calls to execute_test, but then we need to factor out how long it took to execute them... also
        #  it does not account for invalid tests...
        # self.last_generation_time = time.monotonic()
        # self.test_generation_times = []

    def __str__(self):
        msg = ""
        msg += "test generated: " + str(self.test_generated) + "\n"
        msg += "test valid: " + str(self.test_valid) + "\n"
        msg += "test invalid: " + str(self.test_invalid) + "\n"
        msg += "test passed: " + str(self.test_passed) + "\n"
        msg += "test failed: " + str(self.test_failed) + "\n"
        msg += "test in_error: " + str(self.test_in_error) + "\n"
        msg += "(real) time spent in execution :" + str(sum(self.test_execution_real_times)) + "\n"
        # self.test_execution_simulation_times = []
        return msg


class AbstractTestExecutor(ABC):

    start_time = None

    def __init__(self, time_budget=None, map_size=None):

        self.stats = TestGenerationStatistic()

        self.time_budget = time_budget
        self.test_validator = TestValidator(map_size)
        self.start_time = time.monotonic()
        self.total_elapsed_time = 0

        self.rounding_precision = 3
        self.min_num_nodes = 20
        # every meter more or less we need to place a node
        self.interpolation_distance = 1
        self.line_width = 0.15
        self.smoothness = 0

        super().__init__()

    # TODO Move this into RoadTest Class
    def _interpolate(self, the_test):
        old_x_vals = [t[0] for t in the_test ]
        old_y_vals = [t[1] for t in the_test]
        old_width_vals = [8.0 for t in the_test]

        # This is an approximation based on whatever input is given
        test_road_lenght = LineString([(t[0], t[1]) for t in the_test]).length
        num_nodes = int(test_road_lenght / self.interpolation_distance)
        if num_nodes < self.min_num_nodes:
            num_nodes = self.min_num_nodes

        k = 1 if len(old_x_vals) <= 3 else 3
        pos_tck, pos_u = splprep([old_x_vals, old_y_vals], s=self.smoothness, k=k)

        # Made this proportional to the lenght of the road

        step_size = 1 / num_nodes
        unew = arange(0, 1 + step_size, step_size)

        new_x_vals, new_y_vals = splev(unew, pos_tck)
        width_tck, width_u = splprep([pos_u, old_width_vals], s=self.smoothness, k=k)
        _, new_width_vals = splev(unew, width_tck)
        # Reduce floating point rounding errors otherwise these may cause problems with calculating parallel_offset
        # TODO Return the 4-tuple with standard z and width... this is bad but I cannot think of another solution
        return list(zip([round(v, self.rounding_precision) for v in new_x_vals],
                        [round(v, self.rounding_precision) for v in new_y_vals],
                        [-28.0 for v in new_x_vals],
                        [8.0 for v in new_x_vals]))

    # TODO Add type hint to the_test
    def execute_test(self, the_test):

        self.stats.test_generated += 1

        # TODO There is a mistmacth between the format of the_test and the format expected by the executors 4-tuple
        # This should be solved when we introduce RoadTest
        the_test_as_4tuple = [(float(t[0]), float(t[1]), -28.0, 8.0) for t in the_test]

        is_valid, validation_msg = self.validate_test(the_test_as_4tuple)

        if is_valid:
            self.stats.test_valid += 1
            start_execution_real_time = time.monotonic()

            test_outcome, description, execution_data = self._execute(the_test_as_4tuple)

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
                return test_outcome, description, execution_data
        else:
            self.stats.test_invalid += 1
            return "INVALID", validation_msg, []

    def validate_test(self, the_test):
        print("Validating test")
        return self.test_validator.validate_test(the_test)

    def get_elapsed_time(self):
        return self.total_elapsed_time

    def get_remaining_time(self):
        return self.time_budget - (self.get_elapsed_time())

    def get_stats(self):
        return self.stats

    @abstractmethod
    def _execute(self, the_test):
        if self.get_remaining_time() <= 0:
            raise TimeoutError("Time budget is over, cannot run more tests")
        pass


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

        print("Pretend test is executing")
        time.sleep(5)
        self.total_elapsed_time += 5

        return test_outcome, description, execution_data


if __name__ == '__main__':
    from beamng_executor import BeamngExecutor
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
