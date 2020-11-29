# TODO Create an abstract class to host the logic for calling the validation and the timing
# Use this class to output tests in predefined locations after execution
import json


from validation import TestValidator
from abc import ABC, abstractmethod

from scipy.interpolate import splev, splprep
from numpy.ma import arange
from numpy import repeat, linspace
from shapely.geometry import LineString


import time


class AbstractTestExecutor(ABC):

    start_time = None

    def __init__(self, time_budget=None, map_size=None):
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

    def execute_test(self, the_test):
        # TODO There is a mistmacth between the format of the_test and the format expected by the executors 4-tuple
        the_test_as_4tuple = [(float(t[0]), float(t[1]), -28.0, 8.0) for t in the_test]

        is_valid, validation_msg = self.validate_test(the_test_as_4tuple)

        if is_valid:
            test_outcome, description, execution_data = self._execute(the_test_as_4tuple)
            if test_outcome == "ERROR":
                # This indicates a generic error during the execution, usually caused by a malformed test that the
                # validation logic was not able to catch.
                return "INVALID", description, []
            else:
                # Valid, either pass or fail
                return test_outcome, description, execution_data
        else:
            return "INVALID", validation_msg, []

    def validate_test(self, the_test):
        print("Validating test")
        return self.test_validator.validate_test(the_test)

    def get_elapsed_time(self):
        return self.total_elapsed_time

    def get_remaining_time(self):
        return self.time_budget - (self.get_elapsed_time())

    @abstractmethod
    def _execute(self, the_test):
        if self.get_remaining_time() <= 0:
            raise TimeoutError("Time budget is over, cannot run more tests")
        pass


class MockExecutor(AbstractTestExecutor):

    def _execute(self, the_test):
        # Ensure we do not execute anything longer than the time budget
        super()._execute(the_test)

        print("(Random) Executing the test")
        test_outcome = "FAIL"
        description = "Not implemented"
        execution_data = []
        # TODO Make sure to reimplemet get_elapsed_time to include simulation time
        time.sleep(5)

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
