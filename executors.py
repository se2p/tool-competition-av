# TODO Create an abstract class to host the logic for calling the validation and the timing
# Use this class to output tests in predefined locations after execution
import json


from validation import TestValidator
from abc import ABC, abstractmethod

import time

class AbstractTestExecutor(ABC):

    start_time = None

    def __init__(self, time_budget=None, map_size=None):
        self.time_budget = time_budget
        self.test_validator = TestValidator(map_size)
        self.start_time = time.monotonic()

        super().__init__()

    # test_outcome, description, execution_data
    def execute_test(self, the_test):
        is_valid, validation_msg = self.validate_test(the_test)
        if is_valid:
            return self._execute(the_test)
        else:
            return "INVALID", validation_msg, []

    def validate_test(self, the_test):
        return self.test_validator.validate_test(the_test)

    def get_elapsed_time(self):
        return time.monotonic() - self.start_time

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
    executor = BeamngExecutor(time_budget=250000, map_size=250)
    ROAD_PATH = r"data\seed0.json"
    with open(ROAD_PATH, 'r') as f:
        dict = json.loads(f.read())
    sample_nodes = [tuple(t) for t in dict['sample_nodes']]

    # nodes should be a list of (x,y) float coordinates
    nodes = [sample[:2] for sample in sample_nodes]
    nodes = [(node[0], node[1], -28.0, 8.0) for node in nodes]

    tc = nodes
    test_outcome, description, execution_data= executor.execute_test(tc)
