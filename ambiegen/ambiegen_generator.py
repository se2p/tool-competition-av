from code_pipeline.tests_generation import RoadTestFactory
from time import sleep
import logging as log
import ambiegen.Optimize as optim



class AmbieGenTestGenerator:
    """
    This test generator creates road points using affine tratsformations to vectors.
    Initially generated test cases are optimized by NSGA2 algorithm with two objectives:
    fault revealing power and diversity. We use a simplified model of a vehicle to
    estimate the fault revealing power (as the maximum deviation from the road center).
    We use 100 generations and 100 population size. In each iteration of the generator 
    the Pareto optimal solutions are provided and executed. Then the algorithm is launched again.
    """

    def __init__(self, time_budget=None, executor=None, map_size=None):
        self.map_size = map_size
        self.time_budget = time_budget
        self.executor = executor

    def start(self):

        while not self.executor.is_over():

            cases = optim.optimize()

            for case in cases:

                # Some debugging
                log.info(
                    "Starting test generation. Remaining time %s",
                    self.executor.get_remaining_time(),
                )

                the_test = RoadTestFactory.create_road_test(cases[case])

                # Try to execute the test
                test_outcome, description, execution_data = self.executor.execute_test(
                    the_test
                )

                log.info("test_outcome %s", test_outcome)
                log.info("description %s", description)

                if self.executor.road_visualizer:
                    sleep(1)
