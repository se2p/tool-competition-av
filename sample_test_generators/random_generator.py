from random import randint


class RandomTestGenerator():
    """
        This simple (naive) test generator creates roads using 4 points randomly placed on the map.
        We expect that this generator quickly creates plenty of tests, but many of them will be invalid as roads
        will likely self-intersect.
    """

    def __init__(self, time_budget=None, executor=None, map_size=None):
        self.time_budget = time_budget
        self.executor = executor
        self.map_size = map_size

    def start(self):

        while self.executor.get_remaining_time() > 0:
            # Some debugging
            print("Starting test generation. Remaining time ", self.executor.get_remaining_time())

            # Pick up random points from the map. They will be interpolated anyway to generate the road
            test = []
            for i in range(0, 3):
                test.append((randint(0, self.map_size), randint(0, self.map_size)))

            # Some more debugging
            print("Generated test: ", test)

            # Try to execute the test
            test_outcome, description, execution_data = self.executor.execute_test(test)

            # Print the result from the test and continue
            print("test_outcome ", test_outcome)
            print("description ", description)

