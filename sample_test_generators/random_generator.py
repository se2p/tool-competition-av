from random import randint


class RandomTestGenerator():

    def __init__(self, time_budget=None, executor=None, map_size=None):
        self.time_budget = time_budget
        self.executor = executor
        self.map_size = map_size

    def start(self):
        print("Starting test generation")
        while self.executor.get_remaining_time() > 0:
            print("Starting test generation. Remaining time ", self.executor.get_remaining_time())

            test = []
            for i in range(1, 3):
                test.append((randint(0, self.map_size), randint(0, self.map_size)))

            print("Generated test: ", test)
            test_outcome, description, execution_data = self.executor.execute_test(test)

            print("test_outcome ", test_outcome)
            print("description ", description)
            print("execution_data ", execution_data)

