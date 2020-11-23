from random import randint


class OneTestGenerator():

    def __init__(self, time_budget=None, executor=None, map_size=None):
        self.time_budget = time_budget
        self.executor = executor
        self.map_size = map_size

    def start(self):
        print("Starting test generation")
        # Generate a single test.

        test = []

        test.append( (0, 0) )
        # Curves cannot be too sharp?
        test.append( (0, 80) )
        test.append( (0, 100) )
        test.append( (100, 100) )
        test.append( (100, 0) )

        print("Generated test: ", test)
        test_outcome, description, execution_data = self.executor.execute_test(test)

        #print(execution_data)
        # Check that data is timestamped
        pass

