from random import randint
from code_pipeline.tests_generation import RoadTestFactory
from time import  sleep
from code_pipeline.visualization import RoadTestVisualizer

import logging as log
from csv import reader
import os
import glob
from jpype import startJVM, shutdownJVM, java, addClassPath, JClass, JInt
import jpype
import jpype.imports
from jpype.types import *
import matplotlib.pyplot as plt

def read_points_from_csv(filename):
    list_of_rows = []
    with open(filename, 'r') as f:
        for line in f:
            values = line.strip().split(",")
            list_of_rows.append([int(values[0]), int(values[1])])
    print(list_of_rows)
    return list_of_rows

def get_tests(tests_dir):
    return glob.glob(os.path.join(tests_dir, '*.csv'))

class MBTGenerator():
    """
        This is a wrapper for the MBT generator in Java
    """

    def __init__(self, executor=None, map_size=None):
        self.executor = executor
        self.map_size = map_size

    def start(self):
        total_budget = 60  # get from the execution environment
        startJVM(convertStrings=False, classpath=['jars/mbt-1.0.2-jar-with-dependencies.jar'])
        from eu.fbk.iv4xr.mbt import SBSTMain
        mbt = SBSTMain(int(total_budget * 0.1), 'sbst2022.nine_states')
        log.info("MBT generated %s tests", mbt.totalTests())
        #test_files = get_tests('X:/projects/iv4xr/MBT/iv4xr-mbt/mbt-files/tests/sbst2022.nine_states/MOSA/1641376546606')
        #count = 0

        while not self.executor.is_over() and mbt.hasMoreTests():
            # Some debugging
            log.info(f"Starting test generation. Remaining time {self.executor.get_remaining_time()}")
            # Load the points from the csv file. They will be interpolated anyway to generate the road
            raw_mbt_points = mbt.getNextTest()  # read_points_from_csv(test_files[count])
            #count += 1
            road_points = []
            for mbt_point in raw_mbt_points:
                road_points.append([mbt_point[0], mbt_point[1]])

            # Some more debugging
            log.info("Generated test using: %s", road_points)
            # Decorate the_test object with the id attribute
            the_test = RoadTestFactory.create_road_test(road_points)
            # Some more debugging
            log.info("Decorated points: %s", the_test.interpolated_points)
            log.info(f"Remaining time {self.executor.get_remaining_time()}")

            # Visualise test
            RoadTestVisualizer(self.map_size).visualize_road_test(the_test, False)

            # Try to execute the test
            test_outcome, description, execution_data = self.executor.execute_test(the_test)
            log.info(f"Executed test {the_test.id}. Remaining time {self.executor.get_remaining_time()}")

            # oob_percentage = [state.oob_percentage for state in execution_data]
            # log.info("Collected %d states information. Max is %.3f", len(oob_percentage), max(oob_percentage))
            #
            # plt.figure()
            # plt.plot(oob_percentage, 'bo')
            # plt.show()

            # Print the result from the test and continue
            log.info("test_outcome %s", test_outcome)
            log.info("description %s", description)
        #java.lang.System.exit(0)
        #shutdownJVM()
        log.info("MBTGenerator has finished.")
