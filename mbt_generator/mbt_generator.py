from random import randint
from code_pipeline.tests_generation import RoadTestFactory
from time import  sleep
from code_pipeline.visualization import RoadTestVisualizer

import logging as log
from csv import reader
import os
import glob
from code_pipeline.tests_generation import RoadTestFactory
from self_driving.bbox import RoadBoundingBox
import numpy as np
import logging as log
from self_driving.road_polygon import RoadPolygon
from scipy.interpolate import splev, splprep
from numpy.ma import arange
from shapely.geometry import LineString
import subprocess


def read_points_from_csv(filename):
    list_of_rows = []
    print(filename)
    with open(filename, 'r') as f:
        for line in f:
            values = line.strip().split(",")
            list_of_rows.append([float(values[0]), float(values[1])])
    print(list_of_rows)
    return list_of_rows


def get_tests(tests_dir):
    return glob.glob(os.path.join(tests_dir, '*.csv'))


class MBTGenerator:
    """
        This is a wrapper for the MBT generator in Java
    """

    def __init__(self, executor=None, map_size=None):
        self.executor = executor
        self.map_size = map_size

    def run_mbt(self, generation_budget, map_size, tests_dir):
        subprocess.call(['java', '-jar', 'mbt-1.0.2-jar-with-dependencies.jar', str(generation_budget), str(map_size), tests_dir])

    def start(self):
        # temporary directory where MBT writes tests
        tests_dir = 'tmp_mbt'

        generation_budget = self.executor.get_remaining_time()["generation-budget"]
        # proportion represents the percentage of the generation-budget
        # to be used by MBT for test generation
        proportion = 0.8
        mbt_generation_budget = int(generation_budget * proportion)
        log.info("Starting test generation using MBT with generation budget %i ...", mbt_generation_budget)
        self.run_mbt(mbt_generation_budget, self.map_size, tests_dir)

        test_files = get_tests(tests_dir)
        total_tests = len(test_files)
        log.info("MBT generated %s tests with budget %i", total_tests, mbt_generation_budget)

        test_number = 0
        cnt_invalid = 0
        while not self.executor.is_over() and test_number < total_tests:
            log.info(f"Starting test execution. Remaining time {self.executor.get_remaining_time()}")
            log.info("Test %i", test_number+1)
            # Load the points from the csv file
            road_points = read_points_from_csv(test_files[test_number])
            if self.invalid(road_points, self.map_size):
                test_number += 1
                cnt_invalid += 1
                # for debugging purpose comment continue and compare with competition check
                continue

            # Some more debugging
            log.info("Generated test using: %s", road_points)
            # Decorate the_test object with the id attribute
            the_test = RoadTestFactory.create_road_test(road_points)
            # Some more debugging
            log.info("Decorated points: %s", the_test.interpolated_points)
            log.info(f"Remaining time {self.executor.get_remaining_time()}")

            # Try to execute the test
            test_outcome, description, execution_data = self.executor.execute_test(the_test)
            log.info(f"Executed test {the_test.id}. Remaining time {self.executor.get_remaining_time()}")

            # Print the result from the test and continue
            log.info("test_outcome %s", test_outcome)
            log.info("description %s", description)
            test_number += 1

        log.info("total invalid generated %i", cnt_invalid)
        log.info("MBTGenerator has finished.")

    def invalid(self, road_points, map_size):
        # Constants
        rounding_precision = 3
        interpolation_distance = 1
        smoothness = 0
        min_num_nodes = 20

        box = (0, 0, map_size, map_size)
        road_bbox = RoadBoundingBox(box)

        def is_too_short(the_test):
            if len(the_test) < 4:
                return True
            return False

        def _interpolate(the_test):
            """
                Interpolate the road points using cubic splines and ensure we handle 4F tuples for compatibility
            """
            old_x_vals = [t[0] for t in the_test]
            old_y_vals = [t[1] for t in the_test]

            # This is an approximation based on whatever input is given
            test_road_lenght = LineString([(t[0], t[1]) for t in the_test]).length
            num_nodes = int(test_road_lenght / interpolation_distance)
            if num_nodes < min_num_nodes:
                num_nodes = min_num_nodes

            assert len(old_x_vals) >= 2, "You need at leas two road points to define a road"
            assert len(old_y_vals) >= 2, "You need at leas two road points to define a road"

            if len(old_x_vals) == 2:
                # With two points the only option is a straight segment
                k = 1
            elif len(old_x_vals) == 3:
                # With three points we use an arc, using linear interpolation will result in invalid road tests
                k = 2
            else:
                # Otheriwse, use cubic splines
                k = 3

            pos_tck, pos_u = splprep([old_x_vals, old_y_vals], s=smoothness, k=k)

            step_size = 1 / num_nodes
            unew = arange(0, 1 + step_size, step_size)

            new_x_vals, new_y_vals = splev(unew, pos_tck)

            # Return the 4-tuple with default z and defatul road width
            return list(zip([round(v, rounding_precision) for v in new_x_vals],
                            [round(v, rounding_precision) for v in new_y_vals],
                            [-28.0 for v in new_x_vals],
                            [8.0 for v in new_x_vals]))

        def find_circle(p1, p2, p3):
            """
            Returns the center and radius of the circle passing the given 3 points.
            In case the 3 points form a line, returns (None, infinity).
            """
            temp = p2[0] * p2[0] + p2[1] * p2[1]
            bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
            cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
            det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

            if abs(det) < 1.0e-6:
                return np.inf

            # Center of circle
            cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
            cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

            radius = np.sqrt((cx - p1[0]) ** 2 + (cy - p1[1]) ** 2)
            return radius

        def min_radius(x, w=5):
            mr = np.inf
            nodes = x
            for i in range(len(nodes) - w):
                p1 = nodes[i]
                p2 = nodes[i + int((w - 1) / 2)]
                p3 = nodes[i + (w - 1)]
                radius = find_circle(p1, p2, p3)
                if radius < mr:
                    mr = radius
            if mr == np.inf:
                mr = 0

            return mr * 3.280839895  # , mincurv

        def is_too_sharp(the_test, TSHD_RADIUS=47):
            if TSHD_RADIUS > min_radius(the_test) > 0.0:
                check = True

            else:
                check = False

            return check

        def is_inside_map(the_test):
            """
                Take the extreme points and ensure that their distance is smaller than the map side
            """
            xs = [t[0] for t in the_test]
            ys = [t[1] for t in the_test]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            return 0 < min_x or min_x > self.map_size and \
                   0 < max_x or max_x > self.map_size and \
                   0 < min_y or min_y > self.map_size and \
                   0 < max_y or max_y > self.map_size and the_test.get_road_length() > 20

        # return RoadPolygon.from_nodes(_interpolate(road_points)).is_valid()
        return road_bbox.intersects_boundary(RoadPolygon.from_nodes(_interpolate(road_points)).polygon) \
               or not RoadPolygon.from_nodes(_interpolate(road_points)).is_valid() \
               or not is_inside_map(_interpolate(road_points)) \
               or is_too_sharp(_interpolate(road_points)) \
               or is_too_short(road_points)