#
#
#
#
import datetime
import json
import math

import click
import importlib
import traceback
import time
import os
import sys
import errno
import logging as log
import csv
import glob
import logging
import itertools
import statistics

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import tee
from itertools import combinations
from typing import List, Tuple

from code_pipeline.visualization import RoadTestVisualizer
from code_pipeline.tests_generation import TestGenerationStatistic
from code_pipeline.test_generation_utils import register_exit_fun
from self_driving.simulation_data import SimulationDataRecord
from code_pipeline.tests_evaluation import OOBAnalyzer

AngleLength = Tuple[float, float]
ListOfAngleLength = List[AngleLength]

Point = Tuple[float, float]
ListOfPoints = List[Point]
# TODO Make this configurable?

OUTPUT_RESULTS_TO = 'results'
THE_NORTH = [0, 1]


def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def validate_speed_limit(ctx, param, value):
    """
    The speed limit must be a positive integer greater than 10 km/h (lower values might trigger the
    car-not-moving oracle
    """
    if int(value) < 10:
        raise click.UsageError(
            'The provided value for ' + str(param) + ' is invalid. Choose a value greater than 10')
    else:
        return int(value)


def validate_oob_tolerance(ctx, param, value):
    """
    OOB tolerance must be a value between 0.0 and 1.0
    """
    if value < 0.0 or value > 1.0:
        raise click.UsageError(
            'The provided value for ' + str(param) + ' is invalid. Choose a value between 0.0 and 1.0')
    else:
        return value


def validate_map_size(ctx, param, value):
    """
    The size of the map is defined by its edge. The edge can be any (integer) value between 100 and 1000
    """
    if int(value) < 100 or int(value) > 1000:
        raise click.UsageError(
            'The provided value for ' + str(param) + ' is invalid. Choose an integer between 100 and 1000')
    else:
        return int(value)


def validate_time_budget(ctx, param, value):
    """
    A valid time budget is a positive integer of 'seconds'
    """
    if int(value) < 1:
        raise click.UsageError('The provided value for ' + str(param) + ' is invalid. Choose any positive integer')
    else:
        return int(value)


def create_experiment_description(result_folder, params_dict):
    log.info("Creating Experiment Description")
    experiment_description_file = os.path.join(result_folder, "experiment_description.csv")
    csv_columns = params_dict.keys()
    try:
        with open(experiment_description_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerow(params_dict)
            log.info("Experiment Description available: %s", experiment_description_file)
    except IOError:
        log.error("I/O error. Cannot write Experiment Description")


def create_summary(result_folder, raw_data):
    log.info("Creating Reports")

    # Refactor this
    if type(raw_data) is TestGenerationStatistic:
        log.info("Creating Test Statistics Report:")
        summary_file = os.path.join(result_folder, "generation_stats.csv")
        csv_content = raw_data.as_csv()
        with open(summary_file, 'w') as output_file:
            output_file.write(csv_content)
        log.info("Test Statistics Report available: %s", summary_file)

    log.info("Creating OOB Report")
    oobAnalyzer = OOBAnalyzer(result_folder)
    oob_summary_file = os.path.join(result_folder, "oob_stats.csv")
    csv_content = oobAnalyzer.create_summary()
    with open(oob_summary_file, 'w') as output_file:
        output_file.write(csv_content)

    log.info("OOB  Report available: %s", oob_summary_file)


def load_the_interpolated_points_from_json_file(path_to_json):
    '''
    loads interpolation points (sample_points) from the json file
    path_to_json is the path + the name of the json file
    '''
    # path_json = "./results/" + folder + "/" + param
    with open(path_to_json, 'r', encoding='utf-8') as f:
        obj = json.load(f)
        return obj['interpolated_points']


def load_test_data_from_json_file(path_to_json):
    '''
    loads all the test data from the json file
    path_to_json is the path + the name of the json file
    '''
    # path_json = "./results/" + folder + "/" + param
    with open(path_to_json, 'r', encoding='utf-8') as f:
        obj = json.load(f)
        return obj


def load_the_simulation_records_data_from_json(path_to_json):
    '''
    loads simulation_data_record from the json file
    path_to_json is the path + the name of the json file
    '''
    # path_json = "./results/" + folder + "/" + param
    with open(path_to_json, 'r') as f:
        obj = json.loads(f.read())

    states = [SimulationDataRecord(*r) for r in obj["execution_data"]]
    return states


def define_circle(p1, p2, p3):
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


def min_radius(nodes, w=5):
    '''
    calculates the value of 'min_radius' feature dimension. Nodes are interpolated_points
    '''
    mr = np.inf
    mincurv = []
    # nodes = x.roads.nodes # x.m.sample_nodes
    for i in range(len(nodes) - w):
        p1 = nodes[i]
        p2 = nodes[i + int((w - 1) / 2)]
        p3 = nodes[i + (w - 1)]
        # radius = findCircle(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
        radius = define_circle(p1, p2, p3)
        if radius < mr:
            mr = radius
            mincurv = [p1, p2, p3]

    if mr > 90:
        mr = 90

    return int(mr * 3.280839895)  # , mincurv


# def tee(iterable, n=2):  # real signature unknown; restored from __doc__
#     """ tee(iterable, n=2) --> tuple of n independent iterators. """
#     pass


# https://docs.python.org/3/library/itertools.html
# Itertools Recipes
def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def direction_coverage(nodes, n_bins=25):
    """Measure the coverage of road directions w.r.t. to the North (0,1) using the control points of the given road
    to approximate the road direction. BY default we use 36 bins to have bins of 10 deg each"""
    # Note that we need n_bins+1 because the interval for each bean is defined by 2 points
    coverage_buckets = np.linspace(0.0, 360.0, num=n_bins + 1)
    direction_list = []
    for a, b in _pairwise(nodes):  # (x.m.sample_nodes):
        # Compute the direction of the segment defined by the two points
        road_direction = [b[0] - a[0], b[1] - a[1]]
        # Compute the angle between THE_NORTH and the road_direction.
        # E.g. see: https://www.quora.com/What-is-the-angle-between-the-vector-A-2i+3j-and-y-axis
        # https://www.kite.com/python/answers/how-to-get-the-angle-between-two-vectors-in-python
        unit_vector_1 = road_direction / np.linalg.norm(road_direction)
        dot_product = np.dot(unit_vector_1, THE_NORTH)
        angle = math.degrees(np.arccos(dot_product))
        direction_list.append(angle)

    # Place observations in bins and get the covered bins without repetition
    covered_elements = set(np.digitize(direction_list, coverage_buckets))
    return int((len(covered_elements) / len(coverage_buckets)) * 100)


def mean_lateral_position(x):
    states = x  # x.m.simulation.states
    lp = []
    for state in states:
        lp.append(state.oob_distance)
    mean_lp = np.mean(lp) * 100
    return int(mean_lp)


def segment_count(nodes):
    # nodes = x.road.nodes # x.m.sample_nodes

    count, segments = identify_segment(nodes)
    return count  # , segments
    # TODO Note that this is identify_segments with a final 's'
    # segments = identify_segments(nodes)
    # return len(segments), segments


def _calc_angle_distance(v0, v1):
    at_0 = np.arctan2(v0[1], v0[0])
    at_1 = np.arctan2(v1[1], v1[0])
    return at_1 - at_0


def _calc_dist_angle(points: ListOfPoints) -> ListOfAngleLength:
    assert len(points) >= 2, f'at least two points are needed'

    def vector(idx):
        return np.subtract(points[idx + 1], points[idx])

    n = len(points) - 1
    result: ListOfAngleLength = [None] * (n)
    b = vector(0)
    for i in range(n):
        a = b
        b = vector(i)
        angle = _calc_angle_distance(a, b)
        distance = np.linalg.norm(b)
        result[i] = (angle, distance, [points[i + 1], points[i]])
    return result


# counts only turns, split turns
def identify_segment(nodes):
    # result is angle, distance, [x2,y2], [x1,y1]
    result = _calc_dist_angle(nodes)

    segments = []
    SEGMENT_THRESHOLD = 15
    SEGMENT_THRESHOLD2 = 10
    ANGLE_THRESHOLD = 0.005

    # iterate over the nodes to get the turns bigger than the threshold
    # a turn category is assigned to each node
    # l is a left turn
    # r is a right turn
    # s is a straight segment
    # TODO: first node is always a s
    turns = []
    for i in range(0, len(result)):
        # result[i][0] is the angle
        angle_1 = (result[i][0] + 180) % 360 - 180
        if np.abs(angle_1) > ANGLE_THRESHOLD:
            if (angle_1) > 0:
                turns.append("l")
            else:
                turns.append("r")
        else:
            turns.append("s")

    # this generator groups the points belonging to the same category
    def grouper(iterable):
        prev = None
        group = []
        for item in iterable:
            if not prev or item == prev:
                group.append(item)
            else:
                yield group
                group = [item]
            prev = item
        if group:
            yield group

    # this generator groups:
    # - groups of points belonging to the same category
    # - groups smaller than 10 elements
    def supergrouper1(iterable):
        prev = None
        group = []
        for item in iterable:
            if not prev:
                group.extend(item)
            elif len(item) < SEGMENT_THRESHOLD2 and item[0] == "s":
                item = [prev[-1]] * len(item)
                group.extend(item)
            elif len(item) < SEGMENT_THRESHOLD and item[0] != "s" and prev[-1] == item[0]:
                item = [prev[-1]] * len(item)
                group.extend(item)
            else:
                yield group
                group = item
            prev = item
        if group:
            yield group

    # this generator groups:
    # - groups of points belonging to the same category
    # - groups smaller than 10 elements
    def supergrouper2(iterable):
        prev = None
        group = []
        for item in iterable:
            if not prev:
                group.extend(item)
            elif len(item) < SEGMENT_THRESHOLD:
                item = [prev[-1]] * len(item)
                group.extend(item)
            else:
                yield group
                group = item
            prev = item
        if group:
            yield group

    groups = grouper(turns)

    supergroups1 = supergrouper1(groups)

    supergroups2 = supergrouper2(supergroups1)

    count = 0
    segment_indexes = []
    segment_count = 0
    for g in supergroups2:
        if g[-1] != "s":
            segment_count += 1
        # TODO
        # count += (len(g) - 1)
        count += (len(g))
        # TODO: count -1?
        segment_indexes.append(count)

    # TODO
    # segment_indexes.append(len(turns) - 1)

    segment_begin = 0
    for idx in segment_indexes:
        segment = []
        # segment_end = idx + 1
        segment_end = idx
        for j in range(segment_begin, segment_end):
            if j == 0:
                segment.append([result[j][2][0], result[j][0]])
            segment.append([result[j][2][1], result[j][0]])
        segment_begin = segment_end
        segments.append(segment)

    return segment_count, segments


def sd_steering(x):
    states = x  # x.m.simulation.states
    steering = []
    for state in states:
        steering.append(state.steering)
    sd_steering = np.std(steering)
    return int(sd_steering)


def curvature(nodes, w=5):
    mr = np.inf
    mincurv = []
    # nodes = x.road.nodes # x.m.sample_nodes
    for i in range(len(nodes) - w):
        p1 = nodes[i]
        p2 = nodes[i + int((w - 1) / 2)]
        p3 = nodes[i + (w - 1)]
        # radius = findCircle(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
        radius = define_circle(p1, p2, p3)
        if radius < mr:
            mr = radius
            mincurv = [p1, p2, p3]

    curvature = (1 / mr) * 100

    return int(curvature)  # , mincurv


def fitness_function(results):
    if results['test_outcome'] == "PASS":
        return 0
    else:
        return 1


def compute_features_for_the_simulation(folder):
    """
    creates a dictionary with feature values for simulations of the test in the folder 'folder'.
     - key 'MinRadius'
     - key 'DirectionCoverage'
     - key 'MeanLateralPosition'
     - key 'SegmentCount'
     - key 'SDSteeringAngle'
     - key 'Curvature'
      and the fitness value of this test
      - key 'FitnessFunction'
      """

    # creates an empty list of the featuresValues
    feature_values_dict = {}

    # fetches simulation data from json file
    test_data = load_test_data_from_json_file(folder)
    if (test_data['is_valid']):
        test_data = load_test_data_from_json_file(folder)
        interpolated_points = load_the_interpolated_points_from_json_file(folder)
        states = load_the_simulation_records_data_from_json(folder)

        # calculates the value of 'MinRadius' feature
        min_Radius_Value = min_radius(interpolated_points, )

        feature_values_dict['MinRadius'] = min_Radius_Value

        # calculates the value of 'DirectionCoverage' feature
        direction_Coverage = direction_coverage(interpolated_points, )

        feature_values_dict['DirectionCoverage'] = direction_Coverage

        # calculates the value of 'MeanLateralPosition' feature
        mean_Lateral_Position = mean_lateral_position(states)

        feature_values_dict['MeanLateralPosition'] = mean_Lateral_Position

        # calculates the value of 'SegmentCount' feature
        segment_count_value = segment_count(interpolated_points)

        feature_values_dict['SegmentCount'] = segment_count_value

        # calculates the value of 'SDSteeringAngle' feature
        sds_steering_angle_value = sd_steering(states)

        feature_values_dict['SDSteeringAngle'] = sds_steering_angle_value

        # calculates the value of 'Curvature' feature
        curvature_value = curvature(interpolated_points, )

        feature_values_dict['Curvature'] = curvature_value

        # calculates fitness function for the test
        # the value is 1 if the test has failed and 0 if the test has passed
        fitness_function_value = fitness_function(test_data)
        feature_values_dict['FitnessFunction'] = fitness_function_value

        # adds test_data to the dictionary
        feature_values_dict['TestData'] = test_data

        # adds interpolated_points to the dictionary
        feature_values_dict['InterpolatedPoints'] = interpolated_points

        # adds states to the dictionary
        feature_values_dict['States'] = states

        return feature_values_dict


def create_array_of_computed_features_dictionaries_of_all_tests():
    '''
    creates a list of Dictionaries, that contains fitness function and all features
    of all 'test.0001.json' files in the 'results' folder
    '''
    list_of_dictionaries_of_computed_features = []
    path_to_results = "../tool-competition-av/results/sample_test_generators.*/test.*.json"

    # returns paths to test.0001.json files
    list_of_paths_to_json_files = glob.glob(path_to_results)
    for path in list_of_paths_to_json_files:
        dictionary_of_features = compute_features_for_the_simulation(path)
        # if the simulation data is not valid
        if (dictionary_of_features != None):
            list_of_dictionaries_of_computed_features.append(compute_features_for_the_simulation(path))

    return list_of_dictionaries_of_computed_features


def find_max_value_of_feature(feature_name):
    list_of_dictionaries = create_array_of_computed_features_dictionaries_of_all_tests()
    max_value = float('-inf')
    for dictionary in list_of_dictionaries:
        if (dictionary[feature_name] > max_value):
            max_value = dictionary[feature_name]
    return max_value


def find_min_value_of_feature(feature_name):
    # TODO to prove that list of dictionaries is not None
    list_of_dictionaries = create_array_of_computed_features_dictionaries_of_all_tests()
    min_value = float('inf')
    for dictionary in list_of_dictionaries:
        if (dictionary[feature_name] < min_value):
            min_value = dictionary[feature_name]
    return min_value


def create_array_of_features_for_axis(names_of_features):
    array_of_features = []
    for feature in names_of_features:
        min_value = find_min_value_of_feature(feature)
        max_value = find_max_value_of_feature(feature)
        feature_array = []
        feature_array.append(feature)
        feature_array.append(min_value)
        feature_array.append(max_value)
        feature_array.append(5) # TODO how to put a value that equal the number of cells
        array_of_features.append(feature_array)
    return array_of_features

def manhattan(coords_ind1, coords_ind2):
    return abs(coords_ind1[0] - coords_ind2[0]) + abs(coords_ind1[1] - coords_ind2[1])

def drop_outliers_for(feature, samples):
    """
    Return a list of samples that have min/max. There might be a more pythonic way of doing it using a filter
    """
    return [s for s in samples if not feature.is_outlier(s)]

# https://stackoverflow.com/questions/34630772/removing-duplicate-elements-by-their-attributes-in-python
def deduplicate(items):
    """
    Make sure that items with the same ID are removed
    TODO This might be also possible to refine
    Args:
        items:

    Returns:

    """
    seen = set()
    for item in items:
        if not item.id in seen:
            seen.add(item.id)
            yield item
        else:
            logging.debug("Removing duplicated sample %s", item.id)

def select_samples_by_elapsed_time(max_elapsed_minutes, min_elapsed_minutes=0):
    """
    Consider the samples only if their elapsed time is between min and max. Min is defaulted to 0.0.
    Everything is expressed in minutes

    Args:
        max_elapsed_minutes:
        min_elapsed_minutes

    Returns:

    """

    def _f(samples):
        # Convert minutes in seconds. Note we need to account for minutes that are over the hour, but also that times
        # must all have the same year (gmtime and strptime start from 1970 and 1900)
        elapsed_time_min = time.strptime(time.strftime('%H:%M:%S', time.gmtime(min_elapsed_minutes * 60)), "%H:%M:%S")
        elapsed_time_max = time.strptime(time.strftime('%H:%M:%S', time.gmtime(max_elapsed_minutes * 60)), "%H:%M:%S")

        logging.debug("CONSIDERING ONLY SAMPLES BETWEEN \n\t%s\n\t%s ", elapsed_time_min, elapsed_time_max)

        return [sample for sample in samples if
                elapsed_time_min <= time.strptime(sample.elapsed, "%H:%M:%S.%f") <= elapsed_time_max]

    return _f

class IlluminationMap:
    """
            Data structure that represent a map. The map is defined in terms of its axes
        """

    def __init__(self, axes: list, samples: set, drop_outliers=False):
        """
        Note that axes are positional, the first [0] is x, the second[1] is y, the third [2] is z, etc.
        Args:
            axes:
        """
        self.logger = logging.getLogger('illumination_map.IlluminationMapDefinition')

        assert len(axes) > 1, "Cannot build a map with only one feature"

        # Definition of the axes
        self.axes = axes

        # Hide samples that fall ouside the maps as defined by the axes
        self.drop_outliers = drop_outliers

        # Remove duplicated samples: samples with same ID but different attributes (e.g. basepath)
        all_samples = list(deduplicate(samples))
        # Split samples in to valid, invalid, outliers
        self.samples = [sample for sample in all_samples if sample.is_valid]
        # We keep them around no matter what, so we can count them, see them, etc.
        self.invalid_samples = [sample for sample in all_samples if not sample.is_valid]

    def _avg_sparseness_from_map(self, map_data):

        # If there are no samples, we cannot compute it
        if np.count_nonzero(map_data) == 0:
            return np.NaN

        # Iterate over all the non empty cells and compute the distance to all the others non empty cells
        avg_sparseness = 0

        # https://numpy.org/doc/stable/reference/arrays.nditer.html
        # This should iterate over all the elements
        it = np.nditer(map_data, flags=['multi_index'])
        samples = []
        for a in it:
            # print("Considering index ", it.multi_index)
            samples.append(it.multi_index)

        # TODO This can be python-ized
        # Total number of samples

        # k = # observations to compute the mean
        k = 0
        for (sample1, sample2) in combinations(samples, 2):

            if map_data[sample1] == 0 or map_data[sample2] == 0:
                continue

            # Compute distance
            distance = manhattan(sample1, sample2)

            # Increment number of observations
            k += 1

            # Update the avg distance
            # See https://math.stackexchange.com/questions/106700/incremental-averageing

            # print("Considering:", sample1, sample2)
            # print("K", k)
            # print("AVG ", avg_sparseness, end=" ")
            avg_sparseness = avg_sparseness + (distance - avg_sparseness) / k
            # print("AVG ", avg_sparseness)

        return avg_sparseness

    def _avg_max_distance_between_filled_cells_from_map(self, map_data):
        """
        Alternative fomulation for sparseness: for each cell consider only the distance to the farthest cell

        Args:
            map_data:

        Returns:

        """

        # If there are no samples, we cannot compute it
        if np.count_nonzero(map_data) == 0:
            return np.NaN

        # Iterate over all the non empty cells and compute the distance to all the others non empty cells
        avg_sparseness = 0

        # https://numpy.org/doc/stable/reference/arrays.nditer.html
        # This should iterate over all the elements
        it = np.nditer(map_data, flags=['multi_index'])
        samples = []
        for a in it:
            # print("Considering index ", it.multi_index)
            samples.append(it.multi_index)

        # TODO This can be python-ized
        # Total number of samples

        last_sample = None

        max_distances_starting_from_sample = {}

        for (sample1, sample2) in combinations(samples, 2):
            # Combinations do not have repetition, so everytime that sample1 changes, we need to "recompute the
            # max distance". We use sample1 as index to store the max distances starting from a sample

            # Do not consider empty cells
            if map_data[sample1] == 0 or map_data[sample2] == 0:
                continue

            # Compute distance between cells
            distance = manhattan(sample1, sample2)

            if sample1 in max_distances_starting_from_sample.keys():
                max_distances_starting_from_sample[sample1] = max(max_distances_starting_from_sample[sample1], distance)
            else:
                max_distances_starting_from_sample[sample1] = distance

        # Compute the average
        if len(max_distances_starting_from_sample) < 1:
            return 0.0
        else:
            return np.mean([list(max_distances_starting_from_sample.values())])

    # @deprecated("Use np.size instead")
    # def _total_cells(self, features):
    #     return int(np.prod([a.num_cells for a in features]))

    def _total_misbehaviors(self, samples: set) -> int:
        return len([s for s in samples if s.is_misbehavior()])

    def _total_samples(self, samples: set) -> int:
        return len(samples)

    def _mapped_misbehaviors_from_map(self, misbehavior_data):
        """
        Args:
            misbehaviour_data a matrix that contains the count of misbheaviors per cell
        Returns:
            the count of cells for which at least one sample is a misbehavior.

        """
        return np.count_nonzero(misbehavior_data > 0)

    def _filled_cells_from_map(self, coverage_data):
        """
        Returns:
            the count of cells covered by at least one sample.
        """
        # https://note.nkmk.me/en/python-numpy-count/
        return np.count_nonzero(coverage_data > 0)

    def _relative_density_of_mapped_misbehaviors_from_map(self, coverage_data, misbehavior_data):
        """
        Returns:
            the density of misbehaviors in a map computed w.r.t. the amount of filled cells in the map
        """
        filled_cells = self._filled_cells_from_map(coverage_data)
        return self._mapped_misbehaviors_from_map(misbehavior_data) / filled_cells if filled_cells > 0 else np.NaN

    def _density_of_mapped_misbehavior_from_maps(self, misbehavior_data):
        """
        Returns:
            the density of misbehaviors in a map computed
        """
        return self._mapped_misbehaviors_from_map(misbehavior_data) / misbehavior_data.size

    def _density_of_covered_cells_from_map(self, coverage_data):
        """
        Returns:
            the density of covered cell in the map. This is basically the coverage of the map
        """
        return self._filled_cells_from_map(coverage_data) / coverage_data.size

    def _count_collisions(self, map_data):
        """
        Returns:
            the overall count of cells with a collision, i.e., where there's more than one sample
        """
        return np.count_nonzero(map_data > 1)

    def _collisions_ratio(self, map_data):
        """
        Returns:
            the amount of collisions in the map (when two or more samples hit the same cell).
        """
        filled_cells = self._filled_cells_from_map(map_data)
        total_samples = np.sum(map_data)
        return (total_samples - filled_cells) / filled_cells if filled_cells > 0 else np.NaN

    def _get_tool(self, samples):
        # TODO Assume that all the samples belong to the same tool
        # TODO trigger exception otherwise
        if len(samples) > 0:
            for sample in samples:
                return sample.tool
        return None

    def _get_run_id(self, samples):
        # TODO Assume that all the samples belong to the same run_id
        # TODO trigger exception otherwise
        if len(samples) > 0:
            for sample in samples:
                return sample.run
        return None

    def compute_statistics(self, tags=[], feature_selector=None, sample_selector=None):
        """
        Compute the statistics for this map optionally selecting the samples according to the give selector and a
        subset of the features. Otherwise one report for each feature combination will be generated.

        The selector for example, can tell whether or not consider the samples generated in a given interval of
        time.

        Args:
            tags: list of tags, includig the "within X mins"
            feature_selector: a function to select the features
            sample_selector: a function to select the samples. For example, only the samples collected in the first
            15 minutes of run

        Returns:

        """
        filtered_samples = self.samples
        self.logger.debug("Valid samples: %s", len(filtered_samples))

        filtered_invalid_samples = self.invalid_samples
        self.logger.debug("Invalid samples: %s", len(filtered_invalid_samples))

        if sample_selector is not None:
            filtered_samples = sample_selector(self.samples)
            self.logger.debug("Filtered valid samples: %s", len(filtered_samples))

            filtered_invalid_samples = sample_selector(self.invalid_samples)
            self.logger.debug("Filtered invalid samples: %s", len(filtered_invalid_samples))

        filtered_features = self.axes
        if feature_selector is not None:
            filtered_features = feature_selector(self.axes)

        assert len(filtered_features) > 1, "Cannot compute statistics with less than two features"

        report = {}
        # Meta data
        report["Tool"] = self._get_tool(filtered_samples)
        report["Run ID"] = self._get_run_id(filtered_samples)

        report["Tags"] = tags

        # Totals
        report["Total Samples"] = self._total_samples(filtered_samples) + self._total_samples(filtered_invalid_samples)
        report["Valid Samples"] = self._total_samples(filtered_samples)
        report["Invalid Samples"] = self._total_samples(filtered_invalid_samples)
        report["Total Misbehaviors"] = self._total_misbehaviors(filtered_samples)
        report["MisbehaviorPerSample"] = report["Total Misbehaviors"] / report["Total Samples"]

        # Per Feature Statistics
        report["Features"] = {}
        for feature in filtered_features:
            report["Features"][feature.feature_name] = {}
            report["Features"][feature.feature_name]["meta"] = feature.to_dict()
            report["Features"][feature.feature_name]["stats"] = {}

            feature_raw_data = [sample.get_value(feature.feature_name) for sample in filtered_samples]

            report["Features"][feature.feature_name]["stats"]["mean"] = np.NaN
            report["Features"][feature.feature_name]["stats"]["stdev"] = np.NaN
            report["Features"][feature.feature_name]["stats"]["median"] = np.NaN

            if len(feature_raw_data) > 2:
                report["Features"][feature.feature_name]["stats"]["mean"] = statistics.mean(feature_raw_data)
                report["Features"][feature.feature_name]["stats"]["stdev"] = statistics.stdev(feature_raw_data)
                report["Features"][feature.feature_name]["stats"]["median"] = statistics.median(feature_raw_data)
            elif len(feature_raw_data) == 1:
                report["Features"][feature.feature_name]["stats"]["mean"] = feature_raw_data[0]
                report["Features"][feature.feature_name]["stats"]["stdev"] = 0.0
                report["Features"][feature.feature_name]["stats"]["median"] = feature_raw_data[0]

        # Create one report for each pair of selected features
        report["Reports"] = []

        # We filter outliers

        total_samples_in_the_map = filtered_samples

        for feature1, feature2 in itertools.combinations(filtered_features, 2):

            selected_features = [feature1, feature2]

            # make sure we reset this across maps
            filtered_samples = total_samples_in_the_map

            # Remove samples that are outliers for this map
            if self.drop_outliers:
                filtered_samples = drop_outliers_for(feature1, filtered_samples)
                filtered_samples = drop_outliers_for(feature2, filtered_samples)

            # Build the map data: For the moment forget about outer maps, those are mostly for visualization!
            coverage_data, misbehavior_data, _, _ = self._compute_maps_data(feature1, feature2, filtered_samples)

            # Compute statistics over the map data
            map_report = {
                # Meta data
                'Features': [feature.feature_name for feature in selected_features],
                # Counts
                'Sample Count': len(total_samples_in_the_map),
                'Outlier Count': len(total_samples_in_the_map) - len(filtered_samples),
                'Total Cells': coverage_data.size,
                'Filled Cells': self._filled_cells_from_map(coverage_data),
                'Mapped Misbehaviors': self._mapped_misbehaviors_from_map(misbehavior_data),
                # Density
                'Misbehavior Relative Density': self._relative_density_of_mapped_misbehaviors_from_map(coverage_data,
                                                                                                       misbehavior_data),
                'Misbehavior Density': self._density_of_mapped_misbehavior_from_maps(misbehavior_data),
                'Filled Cells Density': self._density_of_covered_cells_from_map(coverage_data),
                'Collisions': self._count_collisions(coverage_data),
                'Misbehavior Collisions': self._count_collisions(misbehavior_data),
                'Collision Ratio': self._collisions_ratio(coverage_data),
                'Misbehavior Collision Ratio': self._collisions_ratio(misbehavior_data),
                # Sparseness
                'Coverage Sparseness': self._avg_max_distance_between_filled_cells_from_map(coverage_data),
                'Misbehavior Sparseness': self._avg_max_distance_between_filled_cells_from_map(misbehavior_data),
                # The follwing two only for retro-compability
                'Avg Sample Distance': self._avg_sparseness_from_map(coverage_data),
                'Avg Misbehavior Distance': self._avg_sparseness_from_map(misbehavior_data)
            }

            report["Reports"].append(map_report)

        return report

    def _compute_maps_data(self, feature1, feature2, samples):
        """
        Create the raw data for the map by placing the samples on the map and counting for each cell how many samples
        are there and how many misbehaviors
        Args:
            feature1:
            feature2:
            samples:

        Returns:
            coverage_map, misbehavior_map
            coverage_outer_map, misbehavior_outer_map
        """
        # TODO Refactor:

        # Reshape the data as ndimensional array. But account for the lower and upper bins.
        coverage_data = np.zeros(shape=(feature1.num_cells, feature2.num_cells), dtype=int)
        misbehaviour_data = np.zeros(shape=(feature1.num_cells, feature2.num_cells), dtype=int)

        coverage_outer_data = np.zeros(shape=(feature1.num_cells + 2, feature2.num_cells + 2), dtype=int)
        misbehaviour_outer_data = np.zeros(shape=(feature1.num_cells + 2, feature2.num_cells + 2), dtype=int)

        for sample in samples:

            # Coordinates reason in terms of bins 1, 2, 3, while data is 0-indexed
            x_coord = feature1.get_coordinate_for(sample, is_outer_map=False) - 1
            y_coord = feature2.get_coordinate_for(sample, is_outer_map=False) - 1

            # Increment the coverage cell
            coverage_data[x_coord, y_coord] += 1

            # Increment the misbehaviour cell
            if sample.is_misbehavior():
                misbehaviour_data[x_coord, y_coord] += 1

            # Outer Maps
            x_coord = feature1.get_coordinate_for(sample, is_outer_map=True) - 1
            y_coord = feature2.get_coordinate_for(sample, is_outer_map=True) - 1

            # Increment the coverage cell
            coverage_outer_data[x_coord, y_coord] += 1

            # Increment the misbehaviour cell
            if sample.is_misbehavior():
                misbehaviour_outer_data[x_coord, y_coord] += 1

        return coverage_data, misbehaviour_data, coverage_outer_data, misbehaviour_outer_data

    def visualize_probability(self, tags=None, feature_selector=None, sample_selector=None):
        """
            Visualize the probability of finding a misbehavior in a give cell, computed as the total of misbehavior over
            the total samples in each cell. This is defined only for cells that have samples in them. Also store
            the probability data so they can be post-processed (e.g., average across run/configuration)
        """
        # Prepare the data by selecting samples and features

        filtered_samples = self.samples
        self.logger.debug("All samples: %s", len(filtered_samples))
        if sample_selector is not None:
            filtered_samples = sample_selector(self.samples)
            self.logger.debug("Filtered samples: %s", len(filtered_samples))

        filtered_features = self.axes
        if feature_selector is not None:
            filtered_features = feature_selector(self.axes)

        figures = []
        # Might be redundant if we store also misbehaviour_maps and coverage_maps
        probability_maps = []
        # To compute confidence intervals and possibly other metrics on the map
        misbehaviour_maps = []
        coverage_maps = []

        total_samples_in_the_map = filtered_samples

        # Create one visualization for each pair of self.axes selected in order
        for feature1, feature2 in itertools.combinations(filtered_features, 2):

            # Make sure we reset this for each feature combination
            filtered_samples = total_samples_in_the_map
            # Remove samples that are outliers for this map
            if self.drop_outliers:
                filtered_samples = drop_outliers_for(feature1, filtered_samples)
                filtered_samples = drop_outliers_for(feature2, filtered_samples)

            coverage_data, misbehaviour_data, _, _ = self._compute_maps_data(feature1, feature2, filtered_samples)

            # figure
            fig, ax = plt.subplots(figsize=(8, 8))

            cmap = sns.cubehelix_palette(dark=0.1, light=0.9, as_cmap=True)
            # Cells have a value between 0.0 and 1.0 since they represent probabilities

            # Set the color for the under the limit to be white (0.0) so empty cells are not visualized
            # cmap.set_under('0.0')
            # Plot NaN in white
            cmap.set_bad(color='white')

            # Coverage data might be zero, so this produces Nan. We convert that to 0.0
            # probability_data = np.nan_to_num(misbehaviour_data / coverage_data)
            raw_probability_data = misbehaviour_data / coverage_data

            # For some weird reason the data in the heatmap are shown with the first dimension on the y and the
            # second on the x. So we transpose
            probability_data = np.transpose(raw_probability_data)

            sns.heatmap(probability_data, vmin=0.0, vmax=1.0, square=True, cmap=cmap)

            xtickslabel = [round(the_bin, 1) for the_bin in feature1.get_bins_labels(is_outer_map=False)]
            ytickslabel = [round(the_bin, 1) for the_bin in feature2.get_bins_labels(is_outer_map=False)]
            #
            ax.set_xticklabels(xtickslabel)
            plt.xticks(rotation=45)
            ax.set_yticklabels(ytickslabel)
            plt.yticks(rotation=0)

            tool_name = str(self._get_tool(filtered_samples))
            run_id = str(self._get_run_id(filtered_samples)).zfill(3)

            title_tokens = ["Mishbehavior Probability", "\n"]
            title_tokens.extend(["Tool:", tool_name, "--", "Run ID:", run_id])

            if tags is not None and len(tags) > 0:
                title_tokens.extend(["\n", "Tags:"])
                title_tokens.extend([str(t) for t in tags])

            the_title = " ".join(title_tokens)

            fig.suptitle(the_title, fontsize=16)

            # Plot small values of y below.
            # We need this to have the y axis start from zero at the bottom
            ax.invert_yaxis()

            # axis labels
            plt.xlabel(feature1.feature_name)
            plt.ylabel(feature2.feature_name)

            # Include data to store the file with same prefix

            # Add the store_to attribute to the figure and maps object
            setattr(fig, "store_to",
                    "-".join(["probability", tool_name, run_id, feature1.feature_name, feature2.feature_name]))
            figures.append(fig)

            probability_maps.append({
                "data": raw_probability_data,
                "store_to": "-".join(["probability", tool_name, run_id, feature1.feature_name, feature2.feature_name])
            })

            misbehaviour_maps.append({
                "data": misbehaviour_data,
                "store_to": "-".join(["misbehaviour", tool_name, run_id, feature1.feature_name, feature2.feature_name])
            })

            coverage_maps.append({
                "data": coverage_data,
                "store_to": "-".join(["coverage", tool_name, run_id, feature1.feature_name, feature2.feature_name])
            })

        return figures, probability_maps, misbehaviour_maps, coverage_maps

    def visualize(self, tags=None, feature_selector=None, sample_selector=None):
        """
            Visualize the samples and the features on a map. The map cells contains the number of samples for each
            cell, so empty cells (0) are white, cells with few elements have a light color, while cells with more
            elements have darker color. This gives an intuition on the distribution of the misbheaviors and the
            collisions

        Args:
            tags: List of tags to the title so we can easily identify the maps
            feature_selector:
            sample_selector:

        Returns:


        """

        filtered_samples = self.samples
        self.logger.debug("All samples: %s", len(filtered_samples))
        if sample_selector is not None:
            filtered_samples = sample_selector(self.samples)
            self.logger.debug("Filtered samples: %s", len(filtered_samples))

        filtered_features = self.axes
        if feature_selector is not None:
            filtered_features = feature_selector(self.axes)

        figures = []

        total_samples_in_the_map = filtered_samples

        # Create one visualization for each pair of self.axes selected in order
        for feature1, feature2 in itertools.combinations(filtered_features, 2):

            # Make sure we reset this for each feature combination
            filtered_samples = total_samples_in_the_map
            # Remove samples that are outliers for this map
            if self.drop_outliers:
                filtered_samples = drop_outliers_for(feature1, filtered_samples)
                filtered_samples = drop_outliers_for(feature2, filtered_samples)

            # TODO For the moment, since filtered_Samples might be different we need to rebuild this every time
            coverage_data, misbehaviour_data, _, _ = self._compute_maps_data(feature1, feature2, filtered_samples)

            # figure
            fig, ax = plt.subplots(figsize=(8, 8))

            cmap = sns.cubehelix_palette(dark=0.5, light=0.9, as_cmap=True)
            # Set the color for the under the limit to be white (so they are not visualized)
            cmap.set_under('1.0')

            # For some weird reason the data in the heatmap are shown with the first dimension on the y and the
            # second on the x. So we transpose
            coverage_data = np.transpose(coverage_data)

            sns.heatmap(coverage_data, vmin=1, vmax=20, square=True, cmap=cmap)

            # Plot misbehaviors - Iterate over all the elements of the array to get their coordinates:
            it = np.nditer(misbehaviour_data, flags=['multi_index'])
            for v in it:
                # Plot only misbehaviors
                if v > 0:
                    alpha = 0.1 * v if v <= 10 else 1.0
                    (x, y) = it.multi_index
                    # Plot as scattered plot. the +0.5 ensures that the marker in centered in the cell
                    plt.scatter(x + 0.5, y + 0.5, color="black", alpha=alpha, s=50)

            xtickslabel = [round(the_bin, 1) for the_bin in feature1.get_bins_labels(is_outer_map=False)]
            ytickslabel = [round(the_bin, 1) for the_bin in feature2.get_bins_labels(is_outer_map=False)]
            #
            ax.set_xticklabels(xtickslabel)
            plt.xticks(rotation=45)
            ax.set_yticklabels(ytickslabel)
            plt.yticks(rotation=0)

            tool_name = str(self._get_tool(filtered_samples))
            run_id = str(self._get_run_id(filtered_samples)).zfill(3)

            title_tokens = ["Collisions and Mishbehaviors", "\n"]
            title_tokens.extend(["Tool:", tool_name, "--", "Run ID:", run_id])

            if tags is not None and len(tags) > 0:
                title_tokens.extend(["\n", "Tags:"])
                title_tokens.extend([str(t) for t in tags])

            the_title = " ".join(title_tokens)

            fig.suptitle(the_title, fontsize=16)

            # Plot small values of y below.
            # We need this to have the y axis start from zero at the bottom
            ax.invert_yaxis()

            # axis labels
            plt.xlabel(feature1.feature_name)
            plt.ylabel(feature2.feature_name)

            # Add the store_to attribute to the figure object
            store_to = "-".join(
                ["collision", "misbehavior", tool_name, run_id, feature1.feature_name, feature2.feature_name])
            setattr(fig, "store_to", store_to)

            figures.append(fig)

        return figures

def _store_figures_to_folder(figures, tags, run_folder):
    # figures, report
    # "Tool": "DLFuzz",
    #     "Run ID": "1",
    #     "Total Samples": 1098,
    file_format = 'pdf'

    for figure in figures:
        file_name_tokens = [figure.store_to]

        # Add tags if any
        if tags is not None:
            file_name_tokens.extend([str(t) for t in tags])

        # Add File extension
        figure_file_name = "-".join(file_name_tokens) + "." + file_format

        figure_file = os.path.join(run_folder, figure_file_name)
        log.debug("Storing figure to file %s ", figure_file)
        figure.savefig(figure_file, format=file_format)

# creates a folder in the system
def create_folder(folder_name, path):
    _fold_name = folder_name.replace(':', ' ').replace('.', ' ')
    _path = path
    if not (os.path.exists(_path + "/" + _fold_name)):
        os.chdir(_path)
        os.mkdir(_fold_name)
    else:
        print("the folder with the name '" + folder_name + "' is already exist in the system")

def _store_maps_to_folder(maps, tags, run_folder):
    file_format = 'npy'
    # np.save('test3.npy', a)
    for the_map in maps:
        file_name_tokens = [the_map["store_to"]]

        # Add tags if any
        if tags is not None:
            file_name_tokens.extend([str(t) for t in tags])

        # Add File extension
        map_file_name = "-".join(file_name_tokens) + "." + file_format

        map_file = os.path.join(run_folder, map_file_name)
        log.debug("Storing map %s to file %s ", id, map_file)
        # Finally store it in  platform-independent numpy format
        np.save(map_file, the_map["data"])


def _store_report_to_folder(report, tags, run_folder):
    """
    Store the content of the report dict as json in the given run_folder as stats.json

    Args:
        report:
        run_folder:

    Returns:

    """
    # Basic format
    file_name_tokens = [str(report["Tool"]), str(report["Run ID"]).zfill(3)]

    # Add tags if any
    # if tags is not None:
    #     file_name_tokens.extend([str(t) for t in tags])

    # This is the actual file name
    file_name_tokens.append("stats")

    # Add File extension
    report_file_name = "-".join(file_name_tokens) + "." + "json"

    report_file = os.path.join(run_folder, report_file_name)

    log.debug("Storing report %s to file %s ", id, report_file)
    with open(report_file, 'w') as output_file:
        output_file.writelines(json.dumps(report, indent=4))

def create_feature_maps(ctx, visualize, drop_outliers, tag, run_folder):
    # creates map axes
    map_axis = create_illumination_map_axis()

    # creates samples
    samples = create_samples()

    # set_of_samples = set(samples)
    # print(set_of_samples)

    # creates the ilumination map
    the_map = IlluminationMap(map_axis, set(samples), drop_outliers=drop_outliers)

    # creates tags
    tags = []
    for t in tag:
        # TODO Leave Tags Case Sensitive
        t = str(t)
        if t not in tags:
            if ctx.obj['show-progress']:
                print("Tagging using %s" % t)
            tags.append(t)

    # # creates at_minutes
    at_minutes = []
    # for e in at:
    #     if e not in at_minutes:
    #         if ctx.obj['show-progress']:
    #             print("Generating Maps for samples at %s minutes" % e)
    #         at_minutes.append(e)

    # Note the None identify the "use all the samples" setting. By default is always the last one
    at_minutes.append(None)

    report = the_map.compute_statistics(tags=[], sample_selector=None)

    # TODO what is at_minutes?
    for e in at_minutes:

        select_samples = None
        _tags = tags[:]
        if e is not None:
            if ctx.obj['show-progress']:
                print("Selecting samples within ", e, "minutes")
            # Create the sample selector
            select_samples = select_samples_by_elapsed_time(e)

            # Add the minutes ot the tag, create a copy so we do not break the given set of tags
            _tags.append("".join([str(e).zfill(2), "min"]))
        else:
            if ctx.obj['show-progress']:
                print("Selecting all the samples")

        report = the_map.compute_statistics(tags =_tags, sample_selector = select_samples)

        # Show this if debug is enabled
        log.debug(json.dumps(report, indent=4))

        # Store the report as json
        _store_report_to_folder(report, _tags, run_folder)

        # Create coverage and probability figures
        figures = the_map.visualize(tags=_tags, sample_selector=select_samples)
        # store figures without showing them
        _store_figures_to_folder(figures, _tags, run_folder)

        figures, probability_maps, misbehaviour_maps, coverage_maps\
            = the_map.visualize_probability(tags=_tags, sample_selector=select_samples)
        # store the outputs
        _store_figures_to_folder(figures, _tags, run_folder)
        # store maps
        ##_store_maps_to_folder(probability_maps, _tags, run_folder)
        ##_store_maps_to_folder(misbehaviour_maps, _tags, run_folder)
        _store_maps_to_folder(coverage_maps, _tags, run_folder)

        # Visualize Everything at the end
    if visualize:
        plt.show()
    else:
        # Close all the figures if open
        for figure in figures:
            plt.close(figure)


# just to test
def create_list_of_paths():
    path_to_results = "../tool-competition-av/results/sample_test_generators.*/test.*.json"

    # returns paths to test.0001.json files
    list_of_paths_to_json_files = glob.glob(path_to_results)
    return list_of_paths_to_json_files


def create_illumination_map_axis():
    '''
    creates axis for the map
    '''

    # creates array of features names
    features = []
    features.append('MinRadius')
    features.append('DirectionCoverage')
    features.append('MeanLateralPosition')
    features.append('SegmentCount')
    features.append('SDSteeringAngle')
    features.append('Curvature')

    # creates matrix of features, max, min, int = 5
    array = create_array_of_features_for_axis(features)

    # creates the axis
    map_features = []
    for f in array:
        # if ctx.obj['show-progress']:
        #     print("Using feature %s" % f[0])
        map_features.append(IlluminationAxisDefinition(f[0], f[1], f[2], f[3]))
    return map_features


def create_samples():
    '''
    creates BeamNGSamples to use by creating the IlluminationMap
    '''
    samples = []

    list_of_dictionaries = create_array_of_computed_features_dictionaries_of_all_tests()
    basepath = ""

    for dictionary in list_of_dictionaries:
        samples.append(BeamNGSample(basepath, dictionary['TestData']['id'],
                                    dictionary['TestData']['is_valid'],
                                    dictionary['InterpolatedPoints'],
                                    dictionary['States'],
                                    dictionary['TestData'],
                                    dictionary['MinRadius'],
                                    dictionary['DirectionCoverage'],
                                    dictionary['MeanLateralPosition'],
                                    dictionary['SegmentCount'],
                                    dictionary['SDSteeringAngle'],
                                    dictionary['Curvature'],
                                    dictionary['FitnessFunction']))

    return samples


def post_process(ctx, result_folder, the_executor):
    """
        This method is invoked once the test generation is over.
    """
    # Plot the stats on the console
    log.info("Test Generation Statistics:")
    log.info(the_executor.get_stats())

    # Generate the actual summary files
    create_experiment_description(result_folder, ctx.params)

    # Generate the other reports
    create_summary(result_folder, the_executor.get_stats())

    # class Ctx:
    #     obj = {"show-progress": True}
    #
    #
    # ctx = Ctx()
    #
    # # Create the feature map ctx, visualize, drop_outliers, tag, at, run_folder
    # create_feature_maps(ctx, True, True, tag, result_folder + "/map")


def create_post_processing_hook(ctx, result_folder, executor):
    """
        Uses HighOrder functions to setup the post processing hooks that will be trigger ONLY AND ONLY IF the
        test generation has been killed by us, i.e., this will not trigger if the user presses Ctrl-C

    :param result_folder:
    :param executor:
    :return:
    """

    def _f():
        if executor.is_force_timeout():
            # The process killed itself because a timeout, so we need to ensure the post_process function
            # is called
            post_process(ctx, result_folder, executor)

    return _f


def setup_logging(log_to, debug):
    def log_exception(extype, value, trace):
        log.exception('Uncaught exception:', exc_info=(extype, value, trace))

    # Disable annoyng messages from matplot lib.
    # See: https://stackoverflow.com/questions/56618739/matplotlib-throws-warning-message-because-of-findfont-python
    log.getLogger('matplotlib.font_manager').disabled = True

    term_handler = log.StreamHandler()
    log_handlers = [term_handler]
    start_msg = "Started test generation"

    if log_to is not None:
        file_handler = log.FileHandler(log_to, 'a', 'utf-8')
        log_handlers.append(file_handler)
        start_msg += " ".join(["writing to file: ", str(log_to)])

    log_level = log.DEBUG if debug else log.INFO

    log.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=log_level, handlers=log_handlers)

    sys.excepthook = log_exception

    log.info(start_msg)


@click.command()
@click.option('--executor', type=click.Choice(['mock', 'beamng'], case_sensitive=False), default="mock",
              show_default='Mock Executor (meant for debugging)',
              help="The name of the executor to use. Currently we have 'mock' or 'beamng'.")
@click.option('--beamng-home', required=False, default=None, type=click.Path(exists=True),
              show_default='None',
              help="Customize BeamNG executor by specifying the home of the simulator.")
@click.option('--beamng-user', required=False, default=None, type=click.Path(exists=True),
              show_default='Currently Active User (~/BeamNG.research/)',
              help="Customize BeamNG executor by specifying the location of the folder "
                   "where levels, props, and other BeamNG-related data will be copied."
                   "** Use this to avoid spaces in URL/PATHS! **")
@click.option('--time-budget', required=True, type=int, callback=validate_time_budget,
              help="Overall budget for the generation and execution. Expressed in 'real-time'"
                   "seconds.")
@click.option('--map-size', type=int, default=200, callback=validate_map_size,
              show_default='200m, which leads to a 200x200m^2 squared map',
              help="The lenght of the size of the squared map where the road must fit."
                   "Expressed in meters.")
@click.option('--oob-tolerance', type=float, default=0.95, callback=validate_oob_tolerance,
              show_default='0.95',
              help="The tolerance value that defines how much of the vehicle should be outside the lane to "
                   "trigger a failed test. Must be a value between 0.0 (all oob) and 1.0 (no oob)")
@click.option('--speed-limit', type=int, default=70, callback=validate_speed_limit,
              show_default='70 Km/h',
              help="The max speed of the ego-vehicle"
                   "Expressed in Kilometers per hours")
@click.option('--module-name', required=True, type=str,
              help="Name of the module where your test generator is located.")
@click.option('--module-path', required=False, type=click.Path(exists=True),
              help="Path of the module where your test generator is located.")
@click.option('--class-name', required=True, type=str,
              help="Name of the (main) class implementing your test generator.")
# Visual Debugging
@click.option('--visualize-tests', required=False, is_flag=True, default=False,
              show_default='Disabled',
              help="Visualize the last generated test, i.e., the test sent for the execution. "
                   "Invalid tests are also visualized.")
# Logging options
@click.option('--log-to', required=False, type=click.Path(exists=False),
              help="Location of the log file. If not specified logs appear on the console")
@click.option('--debug', required=False, is_flag=True, default=False,
              show_default='Disabled',
              help="Activate debugging (results in more logging)")
@click.pass_context
def generate(ctx, executor, beamng_home, beamng_user,
             time_budget, map_size, oob_tolerance, speed_limit,
             module_name, module_path, class_name,
             visualize_tests, log_to, debug):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)

    # TODO Refactor by adding a create summary command and forwarding the output of this run to that command

    # Setup logging
    setup_logging(log_to, debug)

    # Setup test generator by dynamically loading it
    module = importlib.import_module(module_name, module_path)
    the_class = getattr(module, class_name)

    road_visualizer = None
    # Setup visualization
    if visualize_tests:
        road_visualizer = RoadTestVisualizer(map_size=map_size)

    # Setup folder structure by ensuring that the basic folder structure is there.
    default_output_folder = os.path.join(get_script_path(), OUTPUT_RESULTS_TO)
    try:
        os.makedirs(default_output_folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Create the unique folder that will host the results of this execution using the test generator data and
    # a timestamp as id
    # TODO Allow to specify a location for this folder and the run id
    timestamp_id = time.time_ns() // 1000000
    result_folder = os.path.join(default_output_folder,
                                 "_".join([str(module_name), str(class_name), str(timestamp_id)]))

    try:
        os.makedirs(result_folder)
    except OSError:
        log.fatal("An error occurred during test generation")
        traceback.print_exc()
        sys.exit(2)

    log.info("Outputting results to " + result_folder)

    # Setup executor. All the executor must output the execution data into the result_folder
    if executor == "mock":
        from code_pipeline.executors import MockExecutor
        the_executor = MockExecutor(result_folder, time_budget, map_size,
                                    road_visualizer=road_visualizer)
    elif executor == "beamng":
        from code_pipeline.beamng_executor import BeamngExecutor
        the_executor = BeamngExecutor(result_folder, time_budget, map_size,
                                      oob_tolerance=oob_tolerance, max_speed=speed_limit,
                                      beamng_home=beamng_home, beamng_user=beamng_user,
                                      road_visualizer=road_visualizer)

    # Register the shutdown hook for post processing results
    register_exit_fun(create_post_processing_hook(ctx, result_folder, the_executor))

    try:
        # Instantiate the test generator
        test_generator = the_class(time_budget=time_budget, executor=the_executor, map_size=map_size)
        # Start the generation
        test_generator.start()
    except Exception:
        log.fatal("An error occurred during test generation")
        traceback.print_exc()
        sys.exit(2)
    finally:
        # Ensure the executor is stopped no matter what.
        # TODO Consider using a ContextManager: With executor ... do
        the_executor.close()

    # We still need this here to post process the results if the execution takes the regular flow
    post_process(ctx, result_folder, the_executor)


class Sample:

    def __init__(self):
        self.id = None
        self.tool = None
        self.misbehaviour = False
        self.run = None
        self.timestamp = None
        self.elapsed = None
        self.features = {}
        self.is_valid = True
        self.valid_according_to = None

    # TODO Maybe make this an abstract method?
    def is_misbehavior(self):
        return self.misbehaviour

    def get_value(self, feature_name):
        if feature_name in self.features.keys():
            return self.features[feature_name]
        else:
            return None

    @staticmethod
    def from_dict(the_dict):
        sample = Sample()
        for k in sample.__dict__.keys():
            setattr(sample, k, None if k not in the_dict.keys() else the_dict[k])
        return sample


class BeamNGSample(Sample):
    # At which radius we interpret a tuns as a straight?
    # MAX_MIN_RADIUS = 200
    MAX_MIN_RADIUS = 170

    def __init__(self, basepath,
                 id,
                 is_valid,
                 int_points,
                 states,
                 test_data,
                 minRadius,
                 directionCoverage,
                 meanLateralPosition,
                 segmentCount,
                 sdSteeringAngle,
                 curvature,
                 fitnessFunction):
        super(BeamNGSample, self).__init__()
        self.basepath = basepath
        self.id = id,
        self.is_valid = is_valid,
        self.interpolated_points = int_points
        self.simulation_states = states
        self.test_data = test_data
        self.features["MinRadius"] = minRadius
        self.features["DirectionCoverage"] = directionCoverage
        self.features["MeanLateralPosition"] = meanLateralPosition
        self.features["SegmentCount"] = segmentCount
        self.features["SDSteeringAngle"] = sdSteeringAngle
        self.features["Curvature"] = curvature
        self.features["FitnessFunction"] = fitnessFunction

    # def visualize_misbehaviour(self):
    #     # THIS IS THE CODE FOR OOB
    #     # Create the road geometry from the nodes. At this point nodes have been reversed alredy if needed.
    #     road_geometry = metrics.get_geometry(self.road_nodes)
    #
    #     road_left_edge_x = np.array([e['left'][0] for e in road_geometry])
    #     road_left_edge_y = np.array([e['left'][1] for e in road_geometry])
    #
    #     left_edge_x = np.array([e['middle'][0] for e in road_geometry])
    #     left_edge_y = np.array([e['middle'][1] for e in road_geometry])
    #     right_edge_x = np.array([e['right'][0] for e in road_geometry])
    #     right_edge_y = np.array([e['right'][1] for e in road_geometry])
    #
    #     # Create the road polygon from the geometry
    #
    #     right_edge_road = LineString(zip(right_edge_x[::-1], right_edge_y[::-1]))
    #     left_edge_road = LineString(zip(road_left_edge_x, road_left_edge_y))
    #
    #     l_edge_road = left_edge_road.coords
    #     r_edge_road = right_edge_road.coords
    #
    #     road_polygon = Polygon(list(l_edge_road) + list(r_edge_road))
    #
    #     # Plot the road
    #     plt.gca().add_patch(PolygonPatch(road_polygon, fc='gray', alpha=0.5, zorder=2 ))
    #
    #
    #     # Create the right lane polygon from the geometry
    #     # Note that one must be in reverse order for the polygon to close correctly
    #     right_edge = LineString(zip(right_edge_x[::-1], right_edge_y[::-1]))
    #     left_edge = LineString(zip(left_edge_x, left_edge_y))
    #
    #     l_edge = left_edge.coords
    #     r_edge = right_edge.coords
    #
    #     right_lane_polygon = Polygon(list(l_edge) + list(r_edge))
    #
    #     # TODO Plot road as well to understand if this is exactly the side we thing it is
    #     plt.plot(*right_lane_polygon.exterior.xy, color='gray')
    #
    #     # Plot all the observations in trasparent green except the OOB
    #     for position in [Point(sample["pos"][0], sample["pos"][1]) for sample in self.simulation_states]:
    #         if right_lane_polygon.contains(position):
    #             plt.plot(position.x, position.y, 'o', color='green', alpha=0.2)
    #         else:
    #             plt.plot(position.x, position.y, 'o', color='red', alpha=1.0)
    #
    #     plt.gca().set_aspect('equal')
    #
    #
    # def _resampling(self, sample_nodes, dist=1.5):
    #     new_sample_nodes = []
    #     dists = []
    #     for i in range(1, len(sample_nodes)):
    #         x0 = sample_nodes[i - 1][0]
    #         x1 = sample_nodes[i][0]
    #         y0 = sample_nodes[i - 1][1]
    #         y1 = sample_nodes[i][1]
    #
    #         d = sqrt(pow((x1 - x0), 2) + pow((y1 - y0), 2))
    #         dists.append(d)
    #         if d >= dist:
    #             dt = dist
    #             new_sample_nodes.append([x0, y0, -28.0, 8.0])
    #             while dt <= d - dist:
    #                 t = dt / d
    #                 xt = ((1 - t) * x0 + t * x1)
    #                 yt = ((1 - t) * y0 + t * y1)
    #                 new_sample_nodes.append([xt, yt, -28.0, 8.0])
    #                 dt = dt + dist
    #             new_sample_nodes.append([x1, y1, -28.0, 8.0])
    #         else:
    #             new_sample_nodes.append([x0, y0, -28.0, 8.0])
    #             new_sample_nodes.append([x1, y1, -28.0, 8.0])
    #
    #     points_x = []
    #     points_y = []
    #     final_nodes = list()
    #     # discard the Repetitive points
    #     for i in range(1, len(new_sample_nodes)):
    #         if new_sample_nodes[i] != new_sample_nodes[i - 1]:
    #             final_nodes.append(new_sample_nodes[i])
    #             points_x.append(new_sample_nodes[i][0])
    #             points_y.append(new_sample_nodes[i][1])
    #     return final_nodes
    #
    # def compute_input_metrics(self, resampled_road_nodes):
    #     # Input features
    #     self.features["min_radius"] = metrics.capped_min_radius(self.MAX_MIN_RADIUS, resampled_road_nodes)
    #     self.features["segment_count"] = metrics.segment_count(resampled_road_nodes)
    #     self.features["direction_coverage"] = metrics.direction_coverage(resampled_road_nodes)
    #
    # def compute_output_metrics(self, simulation_states):
    #     # Output features
    #     self.features["sd_steering"] = metrics.sd_steering(simulation_states)
    #
    #     #self.features["mean_lateral_position"] = metrics.mean_absolute_lateral_position(simulation_states)
    #     road_geometry = metrics.get_geometry(self.road_nodes)
    #
    #     middle = [e['middle'] for e in road_geometry]
    #     right = [e['right'] for e in road_geometry]
    #     middle_points = [(p[0], p[1]) for p in middle]
    #     right_points = [(p[0], p[1]) for p in right]
    #
    #     right_polyline = compute_right_polyline(middle_points, right_points)
    #
    #     # road_spine = LineString(middle_points)
    #
    #     # road_polygon = _polygon_from_geometry(road_geometry)
    #     #
    #     # # Plot road
    #     # plt.plot(*road_polygon.exterior.xy)
    #     # # Plot centeral spine
    #     # plt.plot(*road_spine.xy, "r-")
    #     #
    #     # # LineString
    #
    #     # plt.plot(*right_polyline.xy)
    #     # positions = [ (state["pos"][0], state["pos"][1]) for state in simulation_states]
    #     #
    #     # for s in positions:
    #     #     plt.plot(s[0], s[1], 'ob')
    #     #     pass
    #     #for state in segment.simulation_states:
    #     #    dist = oob_distance(state["pos"], right_poly)
    #     #    dist2 = state["oob_distance"]
    #     #    assert (dist == dist2)
    #     self.features["mean_lateral_position"] = metrics.mean_lateral_position(simulation_states, right_polyline)

    def to_dict(self):
        """
            This is common for all the BeamNG samples
        """

        return {'id': self.id,
                'is_valid': self.is_valid,
                'valid_according_to': self.valid_according_to,
                'misbehaviour': self.is_misbehavior(),
                'elapsed': self.elapsed,
                'timestamp': self.timestamp,
                'interpolated_points': self.interpolated_points,
                'simulation_states': self.simulation_states,
                'test_data': self.test_data,
                'MinRadius': self.get_value("MinRadius"),
                'SegmentCount': self.get_value("SegmentCount"),
                'DirectionCoverage': self.get_value("DirectionCoverage"),
                'SDSteeringAngle': self.get_value("SDSteeringAngle"),
                'MeanLateralPosition': self.get_value("MeanLateralPosition"),
                'Curvature': self.get_value("Curvature"),
                'FitnessFunction': self.get_value("FitnessFunction"),
                'tool': self.tool,
                'run': self.run,
                'features': self.features}

    def dump(self):
        data = self.to_dict()
        filedest = os.path.join(os.path.dirname(self.basepath), "info_" + str(self.id) + ".json")
        with open(filedest, 'w') as f:
            (json.dump(data, f, sort_keys=True, indent=4))


class IlluminationAxisDefinition:
    """
        @This class was copied from DeepHyperion tool

        Data structure that model one axis of the map. In general a map can have multiple axes, even if we visualize
        only a subset of them. On axis usually correspond to a feature to explore.

        For the moment we assume that each axis is equally split in `num_cells`
    """

    def __init__(self, feature_name, min_value, max_value, num_cells):
        self.logger = logging.getLogger('illumination_map.IlluminationAxisDefinition')
        self.logger.debug('Creating an instance of IlluminationAxisDefinition for feature %s', feature_name)

        self.feature_name = feature_name
        self.min_value = min_value
        self.max_value = max_value
        self.num_cells = num_cells
        # Definition of the inner map, values might fall outside it if less than min
        self.original_bins = np.linspace(min_value, max_value, num_cells)
        # Definition of the outer map
        # Include the default boundary conditions. Note that we do not add np.PINF, but the max value.
        # Check: https://stackoverflow.com/questions/4355132/numpy-digitize-returns-values-out-of-range
        self.bins = np.concatenate(([np.NINF], self.original_bins, [max_value + 0.001]))

    def get_bins_labels(self, is_outer_map=False):
        """
        Note that here we return explicitly the last bin
        Returns: All the bins plus the default

        """
        if is_outer_map:
            return np.concatenate(([np.NINF], self.original_bins, [np.PINF]))
        else:
            return self.original_bins

    def get_coordinate_for(self, sample: Sample, is_outer_map=False):
        """
        Return the coordinate of this sample according to the definition of this axis. It triggers exception if the
            sample does not declare a field with the name of this axis, i.e., the sample lacks this feature

        Args:
            sample:

        Returns:
            an integer representing the coordinate of the sample in this dimension

        Raises:
            an exception is raised if the sample does not contain the feature
        """

        # TODO Check whether the sample has the feature
        value = sample.get_value(self.feature_name)

        if value < self.min_value:
            self.logger.warning("Sample %s has value %s below the min value %s for feature %s",
                                sample.id, value, self.min_value, self.feature_name)
        elif value > self.max_value:
            self.logger.warning("Sample %s has value %s above the max value %s for feature %s",
                                sample.id, value, self.max_value, self.feature_name)

        if is_outer_map:
            return np.digitize(value, self.bins, right=False)
        else:
            return np.digitize(value, self.original_bins, right=False)

    def is_outlier(self, sample):
        value = sample.get_value(self.feature_name)
        is_outlier = value < self.min_value or value > self.max_value #
        return is_outlier #
        # return value < self.min_value or value > self.max_value

    def to_dict(self):
        the_dict = {
            "name": self.feature_name,
            "min-value": self.min_value,
            "max-value": self.max_value,
            "num-cells": self.num_cells
        }
        return the_dict


if __name__ == '__main__':
    # when current file run as executable
    # result = load_the_interpolated_points_from_json_file("../tool-competition-av/results/sample_test_generators.one_test_generator_OneTestGenerator_1619774479881/test.0001.json")
    # print(result)
    # result2 = load_the_simulation_records_data_from_json("../tool-competition-av/results/sample_test_generators.one_test_generator_OneTestGenerator_1619774479881/test.0001.json")
    # print(result2)
    # list = create_list_of_paths()
    # test_data = load_test_data_from_json_file(list[0])
    # print(test_data['is_valid'])
    # print(list)
    ##interpolated_points = load_the_interpolated_points_from_json_file(list[1])  # ['interpolated_points']
    # print(interpolated_points)

    # states = load_the_simulation_records_data_from_json(list[1])
    # print(states)
    # print("Steering of the first State: " + str(states[1].steering))
    # print("Oob_distance of the first State: " + str(states[1].oob_distance))
    # #compute_features_for_the_simulation("")

    # min_Radius_Value = min_radius(interpolated_points, )
    # direction_Coverage = direction_coverage(interpolated_points, )
    # mean_Lateral_Position = mean_lateral_position(states)
    # segment_count_value = segment_count(interpolated_points)
    # sds_steering_angle_value = sd_steering(states)
    # curvature_value = curvature(interpolated_points, )
    # fitness_function_value = fitness_function(test_data)
    # print(direction_Coverage)

    # dictionary_of_features = compute_features_for_the_simulation(list[1])
    # print(dictionary_of_features)
    # array_of_tests_dict_with_features = create_array_of_computed_features_dictionaries_of_all_tests()
    # print(array_of_tests_dict_with_features)

    max_MinRadius = find_max_value_of_feature('MinRadius')
    min_MinRadius = find_min_value_of_feature('MinRadius')
    print(max_MinRadius)
    print(min_MinRadius)
    features = []
    features.append('MinRadius')
    features.append('DirectionCoverage')
    features.append('MeanLateralPosition')
    features.append('SegmentCount')
    features.append('SDSteeringAngle')
    features.append('Curvature')

    array = create_array_of_features_for_axis(features)
    # print(array)
    #
    samples = create_samples()
    sample1 = samples[1].to_dict()
    # print(samples[1].to_dict())
    #
    #
    class Ctx:
        obj = {"show-progress": True}


    ctx = Ctx()

    tag = []
    tag.append("test")
    # tag.append(str(datetime.datetime.now().time()))
    # print(tag)


    create_folder("map", "../tool-competition-av/results/")
    # if not os.path.exists(os.path.dirname("../tool-competition-av/results/map/")):
    #     dir_name = os.path.dirname(filename)
    #     os.makedirs(dir_name)


    # Create the feature map ctx, visualize, drop_outliers, tag, at, run_folder
    create_feature_maps(ctx, True, True, tag, "../tool-competition-av/results/map")
    print(samples[1].to_dict())

    # generate()
