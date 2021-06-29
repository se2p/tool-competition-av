#
#
#
#
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

import numpy as np
from itertools import tee
from typing import List, Tuple

from code_pipeline.visualization import RoadTestVisualizer
from code_pipeline.tests_generation import TestGenerationStatistic
from code_pipeline.test_generation_utils import register_exit_fun

from code_pipeline.tests_evaluation import OOBAnalyzer

AngleLength = Tuple[float, float]
ListOfAngleLength = List[AngleLength]

Point = Tuple[float, float]
ListOfPoints = List[Point]
# TODO Make this configurable?
from self_driving.simulation_data import SimulationDataRecord

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
        raise click.UsageError('The provided value for ' + str(param) + ' is invalid. Choose an integer between 100 and 1000')
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
            output_file.write( csv_content)
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


def min_radius(nodes, w=5):  # TODO x.m.sample_nodes are interpolated_points
    '''
    calculates the value of 'min_radius' feature dimension
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
    interpolated_points = load_the_interpolated_points_from_json_file(folder)
    states = load_the_simulation_records_data_from_json(folder)
    test_data = load_test_data_from_json_file(folder)

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

    return feature_values_dict


def create_array_of_computed_features_dictionaries_of_all_tests():
    '''
    creates a list of Dictionaries, that contains fitness function and all features
    of all 'test.0001.json' files in the 'results' folder
    '''
    list_of_dictionaries_of_computed_features = []
    # TODO to change the path to the files
    path_to_results = "../tool-competition-av/tests/results/sample_test_generators.*/test.0001.json"

    # returns paths to test.0001.json files
    list_of_paths_to_json_files = glob.glob(path_to_results)
    for path in list_of_paths_to_json_files:
        list_of_dictionaries_of_computed_features.append(compute_features_for_the_simulation(path))

    return list_of_dictionaries_of_computed_features


def create_feature_maps(folder, params):
    pass


# just to test
def create_list_of_paths():
    path_to_results = "../tool-competition-av/tests/results/sample_test_generators.*/test.0001.json"

    # returns paths to test.0001.json files
    list_of_paths_to_json_files = glob.glob(path_to_results)
    return list_of_paths_to_json_files


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

    # Create the feature map
    create_feature_maps(result_folder, ctx.params)





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
        log_handlers.append( file_handler )
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
    result_folder = os.path.join(default_output_folder, "_".join([str(module_name), str(class_name), str(timestamp_id)]))

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


if __name__ == '__main__':
    # when current file run as executable
    # result = load_the_interpolated_points_from_json_file("../tool-competition-av/results/sample_test_generators.one_test_generator_OneTestGenerator_1619774479881/test.0001.json")
    # print(result)
    # result2 = load_the_simulation_records_data_from_json("../tool-competition-av/results/sample_test_generators.one_test_generator_OneTestGenerator_1619774479881/test.0001.json")
    # print(result2)
    list = create_list_of_paths()
    print(list)
    ##interpolated_points = load_the_interpolated_points_from_json_file(list[1])  # ['interpolated_points']
    # print(interpolated_points)
    ##test_data = load_test_data_from_json_file(list[1])
    ##states = load_the_simulation_records_data_from_json(list[1])
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
    ##array_of_tests_dict_with_features = create_array_of_computed_features_dictionaries_of_all_tests()
    ##print(array_of_tests_dict_with_features)
    # generate()
