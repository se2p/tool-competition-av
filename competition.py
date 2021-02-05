#
#
#
#
import click
import importlib
import traceback
import time
import os
import sys
import errno
import logging as log
import json
import numpy as np

from itertools import combinations

from self_driving.simulation_data import SimulationDataRecords

from code_pipeline.visualization import RoadTestVisualizer
from code_pipeline.tests_generation import TestGenerationStatistic
from code_pipeline.test_generation_utils import register_exit_fun
from code_pipeline.tests_evaluation import RoadTestEvaluator
from code_pipeline.edit_distance_polyline import iterative_levenshtein

OUTPUT_RESULTS_TO = 'results'


def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def validate_map_size(ctx, param, value):
    if value < 100 or value > 1000:
        raise click.UsageError('The provived value for ' + str(param) + ' is invalid. Choose an integer between 100 and 1000')
    else:
        return value


def validate_time_budget(ctx, param, value):
    if value <= 0:
        raise click.UsageError('The provived value for ' + str(param) + ' is invalid. Choose a positive integer')
    else:
        return value

# TODO Refactor and move away
from self_driving.simulation_data import SimulationDataRecord

def _load_test_data(execution_data_file):
    # Load the execution data
    with open(execution_data_file) as input_file:
        # TODO What if the test is not valid?
        json_data = json.load(input_file)
        road_data = json_data["road_points"]
        execution_data = [SimulationDataRecord(*record) for record in json_data["execution_data"]] \
            if "execution_data" in json_data else []
    return road_data, execution_data

def create_summary(result_folder, raw_data):
    log.info("Creating Reports")

    if type(raw_data) is TestGenerationStatistic:
        log.info("Creating Test Statistics Report:")
        summary_file = os.path.join(result_folder, "generation_stats.csv")
        csv_content = raw_data.as_csv()
        with open(summary_file, 'w') as output_file:
            output_file.write( csv_content)
        log.info("Test Statistics Report available: %s", summary_file)

    # TODO Refactor and move away
    log.info("Creating OOB Report")

    # Go over all the files in the result folder and extract the interesing road segments for each OOB
    road_test_evaluation = RoadTestEvaluator(road_length_before_oob=30,
                                             road_lengrth_after_oob=30)

    oobs = []
    for subdir, dirs, files in os.walk(result_folder, followlinks=False):
        # Consider only the files that match the pattern
        for sample_file in sorted([os.path.join(subdir, f) for f in files if f.startswith("test.") and f.endswith(".json")]):

            log.debug("Processing test file %s", sample_file)

            road_data, execution_data = _load_test_data(sample_file)

            # If the test was not valid skip the analysis
            if len(execution_data) == 0:
                log.debug(" Test was not valid. Skip")
                continue

            # Extract data about OOB, if any
            # TODO Probably check if test_outcome is FAIL
            oob_pos, segment_before, segment_after, oob_side = road_test_evaluation.identify_interesting_road_segments(
                    road_data, execution_data)

            if oob_pos is None:
                continue

            oobs.append(
                {
                    'simulation file': sample_file,
                    'oob point': oob_pos,
                    'oob side': oob_side,
                    'road segment before oob': segment_before,
                    'road segment after oob': segment_after,
                    # This is the list of points, so we need to extract from LineString objects
                    'interesting segment': list(segment_before.coords) + list(segment_after.coords)
                }
            )

    log.info("Collected data about %d oobs", len(oobs))

    max_distances_starting_from = {}

    for (oob1, oob2) in combinations(oobs, 2):
        # Compute distance between cells
        # check edit_distance_polyline.py inside illumination
        distance = iterative_levenshtein(oob1['interesting segment'], oob2['interesting segment'])

        if oob1['simulation file'] in max_distances_starting_from.keys():
            max_distances_starting_from[oob1['simulation file']] = max(max_distances_starting_from[oob1['simulation file']], distance)
        else:
            max_distances_starting_from[oob1['simulation file']] = distance

    if len(max_distances_starting_from) > 0:
        mean_distance = np.mean([list(max_distances_starting_from.values())])
        std_dev = np.std([list(max_distances_starting_from.values())])
    else:
        mean_distance = np.NaN
        std_dev = np.NaN

    log.info("Sparseness: Mean: %.3f, StdDev: %3f", mean_distance , std_dev)

    n_left = 0
    n_right = 0

    for oob in oobs:
        if oob['oob side'] == "LEFT":
            n_left += 1
        else:
            n_right += 1

    log.info("Left: %d - Right: %d", n_left, n_right)

    oob_summary_file = os.path.join(result_folder, "oob_stats.csv")
    with open(oob_summary_file, 'w') as output_file:
        output_file.write("total_oob,left_oob,right_oob,avg_sparseness,stdev_sparseness\n")
        output_file.write("%d,%d,%d,%.3f,%3.f\n" % (len(oobs), n_left, n_right, mean_distance, std_dev))

    log.info("OOB  Report available: %s", oob_summary_file)

def post_process(result_folder, the_executor):
    """
        This will be invoked after the generation is over. Whatever results is produced will be copied inside
        the result_folder
    """
    # Ensure the executor is stopped
    the_executor.close()

    # Plot the stats on the console
    log.info("Test Generation Statistics:")
    log.info(the_executor.get_stats())

    # Generate the actual summary files
    create_summary(result_folder, the_executor.get_stats())


def create_post_processing_hook(result_folder, executor):
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
            post_process(result_folder, executor)

    return _f



def log_exception(extype, value, trace):
    log.exception('Uncaught exception:', exc_info=(extype, value, trace))


def setup_logging(log_to, debug):
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
def generate(executor, beamng_home, beamng_user, time_budget, map_size, module_name, module_path, class_name, visualize_tests, log_to, debug):
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
        the_executor = MockExecutor(result_folder, time_budget, map_size, road_visualizer=road_visualizer)
    elif executor == "beamng":
        from code_pipeline.beamng_executor import BeamngExecutor
        the_executor = BeamngExecutor(result_folder, time_budget, map_size,
                                      beamng_home=beamng_home, beamng_user=beamng_user,
                                      road_visualizer=road_visualizer)

    # Register the shutdown hook for post processing results
    register_exit_fun(create_post_processing_hook(result_folder, the_executor))

    try:
        # Instantiate the test generator
        test_generator = the_class(time_budget=time_budget, executor=the_executor, map_size=map_size)
        # Start the generation
        test_generator.start()
    except Exception:
        log.fatal("An error occurred during test generation")
        traceback.print_exc()
        sys.exit(2)

    # We still need this here to post process the results if the execution takes the regular flow
    post_process(result_folder, the_executor)


if __name__ == '__main__':
    generate()
