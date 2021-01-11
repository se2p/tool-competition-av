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

from code_pipeline.visualization import RoadTestVisualizer
from code_pipeline.tests_generation import TestGenerationStatistic
from code_pipeline.test_generation_utils import register_exit_fun

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


def create_summary(result_folder, raw_data):
    if type(raw_data) is TestGenerationStatistic:
        summary_file = os.path.join(result_folder, "generation_stats.csv")
        csv_content = raw_data.as_csv()
        with open(summary_file, 'w') as output_file:
            output_file.write( csv_content )


def post_process(result_folder, the_executor):
    """
        This will be invoked after the generation is over. Whatever results is produced will be copied inside
        the result_folder
    """
    # Ensure the executor is stopped
    the_executor.close()

    log.info("Test Generation Statistics:")
    log.info(the_executor.get_stats())
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


def create_summary(result_folder, raw_data):
    if type(raw_data) is TestGenerationStatistic:
        summary_file = os.path.join(result_folder, "generation_stats.csv")
        csv_content = raw_data.as_csv()
        with open(summary_file, 'w') as output_file:
            output_file.write( csv_content )
    pass


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
@click.option('--executor', type=click.Choice(['mock', 'beamng'], case_sensitive=False), default="mock")
@click.option('--beamng-home', required=False, type=click.Path(exists=True))
@click.option('--beamng-user', required=False, type=click.Path(exists=True))
@click.option('--time-budget', required=True, type=int, callback=validate_time_budget)
@click.option('--map-size', type=int, default=200, callback=validate_map_size)
@click.option('--module-name', required=True, type=str)
@click.option('--module-path', required=False, type=click.Path(exists=True))
@click.option('--class-name', required=True, type=str)
# Visual Debugging
@click.option('--visualize-tests', required=False, is_flag=True, default=False, help = "Visualize the last generated test.")
# Logging options
@click.option('--log-to', required=False, type=click.Path(exists=False), help = "File to Log to. If not specified logs will show on the console")
@click.option('--debug', required=False, is_flag=True, default=False, help = "Activate debugging (more logging)")
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

    # Setup executor
    if executor == "mock":
        from code_pipeline.executors import MockExecutor
        the_executor = MockExecutor(time_budget=time_budget, map_size=map_size, road_visualizer=road_visualizer)
    elif executor == "beamng":
        # TODO Make sure this executor outputs the files in the results folder
        from code_pipeline.beamng_executor import BeamngExecutor
        the_executor = BeamngExecutor(beamng_home=beamng_home, beamng_user=beamng_user, time_budget=time_budget,
                                      map_size=map_size, road_visualizer=road_visualizer)

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
