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

from code_pipeline.visualization import RoadTestVisualizer
from code_pipeline.tests_generation import TestGenerationStatistic


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
    pass

@click.command()
@click.option('--executor', type=click.Choice(['mock', 'beamng'], case_sensitive=False), default="mock")
@click.option('--beamng-home', required=False, type=click.Path(exists=True))
@click.option('--time-budget', required=True, type=int, callback=validate_time_budget)
@click.option('--map-size', type=int, default=200, callback=validate_map_size)
@click.option('--module-name', required=True, type=str)
@click.option('--module-path', required=False, type=click.Path(exists=True))
@click.option('--class-name', required=True, type=str)
# Visual Debugging
@click.option('--visualize-tests', required=False, is_flag=True, default=False, help="Visualize the last generated test.")
def generate(executor, beamng_home, time_budget, map_size, module_name, module_path, class_name, visualize_tests):
    # Setup test generator
    # TODO Probably we should validate this somehow?
    # Dynamically load the test generator
    module = importlib.import_module(module_name, module_path)
    the_class = getattr(module, class_name)

    # Pre Execution Setup
    road_visualizer = None

    # Setup visualization
    if visualize_tests:
        road_visualizer = RoadTestVisualizer(map_size=map_size)

    # Setup folder structure

    # Ensure base folder is there.
    # For the moment we HARDCODE the location of the output folder
    default_output_folder = os.path.join(get_script_path(), OUTPUT_RESULTS_TO)
    try:
        os.makedirs(default_output_folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Create the folder that will host the results of this execution
    # Unique ID of the run
    timestamp_id = time.time_ns() // 1000000

    result_folder = os.path.join(default_output_folder, "_".join([str(module_name), str(class_name), str(timestamp_id)]))

    try:
        os.makedirs(result_folder)
    except OSError as e:
        raise

    # TODO Use a logger
    print("Outputting results to " + result_folder)

    # Setup executor
    if executor == "mock":
        from code_pipeline.executors import MockExecutor
        the_executor = MockExecutor(time_budget=time_budget, map_size=map_size, road_visualizer=road_visualizer)
    elif executor == "beamng":
        # TODO Make sure this executor outputs the files in the results folder
        from code_pipeline.beamng_executor import BeamngExecutor
        the_executor = BeamngExecutor(beamng_home=beamng_home, time_budget=time_budget,
                                      map_size=map_size, road_visualizer=road_visualizer)

    try:
        # Instantiate the test generator
        test_generator = the_class(time_budget=time_budget, executor=the_executor, map_size=map_size)
        # Start the generation
        # TODO Consider moving this into a process to enforce global timeouts
        test_generator.start()
    except Exception as ex:
        print("An error occurred during test generation")
        traceback.print_exc()

    finally:
        # TODO Use a logger
        # When the generation ends. Print generic stats, more details will be provided in the summary
        print("Test Generation Statistics:")
        print(the_executor.get_stats())

    # Run the analysis and generate the summary
    # TODO Consider configuring this via command line arguments or moving it to a different (sub-)command
    # Analyses can be implemented as plugins, but how can we pass data to from them? Default constructor?
    # Here I would go with:
    # for analysise in configured_analyses:
    #   analysise.analyze(all_execution_data?)
    #   analysise.create_summary(result_folder)
    # TODO Consider encapsulating this into a class
    create_summary(result_folder, the_executor.get_stats())



if __name__ == '__main__':
    generate()