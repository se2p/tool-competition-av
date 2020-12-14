#
#
#
#
import click
import importlib
import traceback
from code_pipeline.visualization import RoadTestVisualizer

from code_pipeline.test_generation_utils import register_exit_fun


OUTPUT_RESULTS_TO = 'results'


def validate_map_size(ctx, param, value):
    if value < 100 or value > 1000:
        raise click.UsageError('The provived value for ' + str(param) + ' is invalid. Choose an integer between 100 and 1000')
    else:
        return value


def post_process(the_executor):
    print("Test Generation Statistics:")
    print(the_executor.get_stats())


def create_post_processing_hook(executor):

    def _f():
        if executor.is_force_timeout():
            # The process killed itself because a timeout, so we need to ensure the post_process function
            # is called
            post_process(executor)

    return _f


@click.command()
@click.option('--executor', type=click.Choice(['mock', 'beamng'], case_sensitive=False), default="mock")
@click.option('--beamng-home', required=False, type=str)
@click.option('--time-budget', required=True, type=int)
@click.option('--map-size', type=int, default=200)
@click.option('--module-name', required=True, type=str)
# TODO Add type: File
@click.option('--module-path', required=False, type=str)
@click.option('--class-name', required=True, type=str)
# Visual Debugging
@click.option('--visualize-tests', required=False, is_flag=True, default=False, help="Visualize the last generated test.")
def generate(executor, beamng_home, time_budget, map_size, module_name, module_path, class_name, visualize_tests):
    road_visualizer = None
    road_plotter = None

    # Setup visualization
    if visualize_tests:
        road_visualizer = RoadTestVisualizer(map_size=map_size)
        pass

    if executor == "mock":
        from code_pipeline.executors import MockExecutor
        the_executor = MockExecutor(time_budget=time_budget, map_size=map_size, road_visualizer=road_visualizer)
    elif executor == "beamng":
        from code_pipeline.beamng_executor import BeamngExecutor
        the_executor = BeamngExecutor(beamng_home=beamng_home, time_budget=time_budget,
                                      map_size=map_size, road_visualizer=road_visualizer)

    # Dynamically load the test generator
    module = importlib.import_module(module_name, module_path)
    class_ = getattr(module, class_name)

    # Instantiate the test generator
    test_generator = class_(time_budget=time_budget, executor=the_executor, map_size=map_size)

    # Register shutdown hook to run the data analysis, but only if the internal timeout triggers. We do not want
    #   this to trigger if the use crtl+D his/her process, e.g., during development or debugging
    register_exit_fun(create_post_processing_hook(the_executor))

    try:
        # Start the generation
        test_generator.start()
    except Exception as ex:
        print("An error occurred during test generation")
        traceback.print_exc()

    # We still need this here for the regular flow
    post_process(the_executor)


if __name__ == '__main__':
    generate()