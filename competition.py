#
#
#
#
import click
import importlib


@click.command()
@click.option('--executor', type=click.Choice(['mock', 'beamng'], case_sensitive=False), default="mock")
@click.option('--beamng-home', required=False, type=str)
@click.option('--time-budget', required=True, type=int)
@click.option('--map-size', type=int, default=200)
@click.option('--module-name', required=True, type=str)
# TODO Add type: File
@click.option('--module-path', required=False, type=str)
@click.option('--class-name', required=True, type=str)
def generate(executor, beamng_home, time_budget, map_size, module_name, module_path, class_name):
    if executor == "mock":
        from code_pipeline.executors import MockExecutor
        the_executor = MockExecutor(time_budget=time_budget, map_size=map_size)
    elif executor == "beamng":
        from code_pipeline.beamng_executor import BeamngExecutor
        the_executor = BeamngExecutor(beamng_home=beamng_home, time_budget=time_budget, map_size=map_size)

    # Dynamically load test generator
    module = importlib.import_module(module_name, module_path)
    class_ = getattr(module, class_name)
    test_generator = class_(time_budget=time_budget, executor=the_executor, map_size=map_size)

    try:
        # Start the generation
        test_generator.start()
    except Exception as ex:
        print("An error occurred during test generation", ex)
    finally:
        # When the generation ends. Print the stats collected
        print("Test Generation Statistics:")
        print(the_executor.get_stats())


if __name__ == '__main__':
    generate()