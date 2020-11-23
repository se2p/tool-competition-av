#
#
#
#
import click
import importlib
import os

@click.command()
@click.option('--executor', type=click.Choice(['mock', 'beamng'], case_sensitive=False), default="mock")
@click.option('--beamng-home', required=False, type=str)
@click.option('--time-budget', required=True, type=int)
@click.option('--map-size', type=int, default=200)
@click.option('--module-name', required=True, type=str)
# Add type: File
@click.option('--module-path', required=True, type=str)
@click.option('--class-name', required=True, type=str)
def generate(executor, beamng_home, time_budget, map_size, module_name, module_path, class_name):
    if executor == "mock":
        from executors import MockExecutor
        the_executor = MockExecutor(time_budget=time_budget, map_size=map_size)
    elif executor == "beamng":
        from beamng_executor import BeamngExecutor
        the_executor = BeamngExecutor(beamng_home=beamng_home, time_budget=time_budget, map_size=map_size)

    # Dynamically load test generator
    module = importlib.import_module(module_name, module_path)
    class_ = getattr(module, class_name)
    test_generator = class_(time_budget=time_budget, executor=the_executor, map_size=map_size)

    # Start the generation
    test_generator.start()


if __name__ == '__main__':
    generate()