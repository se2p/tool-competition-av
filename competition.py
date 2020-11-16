#
#
#
#
import click
import importlib




@click.command()
@click.option('--executor', type=click.Choice(['mock', 'beamng'], case_sensitive=False), default="mock")
@click.option('--time-budget', required=True, type=int)
@click.option('--map-size', type=int, default=200)
@click.option('--module-name', required=True, type=str)
@click.option('--class-name', required=True, type=str)
def generate(executor, time_budget, map_size, module_name, class_name):
    if executor == "mock":
        from executors import MockExecutor
        the_executor = MockExecutor(time_budget=time_budget, map_size=map_size)
    elif executor == "beamng":
        from beamng_executor import BeamngExecutor
        the_executor = BeamngExecutor(time_budget=time_budget, map_size=map_size)

    # Dynamically load test generator
    # TODO Where the code should be placed? on the python path is enought?
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    test_generator = class_(time_budget=time_budget, executor=the_executor, map_size=map_size)

    # Start the generation
    test_generator.start()


if __name__ == '__main__':
    generate()