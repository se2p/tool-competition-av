import unittest
from click.testing import CliRunner

from competition import generate

# FOR SOME REASONS THOSE TESTS WORK ONLY IF WE START ONE AFTER THE OTHER MANUALLY
# I SUSPECT THE TEST DO NOT WAY FOR THE runner TO FINISH
class CommandLineCombinationTest(unittest.TestCase):

    def test_fail_when_model_is_missing(self):
        runner = CliRunner()
        result = runner.invoke(generate, ['--executor', 'dave2', '--time-budget', '10', '--map-size', '200', '--module-name', 'foo', '--class-name', 'Foo'])
        print(result.output)
        assert "Error: If executor is set to dave2 the option --dave2-model must be specified" in result.output


    def test_do_not_fail_when_model_is_missing_but_executor_is_different(self):
        runner = CliRunner()
        result = runner.invoke(generate, ['--executor', 'mock', '--time-budget', '10', '--map-size', '200', '--module-name', 'foo', '--class-name', 'Foo'])
        print("1" + result.output)
        assert "Started test generation" in result.output


    def test_do_fail_when_model_is_present_but_executor_is_different(self):
        runner = CliRunner()
        result = runner.invoke(generate, ['--executor', 'mock', '--dave2-model', '../dave2/self-driving-car-010-2020.h5', '--time-budget', '10', '--map-size', '200', '--module-name', 'foo', '--class-name', 'Foo'])
        print("2" + result.output)
        assert "Started test generation" in result.output


    def test_do_not_fail_when_model_is_present(self):
        runner = CliRunner()
        result = runner.invoke(generate, ['--executor', 'dave2', '--dave2-model', '../dave2/self-driving-car-010-2020.h5', '--time-budget', '10', '--map-size', '200', '--module-name', 'foo', '--class-name', 'Foo'])
        print("3" + result.output)
        assert "Started test generation" in result.output