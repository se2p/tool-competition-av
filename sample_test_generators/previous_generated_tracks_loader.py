import logging as log
import json
from glob import glob
from sample_test_generators.config import TestGeneratorConfig

from code_pipeline.tests_generation import RoadTestFactory


class PreviousGeneratedTracksLoader():
    """
        Load tracks from previously generated runs.
    """

    def __init__(self, time_budget=None, executor=None, map_size=None):
        self.time_budget = time_budget
        self.executor = executor
        self.map_size = map_size
        config = TestGeneratorConfig()
        self.tracks_path = config.tracks_path

    def start(self):
        log.info("Load generated tracks")

        roads = []
        tracks_files = glob(self.tracks_path + "/*.json")
        for file in tracks_files:
            json_file = open(file, "r")
            json_data = json.load(json_file)
            roads.append(json_data["road_points"])

        # Create RoadTests for all loaded tracks
        tests = []
        for road_points in roads:
            tests.append(RoadTestFactory.create_road_test(road_points))

        # Execute all generated RoadTests
        for test in tests:
            test_outcome, description, execution_data = self.executor.execute_test(test)
            # Print test outcome
            log.info("test_outcome %s", test_outcome)
            log.info("description %s", description)

        import time
        time.sleep(10)
