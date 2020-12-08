import unittest
from code_pipeline.validation import TestValidator
import inspect

class ValidationTest(unittest.TestCase):

    def test_road_that_stars_outside_the_map(self):
        """
        creates a road that start from outside the map. By convention the map is defined as
        (0,0), (map_size, map_size)
        :return:
        """
        print("Running test", inspect.stack()[0][3])

        the_road = []
        the_road.append((-10, -10, -28.0, 8.0))
        the_road.append((50, 50, -28.0, 8.0))

        validator = TestValidator(map_size=200)
        is_valid, validation_msg = validator.validate_test(the_road)

        self.assertFalse(is_valid)

    def test_road_that_ends_outside_the_map(self):
        """
        creates a road that start inside the map but ends outside it.
        :return:
        """
        print("Running test", inspect.stack()[0][3])

        the_road = []
        the_road.append((50, 50, -28.0, 8.0))
        the_road.append((-10, -10, -28.0, 8.0))

        validator = TestValidator(map_size=200)
        is_valid, validation_msg = validator.validate_test(the_road)

        self.assertFalse(is_valid)

    def test_road_that_is_entirely_outside_the_map(self):
        """
        creates a road that stays entirely outside the map
        :return:
        """

        print("Running test", inspect.stack()[0][3])

        the_road = []
        the_road.append((-50, -50, -28.0, 8.0))
        the_road.append((-10, -10, -28.0, 8.0))

        validator = TestValidator(map_size=200)
        is_valid, validation_msg = validator.validate_test(the_road)

        self.assertFalse(is_valid)

    def test_road_that_is_entirely_inside_the_map(self):
        """
        creates a road that stays entirely outside the map
        :return:
        """

        print("Running test", inspect.stack()[0][3])

        the_road = []
        the_road.append((50, 50, -28.0, 8.0))
        the_road.append((10, 10, -28.0, 8.0))

        validator = TestValidator(map_size=200)
        is_valid, validation_msg = validator.validate_test(the_road)

        self.assertTrue(is_valid)

    def test_road_side_partially_outside(self):
        """
        creates a road that stays entirely outside the map
        :return:
        """

        print("Running test", inspect.stack()[0][3])

        the_road = []
        the_road.append((1, 10, -28.0, 8.0))
        the_road.append((1, 50, -28.0, 8.0))

        validator = TestValidator(map_size=200)
        is_valid, validation_msg = validator.validate_test(the_road)

        self.assertFalse(is_valid)

    def test_road_self_intersect(self):
        """
        creates a road that stays entirely outside the map
        :return:
        """

        print("Running test", inspect.stack()[0][3])

        the_road = []
        the_road.append((10, 10, -28.0, 8.0))
        the_road.append((20, 20, -28.0, 8.0))
        the_road.append((10, 20, -28.0, 8.0))
        the_road.append((20, 10, -28.0, 8.0))

        validator = TestValidator(map_size=200)
        is_valid, validation_msg = validator.validate_test(the_road)

        self.assertFalse(is_valid)

    def test_road_self_overlapping(self):
        """
        creates a road that stays entirely outside the map
        :return:
        """

        print("Running test", inspect.stack()[0][3])

        the_road = []
        the_road.append((10, 70, -28.0, 8.0))
        the_road.append((10, 80, -28.0, 8.0))
        the_road.append((15, 95, -28.0, 8.0))
        the_road.append((15, 80, -28.0, 8.0))
        the_road.append((15, 70, -28.0, 8.0))

        validator = TestValidator(map_size=200)
        is_valid, validation_msg = validator.validate_test(the_road)

        self.assertFalse(is_valid)


if __name__ == '__main__':
    unittest.main()