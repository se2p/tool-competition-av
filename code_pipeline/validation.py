from self_driving.bbox import RoadBoundingBox

from shapely.geometry import LineString

from code_pipeline.tests_generation import RoadTest

class TestValidator:

    def __init__(self, map_size, min_road_length = 20):
        self.map_size = map_size
        self.box = (0, 0, map_size, map_size)
        self.road_bbox = RoadBoundingBox(self.box)
        self.min_road_length = min_road_length
        # Not sure how to set this value... This might require to compute some sort of density: not points that are too
        # close to each others
        self.max_points = 500

    def is_enough_road_points(self, the_test):
        return len(the_test.road_points) > 1

    def is_too_many_points(self, the_test):
        return len(the_test.road_points) > self.max_points

    def is_not_self_intersecting(self, the_test):
        road_polygon = the_test.get_road_polygon()
        return road_polygon.is_valid()

    def is_inside_map(self, the_test):
        """
            Take the extreme points and ensure that their distance is smaller than the map side
        """
        xs = [t[0] for t in the_test.interpolated_points]
        ys = [t[1] for t in the_test.interpolated_points]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        return 0 < min_x or min_x > self.map_size and \
               0 < max_x or max_x > self.map_size and \
               0 < min_y or min_y > self.map_size and \
               0 < max_y or max_y > self.map_size

    def is_right_type(self, the_test):
        """
            The type of the_test must be RoadTest
        """
        check = type(the_test) is RoadTest
        return check

    def is_valid_polygon(self, the_test):
        road_polygon = the_test.get_road_polygon()
        check = road_polygon.is_valid()
        return check

    def intersects_boundary(self, the_test):
        road_polygon = the_test.get_road_polygon()
        check = self.road_bbox.intersects_boundary(road_polygon.polygon)
        return check

    def is_minimum_length(self, the_test):
        # This is approximated because at this point the_test is not yet interpolated
        return the_test.get_road_length() > self.min_road_length

    def validate_test(self, the_test):

        is_valid = True
        validation_msg = ""

        if not self.is_right_type(the_test):
            is_valid = False
            validation_msg = "Wrong type"
            return is_valid, validation_msg

        if not self.is_enough_road_points(the_test):
            is_valid = False
            validation_msg = "Not enough road points."
            return is_valid, validation_msg

        if self.is_too_many_points(the_test):
            is_valid = False
            validation_msg = "The road definition contains too many points"
            return is_valid, validation_msg

        if not self.is_inside_map(the_test):
            is_valid = False
            validation_msg = "Not entirely inside the map boundaries"
            return is_valid, validation_msg

        if self.intersects_boundary(the_test):
            is_valid = False
            validation_msg = "Not entirely inside the map boundaries"
            return is_valid, validation_msg

        if not self.is_valid_polygon(the_test):
            is_valid = False
            validation_msg = "The road is self-intersecting"
            return is_valid, validation_msg

        if not self.is_minimum_length(the_test):
            is_valid = False
            validation_msg = "The road is not long enough."
            return is_valid, validation_msg

        return is_valid, validation_msg