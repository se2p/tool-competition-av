from self_driving.road_polygon import RoadPolygon
from shapely.geometry import LineString

class TestValidator:

    def __init__(self, map_size, min_road_lenght = 20):
        self.map_size = map_size
        self.min_road_lenght = min_road_lenght

    def is_not_self_intersecting(self, the_test):
        road_polygon = RoadPolygon.from_nodes(the_test)
        return road_polygon.is_valid()

    def is_inside_map(self, the_test):
        """ Take the extreme points and ensure that their distance is smaller than the map side"""
        xs = [t[0] for t in the_test]
        ys = [t[1] for t in the_test]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        return max_x - min_x <= self.map_size and max_y - min_y <= self.map_size

    def is_right_type(self, the_test):
        check_list = type(the_test) is list
        check_node_type = all(len(i) == 4 for i in the_test)
        check = check_list and check_node_type
        # This is a bit too strick but will fail the execution

        return check

    def is_minimum_length(self, the_test):
        # This is approximated because at this point the_test is not yet interpolated
        return LineString([(t[0], t[1]) for t in the_test]).length > self.min_road_lenght

    def validate_test(self, the_test):
        # TODO This requires 4-tuple so we need to transform the_test appropriately
        the_test_as_4tuple = [(float(t[0]), float(t[1]), -28.0, 8.0) for t in the_test]

        is_valid = True
        validation_msg = ""

        if not self.is_inside_map(the_test_as_4tuple):
            is_valid = False
            validation_msg = "Not entirely inside the map boundaries"
            return is_valid, validation_msg
        if not self.is_not_self_intersecting(the_test_as_4tuple):
           is_valid = False
           validation_msg = "The road is self-intersecting"
           return is_valid, validation_msg
        if not self.is_right_type(the_test_as_4tuple):
            is_valid = False
            validation_msg = "Wrong type"
            return is_valid, validation_msg
        if not self.is_minimum_length(the_test_as_4tuple):
            is_valid = False
            validation_msg = "The road is not long enough."
            return is_valid, validation_msg

        return is_valid, validation_msg