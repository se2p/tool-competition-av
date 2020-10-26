class TestValidator:

    def __init__(self, map_size):
        self.map_size = map_size

    def is_not_self_intersecting(self, the_test):
        pass

    def is_not_overlapping(self, the_test):
        pass

    def is_inside_map(self, the_test):
        """ Take the extreme points and ensure that their distance is smaller than the map side"""
        min_x, max_x = min(the_test)[0], max(the_test)[0]
        min_y, max_y = min(the_test)[1], max(the_test)[1]
        return max_x - min_x <= self.map_size and max_y - min_y <= self.map_size

    def is_right_type(self, the_test):
        pass

    def validate_test(self, the_test):

        is_valid = True
        validation_msg = ""

        if not self.is_inside_map(the_test):
            is_valid = False
            validation_msg = "Not entirely inside the map boundaries"
            return is_valid, validation_msg

        return is_valid, validation_msg