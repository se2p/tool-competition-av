from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

from scipy.interpolate import splev, splprep

from numpy.ma import arange

from math import sqrt

from itertools import islice
import functools
import os
import json

BEFORE_THRESHOLD = 60.0
AFTER_THRESHOLD = 20.0


def _interpolate_and_resample_splines(sample_nodes, nodes_per_meter = 1, smoothness=0, k=3, rounding_precision=4):
    """ Interpolate a list of points as a spline (quadratic by default) and resample it with num_nodes"""

    # Compute lenght of the road
    road_lenght = LineString([(t[0], t[1]) for t in sample_nodes]).length

    num_nodes = nodes_per_meter  * int(road_lenght)

    old_x_vals = [t[0] for t in sample_nodes]
    old_y_vals = [t[1] for t in sample_nodes]
    # old_width_vals  = [t[3] for t in sample_nodes]

    # Interpolate the old points
    pos_tck, pos_u = splprep([old_x_vals, old_y_vals], s=smoothness, k=k)

    # Resample them
    step_size = 1 / num_nodes
    unew = arange(0, 1 + step_size, step_size)

    new_x_vals, new_y_vals = splev(unew, pos_tck)

    # Reduce floating point rounding errors otherwise these may cause problems with calculating parallel_offset
    return list(zip([round(v, rounding_precision) for v in new_x_vals],
                    [round(v, rounding_precision) for v in new_y_vals],
                    # TODO Brutally hard-coded
                    [-28.0 for v in new_x_vals],
                    [8.0 for w in new_x_vals]))


def _find_circle_and_return_the_center_and_the_radius(x1, y1, x2, y2, x3, y3):
    """THIS IS ONLY TO AVOID BREAKING OLD CODE"""
    x12 = x1 - x2;
    x13 = x1 - x3;

    y12 = y1 - y2;
    y13 = y1 - y3;

    y31 = y3 - y1;
    y21 = y2 - y1;

    x31 = x3 - x1;
    x21 = x2 - x1;

    # x1^2 - x3^2
    sx13 = pow(x1, 2) - pow(x3, 2);

    # y1^2 - y3^2
    sy13 = pow(y1, 2) - pow(y3, 2);

    sx21 = pow(x2, 2) - pow(x1, 2);
    sy21 = pow(y2, 2) - pow(y1, 2);

    f = (((sx13) * (x12) + (sy13) *
          (x12) + (sx21) * (x13) +
          (sy21) * (x13)) // (2 *
                              ((y31) * (x12) - (y21) * (x13))));

    g = (((sx13) * (y12) + (sy13) * (y12) +
          (sx21) * (y13) + (sy21) * (y13)) //
         (2 * ((x31) * (y12) - (x21) * (y13))));

    c = (-pow(x1, 2) - pow(y1, 2) -
         2 * g * x1 - 2 * f * y1);

    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
    # where centre is (h = -g, k = -f) and
    # radius r as r^2 = h^2 + k^2 - c
    h = -g;
    k = -f;
    sqr_of_r = h * h + k * k - c;

    # r is the radius
    r = round(sqrt(sqr_of_r), 5);

    return ((h, k), r)


    #print("Centre = (", h, ", ", k, ")");
    #print("Radius = ", r);
    #print("Radius = ", degrees(r));


def _road_segments_grouper(iterable, radius_tolerance=0.3):
    """
        Group road segments by similarity. Similarity is defined by type, radius and the distance between
        interpolating circle centers
    """
    prev = None
    group = []
    next_index = -1
    for index, item in enumerate(iterable):

        if index < next_index:
            continue
        if index == next_index:
            # Initialize the group with the element we identified two steps ago
            group.append(item)
            prev = item
            continue

        # Create a new group if this is the first element
        if not prev:
            group.append(item)
        elif prev["type"] == "straight" and item["type"] == "straight":
            group.append(item)
        elif prev["type"] == "straight" and item["type"] == "turn" or \
                prev["type"] == "turn" and item["type"] == "straight":
            # print("Returning group", prev["type"], "->", item["type"], group)
            # Return the current group
            yield group
            # Prepare the next group
            # prev = None
            group = []
            # Skip then next two elements
            next_index = index + 2
            continue
        else:

            assert prev["type"] != "straight"
            assert item["type"] != "straight"

            perc_diff_prev = abs(prev["radius"] - item["radius"]) / prev["radius"]
            perc_diff_item = abs(prev["radius"] - item["radius"]) / item["radius"]
            distance_between_centers = Point(prev["center"][0], prev["center"][1]).distance( Point(item["center"][0], item["center"][1]))
            if perc_diff_prev < radius_tolerance and perc_diff_item < radius_tolerance and \
                distance_between_centers < prev["radius"] and distance_between_centers < item["radius"]:
                group.append(item)
            else:
                # print("Returning group", prev["type"], "->", item["type"], group)
                # Return the current group
                yield group
                # Prepare the next group
                # prev = None
                group = []
                # Skip then next two elements
                next_index = index + 2
                continue

        prev = item

    # Not sure about this one...
    # Might cause consecutive straights to be reported?
    if group:
        # print("Returning last group ", group)
        # print("\n\n\n")
        yield group


# Merge two segments, but keeping the attributes from the first, but the points from everybody without duplicates
# This cannot be easily fit into a lambda
def _merge_segments_points(s1, s2):
    s1["points"].append(s2["points"][-1])
    return s1


def _merge_segments(s1, s2):
    s1["points"].extend(s2["points"])
    return s1


def _window(seq, n=2):
    """
    Returns a sliding window (of width n) over data from the iterable
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    Taken from: https://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator

    :param seq:
    :param n:
    :return:
    """

    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def _identify_segments(nodes):
    """
        Return grouping of nodes. Each group correspond to a segment [[][]]
        Assumptions: Lines are smooth, so there's no two consecutive straight segments that are not divided
        by a turn.

    """
    assert len(nodes) > 3, "not enough nodes"

    segments = []

    # Accumulate all the segments based on 3 points, and then simplify
    # Probably can do also p1, p2, p3 in ...
    for three_points in _window(nodes, n=3):

        center, radius = _find_circle_and_return_the_center_and_the_radius(
                                        three_points[0][0], three_points[0][1],#
                                        three_points[1][0], three_points[1][1], #
                                        three_points[2][0], three_points[2][1])

        if radius > 400:
            type = "straight"
            center = None
            radius = None
        else:
            type = "turn"

        current_segment = {}

        current_segment["type"] = type
        current_segment["center"] = center
        current_segment["radius"] = radius
        current_segment["points"] = []
        current_segment["points"].append(three_points[0])
        current_segment["points"].append(three_points[1])
        current_segment["points"].append(three_points[2])

        segments.append(current_segment)

    # Simplify the list of segments by grouping together similar elements
    # This procedure is flawed: it can report s->s but also spurios t->t (which are t->s). The issue is that
    # By looking at three points in isolation we can have confiugartions like this:
    # A, B, C -> s1
    # B, C, D -> t1
    # C, D, E -> t1 or s2 does not matter
    # D, E, F -> s2
    # E, F, G -> s2
    # The procedure understands that s1 is not compatible with t1 because of D, so it creates a group
    # Then skips two tripletted, so D becomes the first point (no overlap). But now the triplette is a straight not a turn...

    segments = list(_road_segments_grouper(segments))

    # Make sure you consider list of points and not triplettes.
    for index, segment in enumerate(segments[:]):
        # Replace the element with the merged one
        segments[index] = functools.reduce(lambda a, b: _merge_segments_points(a, b), segment)

    # Resolve and simplify. If two consecutive segments are straights we put them together.
    refined_segments = []

    # If two consecutive segments are similar we put them together
    for s in segments:
        if len(refined_segments) == 0:
            refined_segments.append(s)
        elif refined_segments[-1]["type"] == "straight" and s["type"] == "straight":
            # print("Merging ", refined_segments[-1], "and", s)
            # Take the points from the second segment put them into the first and return the first
            refined_segments[-1] = _merge_segments(refined_segments[-1], s)
        else:
            refined_segments.append(s)


    # At this point we have computed an approximation but we might need to smooth the edges, as
    # there might be little segments that could be attached to the previous ones

    # Associate small segments to prev/latest based on similarity
    segments = []

    # Move forward
    for index, segment in enumerate(refined_segments[:]):
        if len(segments) == 0:
            segments.append(segment)
        elif len(segment["points"]) <= 5:

            # Merge this segment to the previous one if they have the same type
            if segments[-1]["type"] == segment["type"]:
                segments[-1] = _merge_segments(segments[-1], segment)
            # Merge short straights into turns, but never turns into straights?
            else:
                segments.append(segment)
        else:
            segments.append(segment)

    # Repeat the process but moving backward
    refined_segments = segments[:]
    reversed(refined_segments)
    segments = []
    for index, segment in enumerate(refined_segments[:]):
        if len(segments) == 0:
            segments.append(segment)
        elif len(segment["points"]) <= 5:

            # Merge this segment to the previous one if they have the same type
            if segments[-1]["type"] == segment["type"]:
                segments[-1] = _merge_segments(segments[-1], segment)
            # Merge short straights into turns, but never turns into straights?
            else:
                segments.append(segment)
        else:
            segments.append(segment)

    reversed(segments)

    return segments


def _test_failed_with_oob(json_file):
    """
        Load the test from the json file and check the relevant attributes. The test must be valid, and FAILED because
        of OOB
    """
    with open(json_file, 'r') as test_json:
        data = json.load(test_json)
    return data["is_valid"] and data["test_outcome"] == "FAILED" and data["description"].startswith("Car drove out of the lane")


class RoadTestEvaluator:

    def __init__(self, road_length_before_oob = BEFORE_THRESHOLD, road_lengrth_after_oob = AFTER_THRESHOLD):
        self.road_length_before_oob = road_length_before_oob
        self.road_lengrth_after_oob = road_lengrth_after_oob

    # Note execution data also contains the road
    def identify_interesting_road_segments(self, execution_data):
        # Interpolate and resample
        road_points = _interpolate_and_resample_splines(execution_data["road"]["nodes"])

        # Create a LineString out of the road_points
        road_line = LineString([(rp[0], rp[1]) for rp in road_points])

        oob_pos = None
        # TODO This should be the last observation, so we should iterate the list from the last
        # Assuming we stop the execution at OBE
        positions = []
        for record in execution_data["records"]:
            positions.append(Point(record["pos"][0], record["pos"][1]))
            if record["is_oob"]:
                oob_pos = Point(record["pos"][0], record["pos"][1])

        if oob_pos == None:
            # No oob, no interesting segments
            return []

        # Find the point in the interpolated points that is closes to the OOB position
        # https://stackoverflow.com/questions/24415806/coordinates-of-the-closest-points-of-two-geometries-in-shapely
        np = nearest_points(road_line, oob_pos)[0]

        # https://gis.stackexchange.com/questions/84512/get-the-vertices-on-a-linestring-either-side-of-a-point
        before = None
        after = None

        road_coords = list(road_line.coords)
        for i, p in enumerate(road_coords):
            if Point(p).distance(np) < 0.5:  # Since we interpolate at every meter, whatever is closer than half od if
                before = road_coords[0:i]
                before.append(np.coords[0])

                after = road_coords[i:]

        # Take the M meters 'before' the OBE or the entire segment otherwise
        distance = 0
        temp = []
        for p1, p2 in _window(reversed(before), 2):

            if len(temp) == 0:
                temp.append(p1)

            distance += LineString([p1, p2]).length

            if distance >= self.road_length_before_oob:
                break
            else:
                temp.insert(0, p2)

        segment_before = LineString(temp)

        distance = 0
        temp = []
        for p1, p2 in _window(after, 2):

            if len(temp) == 0:
                temp.append(p1)

            distance += LineString([p1, p2]).length

            if distance >= self.road_lengrth_after_oob:
                break
            else:
                temp.append(p2)

        segment_after = LineString(temp)

        # Identify the road segments from ALL the "interesting part of the road"
        interesting_road_segments = _identify_segments(list(segment_before.coords) + list(segment_after.coords))

        # TODO Mark the road segment that contains the OOB?

        # Return them
        return interesting_road_segments


class UniqueOOBAnalysis:

    def __init__(self, result_folder):
        self.result_folder = result_folder

    def _measure_distance_between(self, oob1, oob2):
        # TODO Implement this
        pass

    def analyse(self):
        """
            Iterate over the result_folder, identify the OOB and measure their relative distance, and ... TODO
        """
        file_names = [os.path.join(self.result_folder, fn) for fn in os.listdir(self.result_folder)
                      if fn.startswith('test.') and fn.endswith('json')]

        # Filter tests that have oob

        file_names = filter(_test_failed_with_oob, file_names)

        for file_name in file_names:
            print(file_name)


        pass

    def create_summary(self):
        """ TODO Create whatever summary we need from this """
        pass
