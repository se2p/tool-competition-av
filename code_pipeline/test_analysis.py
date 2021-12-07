#
# This file contains the code originally developed for DeepHyperion for computing exemplary test features.
# Please refer to the following paper and github repo:
# 	Tahereh Zohdinasab, Vincenzo Riccio, Alessio Gambi, Paolo Tonella:
#   DeepHyperion: exploring the feature space of deep learning-based systems through illumination search.
#   ISSTA 2021: 79-90
#
#   Repo URL: https://github.com/testingautomated-usi/DeepHyperion
#

import logging as logger
import math

import numpy as np

from shapely.geometry import Point
from code_pipeline.utils import pairwise

THE_NORTH = [0, 1]

####################################################################################
# Private Utility Methods
####################################################################################


def _calc_angle_distance(v0, v1):
    at_0 = np.arctan2(v0[1], v0[0])
    at_1 = np.arctan2(v1[1], v1[0])
    return at_1 - at_0


def _calc_dist_angle(points):
    assert len(points) >= 2, f'at least two points are needed'

    def vector(idx):
        return np.subtract(points[idx + 1], points[idx])

    n = len(points) - 1
    result = [None] * (n)
    b = vector(0)
    for i in range(n):
        a = b
        b = vector(i)
        angle = _calc_angle_distance(a, b)
        distance = np.linalg.norm(b)
        result[i] = (angle, distance, [points[i+1], points[i]])
    return result

def _identify_segment(nodes):
    # result is angle, distance, [x2,y2], [x1,y1]
    result = _calc_dist_angle(nodes)

    segments = []
    SEGMENT_THRESHOLD = 15
    SEGMENT_THRESHOLD2 = 10
    ANGLE_THRESHOLD = 0.005

    # iterate over the nodes to get the turns bigger than the threshold
    # a turn category is assigned to each node
    # l is a left turn
    # r is a right turn
    # s is a straight segment
    # TODO: first node is always a s
    turns = []
    for i in range(0, len(result)):
        # result[i][0] is the angle
        angle_1 = (result[i][0] + 180) % 360 - 180
        if np.abs(angle_1) > ANGLE_THRESHOLD:
            if (angle_1) > 0:
                turns.append("l")
            else:
                turns.append("r")
        else:
            turns.append("s")

    # this generator groups the points belonging to the same category
    def grouper(iterable):
        prev = None
        group = []
        for item in iterable:
            if not prev or item == prev:
                group.append(item)
            else:
                yield group
                group = [item]
            prev = item
        if group:
            yield group

    # this generator groups:
    # - groups of points belonging to the same category
    # - groups smaller than 10 elements
    def supergrouper1(iterable):
        prev = None
        group = []
        for item in iterable:
            if not prev:
                group.extend(item)
            elif len(item) < SEGMENT_THRESHOLD2 and item[0] == "s":
                item = [prev[-1]] * len(item)
                group.extend(item)
            elif len(item) < SEGMENT_THRESHOLD and item[0] != "s" and prev[-1] == item[0]:
                item = [prev[-1]] * len(item)
                group.extend(item)
            else:
                yield group
                group = item
            prev = item
        if group:
            yield group

    # this generator groups:
    # - groups of points belonging to the same category
    # - groups smaller than 10 elements
    def supergrouper2(iterable):
        prev = None
        group = []
        for item in iterable:
            if not prev:
                group.extend(item)
            elif len(item) < SEGMENT_THRESHOLD:
                item = [prev[-1]] * len(item)
                group.extend(item)
            else:
                yield group
                group = item
            prev = item
        if group:
            yield group

    groups = grouper(turns)

    supergroups1 = supergrouper1(groups)

    supergroups2 = supergrouper2(supergroups1)

    count = 0
    segment_indexes = []
    segment_count = 0
    for g in supergroups2:
        if g[-1] != "s":
            segment_count += 1
        # TODO
        # count += (len(g) - 1)
        count += (len(g))
        # TODO: count -1?
        segment_indexes.append(count)

    # TODO
    # segment_indexes.append(len(turns) - 1)

    segment_begin = 0
    for idx in segment_indexes:
        segment = []
        # segment_end = idx + 1
        segment_end = idx
        for j in range(segment_begin, segment_end):
            if j == 0:
                segment.append([result[j][2][0], result[j][0]])
            segment.append([result[j][2][1], result[j][0]])
        segment_begin = segment_end
        segments.append(segment)

    return segment_count, segments

# TODO Possibly code duplicate
def _define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return None, np.inf

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    center = Point(cx, cy)

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return center, radius

####################################################################################
# Input Features
####################################################################################


# Count how many road segments form the road. A road segment is a segment of the road
# that has a constant curvature
# TODO Does this consider only turns?
def segment_count(the_test):
    nodes = the_test.interpolated_points
    count, _ = _identify_segment(nodes)
    return "COUNT_SEG", count


# Measure the coverage of road directions w.r.t. to the North (0,1) using the control points of the given road
# to approximate the road direction. By default we use 36 bins to have bins of 10 deg each
def direction_coverage(the_test, n_bins=25):
    coverage_buckets = np.linspace(0.0, 360.0, num=n_bins + 1)
    direction_list = []
    for a, b in pairwise(the_test.interpolated_points):
        # Compute the direction of the segment defined by the two points
        road_direction = [b[0] - a[0], b[1] - a[1]]
        # Compute the angle between THE_NORTH and the road_direction.
        # E.g. see: https://www.quora.com/What-is-the-angle-between-the-vector-A-2i+3j-and-y-axis
        # https://www.kite.com/python/answers/how-to-get-the-angle-between-two-vectors-in-python
        unit_vector_1 = road_direction / np.linalg.norm(road_direction)
        dot_product = np.dot(unit_vector_1, THE_NORTH)
        angle = math.degrees(np.arccos(dot_product))
        direction_list.append(angle)

    # Place observations in bins and get the covered bins without repetition
    covered_elements = set(np.digitize(direction_list, coverage_buckets))
    dir_coverage = len(covered_elements) / len(coverage_buckets)
    return "DIR_COV", dir_coverage


def max_curvature(the_test, w=5):
    min_radius = np.inf
    nodes = the_test.interpolated_points
    for i in range(len(nodes) - w):
        p1 = nodes[i]
        p2 = nodes[i + int((w - 1) / 2)]
        p3 = nodes[i + (w - 1)]

        # We care only about the radius of the circle here
        _, radius = _define_circle(p1, p2, p3)
        if radius < min_radius:
            min_radius = radius
    # Max curvature is the inverse of Min radius
    curvature = 1.0 / min_radius

    return "MAX_CURV", curvature

####################################################################################
# Output Features
####################################################################################


#  Standard Deviation of Steering Angle accounts for the variability of the steering
#   angle during the execution
def sd_steering(execution_data):
    steering = []
    for state in execution_data:
        steering.append(state.steering)
    sd_steering = np.std(steering)
    return "STD_SA", sd_steering


#  Mean of Lateral Position of the car accounts for the average behavior of the car, i.e.,
#   whether it spent most of the time traveling in the center or on the side of the lane
def mean_lateral_position(execution_data):
    lp = []
    for state in execution_data:
        lp.append(state.oob_distance)

    mean_lp = np.mean(lp)
    return "MEAN_LP", mean_lp


def max_lateral_position(execution_data):
    lp = []
    for state in execution_data:
        lp.append(state.oob_distance)

    max_lp = np.max(lp)
    return "MAX_LP", max_lp


def compute_all_features(the_test, execution_data):
    features = dict()
    # Input Features
    input_features = [max_curvature, direction_coverage, segment_count]

    # Output Features
    output_features = [sd_steering, mean_lateral_position, max_lateral_position]

    # Mixed Features
    mixed_features = []

    logger.debug("Computing input features")
    for h in input_features:
        key, value = h(the_test)
        features[key] = value

    # TODO Add minimum value here
    if len(execution_data) > 2:
        logger.debug("Computing output features")
        for h in output_features:
            key, value = h(execution_data)
            features[key] = value

        logger.debug("Computing mixed features")
        for h in mixed_features:
            key, value = h(the_test, execution_data)
            features[key] = value

    return features