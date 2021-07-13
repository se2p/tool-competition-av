import json
import os

class Sample:

    def __init__(self):
        self.id = None
        self.tool = None
        self.misbehaviour = False
        self.run = None
        self.timestamp = None
        self.elapsed = None
        self.features = {}
        self.is_valid = True
        self.valid_according_to = None

    # TODO Maybe make this an abstract method?
    def is_misbehavior(self):
        return self.misbehaviour

    def get_value(self, feature_name):
        if feature_name in self.features.keys():
            return self.features[feature_name]
        else:
            return None

    @staticmethod
    def from_dict(the_dict):
        sample = Sample()
        for k in sample.__dict__.keys():
            setattr(sample, k, None if k not in the_dict.keys() else the_dict[k])
        return sample


class BeamNGSample(Sample):
    # At which radius we interpret a tuns as a straight?
    # MAX_MIN_RADIUS = 200
    MAX_MIN_RADIUS = 170

    def __init__(self, basepath,
                 id,
                 is_valid,
                 int_points,
                 states,
                 test_data,
                 minRadius,
                 directionCoverage,
                 meanLateralPosition,
                 segmentCount,
                 sdSteeringAngle,
                 curvature,
                 fitnessFunction):
        super(BeamNGSample, self).__init__()
        self.basepath = basepath
        self.id = id,
        self.is_valid = is_valid,
        self.interpolated_points = int_points
        self.simulation_states = states
        self.test_data = test_data
        self.features["MinRadius"] = minRadius
        self.features["DirectionCoverage"] = directionCoverage
        self.features["MeanLateralPosition"] = meanLateralPosition
        self.features["SegmentCount"] = segmentCount
        self.features["SDSteeringAngle"] = sdSteeringAngle
        self.features["Curvature"] = curvature
        self.features["FitnessFunction"] = fitnessFunction

    # def visualize_misbehaviour(self):
    #     # THIS IS THE CODE FOR OOB
    #     # Create the road geometry from the nodes. At this point nodes have been reversed alredy if needed.
    #     road_geometry = metrics.get_geometry(self.road_nodes)
    #
    #     road_left_edge_x = np.array([e['left'][0] for e in road_geometry])
    #     road_left_edge_y = np.array([e['left'][1] for e in road_geometry])
    #
    #     left_edge_x = np.array([e['middle'][0] for e in road_geometry])
    #     left_edge_y = np.array([e['middle'][1] for e in road_geometry])
    #     right_edge_x = np.array([e['right'][0] for e in road_geometry])
    #     right_edge_y = np.array([e['right'][1] for e in road_geometry])
    #
    #     # Create the road polygon from the geometry
    #
    #     right_edge_road = LineString(zip(right_edge_x[::-1], right_edge_y[::-1]))
    #     left_edge_road = LineString(zip(road_left_edge_x, road_left_edge_y))
    #
    #     l_edge_road = left_edge_road.coords
    #     r_edge_road = right_edge_road.coords
    #
    #     road_polygon = Polygon(list(l_edge_road) + list(r_edge_road))
    #
    #     # Plot the road
    #     plt.gca().add_patch(PolygonPatch(road_polygon, fc='gray', alpha=0.5, zorder=2 ))
    #
    #
    #     # Create the right lane polygon from the geometry
    #     # Note that one must be in reverse order for the polygon to close correctly
    #     right_edge = LineString(zip(right_edge_x[::-1], right_edge_y[::-1]))
    #     left_edge = LineString(zip(left_edge_x, left_edge_y))
    #
    #     l_edge = left_edge.coords
    #     r_edge = right_edge.coords
    #
    #     right_lane_polygon = Polygon(list(l_edge) + list(r_edge))
    #
    #     # TODO Plot road as well to understand if this is exactly the side we thing it is
    #     plt.plot(*right_lane_polygon.exterior.xy, color='gray')
    #
    #     # Plot all the observations in trasparent green except the OOB
    #     for position in [Point(sample["pos"][0], sample["pos"][1]) for sample in self.simulation_states]:
    #         if right_lane_polygon.contains(position):
    #             plt.plot(position.x, position.y, 'o', color='green', alpha=0.2)
    #         else:
    #             plt.plot(position.x, position.y, 'o', color='red', alpha=1.0)
    #
    #     plt.gca().set_aspect('equal')
    #
    #
    # def _resampling(self, sample_nodes, dist=1.5):
    #     new_sample_nodes = []
    #     dists = []
    #     for i in range(1, len(sample_nodes)):
    #         x0 = sample_nodes[i - 1][0]
    #         x1 = sample_nodes[i][0]
    #         y0 = sample_nodes[i - 1][1]
    #         y1 = sample_nodes[i][1]
    #
    #         d = sqrt(pow((x1 - x0), 2) + pow((y1 - y0), 2))
    #         dists.append(d)
    #         if d >= dist:
    #             dt = dist
    #             new_sample_nodes.append([x0, y0, -28.0, 8.0])
    #             while dt <= d - dist:
    #                 t = dt / d
    #                 xt = ((1 - t) * x0 + t * x1)
    #                 yt = ((1 - t) * y0 + t * y1)
    #                 new_sample_nodes.append([xt, yt, -28.0, 8.0])
    #                 dt = dt + dist
    #             new_sample_nodes.append([x1, y1, -28.0, 8.0])
    #         else:
    #             new_sample_nodes.append([x0, y0, -28.0, 8.0])
    #             new_sample_nodes.append([x1, y1, -28.0, 8.0])
    #
    #     points_x = []
    #     points_y = []
    #     final_nodes = list()
    #     # discard the Repetitive points
    #     for i in range(1, len(new_sample_nodes)):
    #         if new_sample_nodes[i] != new_sample_nodes[i - 1]:
    #             final_nodes.append(new_sample_nodes[i])
    #             points_x.append(new_sample_nodes[i][0])
    #             points_y.append(new_sample_nodes[i][1])
    #     return final_nodes
    #
    # def compute_input_metrics(self, resampled_road_nodes):
    #     # Input features
    #     self.features["min_radius"] = metrics.capped_min_radius(self.MAX_MIN_RADIUS, resampled_road_nodes)
    #     self.features["segment_count"] = metrics.segment_count(resampled_road_nodes)
    #     self.features["direction_coverage"] = metrics.direction_coverage(resampled_road_nodes)
    #
    # def compute_output_metrics(self, simulation_states):
    #     # Output features
    #     self.features["sd_steering"] = metrics.sd_steering(simulation_states)
    #
    #     #self.features["mean_lateral_position"] = metrics.mean_absolute_lateral_position(simulation_states)
    #     road_geometry = metrics.get_geometry(self.road_nodes)
    #
    #     middle = [e['middle'] for e in road_geometry]
    #     right = [e['right'] for e in road_geometry]
    #     middle_points = [(p[0], p[1]) for p in middle]
    #     right_points = [(p[0], p[1]) for p in right]
    #
    #     right_polyline = compute_right_polyline(middle_points, right_points)
    #
    #     # road_spine = LineString(middle_points)
    #
    #     # road_polygon = _polygon_from_geometry(road_geometry)
    #     #
    #     # # Plot road
    #     # plt.plot(*road_polygon.exterior.xy)
    #     # # Plot centeral spine
    #     # plt.plot(*road_spine.xy, "r-")
    #     #
    #     # # LineString
    #
    #     # plt.plot(*right_polyline.xy)
    #     # positions = [ (state["pos"][0], state["pos"][1]) for state in simulation_states]
    #     #
    #     # for s in positions:
    #     #     plt.plot(s[0], s[1], 'ob')
    #     #     pass
    #     #for state in segment.simulation_states:
    #     #    dist = oob_distance(state["pos"], right_poly)
    #     #    dist2 = state["oob_distance"]
    #     #    assert (dist == dist2)
    #     self.features["mean_lateral_position"] = metrics.mean_lateral_position(simulation_states, right_polyline)

    def to_dict(self):
        """
            This is common for all the BeamNG samples
        """

        return {'id': self.id,
                'is_valid': self.is_valid,
                'valid_according_to': self.valid_according_to,
                'misbehaviour': self.is_misbehavior(),
                'elapsed': self.elapsed,
                'timestamp': self.timestamp,
                'interpolated_points': self.interpolated_points,
                'simulation_states': self.simulation_states,
                'test_data': self.test_data,
                'MinRadius': self.get_value("MinRadius"),
                'SegmentCount': self.get_value("SegmentCount"),
                'DirectionCoverage': self.get_value("DirectionCoverage"),
                'SDSteeringAngle': self.get_value("SDSteeringAngle"),
                'MeanLateralPosition': self.get_value("MeanLateralPosition"),
                'Curvature': self.get_value("Curvature"),
                'FitnessFunction': self.get_value("FitnessFunction"),
                'tool': self.tool,
                'run': self.run,
                'features': self.features}

    def dump(self):
        data = self.to_dict()
        filedest = os.path.join(os.path.dirname(self.basepath), "info_" + str(self.id) + ".json")
        with open(filedest, 'w') as f:
            (json.dump(data, f, sort_keys=True, indent=4))