import json
import uuid
from typing import Tuple, List


class DecalRoad:
    DEFAULT_MATERIAL = 'tig_road_rubber_sticky'

    def __init__(self, name,
                 material=DEFAULT_MATERIAL,
                 persistentId=None,
                 drivability=1):
        self.name = name
        self.material = material
        self.persistentId = persistentId if persistentId else str(uuid.uuid4())
        self.nodes = []
        self.drivability = drivability

    def add_4d_points(self, nodes: List[Tuple[float, float, float, float]]):
        self._safe_add_nodes(nodes)
        assert len(nodes) > 0, 'nodes should be a non empty list'
        assert all(len(item) == 4 for item in nodes), 'nodes list should contain tuple of 4 elements'
        assert all(all(isinstance(val, float) for val in item) for item in nodes), \
            'points list can contain only float'
        self.nodes += [list(item) for item in nodes]
        return self

    # unused
    def add_2d_points(self, points2d: List[Tuple[float, float]], z=-28, width=8):
        self._safe_add_nodes(points2d)
        assert len(points2d) > 0, 'points2d should be a non empty list'
        assert all(len(item) == 2 for item in points2d), 'points2d list should contain tuple of 2 elements'
        assert all(all(isinstance(val, float) for val in item) for item in points2d), \
            'points list can contain only float'
        self.nodes += [(p[0], p[1], z, width) for p in points2d]
        return self

    def to_dict(self):
        return {
            'name': self.name,
            'nodes': self.nodes
        }

    @classmethod
    def from_dict(cls, d):
        return DecalRoad(name=d['name']).add_4d_points(nodes=d['nodes'])

    def _safe_add_nodes(self, nodes: List):
        l = len(nodes) + len(self.nodes)
        #assert l < 540, f'BeamNG has issues with roads with more than 540 points. This road would have {l} nodes'

    def to_json(self):
        assert len(self.nodes) > 0, 'there are no points in this road'
        roadobj = {}
        roadobj['name'] = self.name
        roadobj['class'] = 'DecalRoad'
        roadobj['breakAngle'] = 180
        roadobj['distanceFade'] = [1000, 1000]
        roadobj['drivability'] = self.drivability
        roadobj['material'] = self.material
        roadobj['overObjects'] = True
        roadobj['persistentId'] = self.persistentId
        roadobj['__parent'] = 'generated'
        roadobj['position'] = tuple(self.nodes[0][:3])  # keep x,y,z discard width
        roadobj['textureLength'] = 2.5
        roadobj['nodes'] = self.nodes
        return json.dumps(roadobj)
