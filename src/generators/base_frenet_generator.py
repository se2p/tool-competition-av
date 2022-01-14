from src.generators.base_generator import BaseGenerator
import src.utils.frenet as frenet
import numpy as np
import logging as log
from scipy import spatial
from shapely.geometry import Polygon

class BaseFrenetGenerator(BaseGenerator):
    def __init__(self, executor=None, map_size=None, margin=10, strict_father=False):
        # Margin size w.r.t the map
        self.margin = margin
        self.recent_count = 0
        super().__init__(executor=executor, map_size=map_size, strict_father=strict_father)

    def kappas_to_road_points(self, kappas, frenet_step=10, theta0=1.57):
        """
        Args:
            kappas: list of kappa values
            frenet_step: The distance between to points.
            theta0: The initial angle of the line. (1.57 == 90 degrees)
        Returns:
            road points in cartesian coordinates
        """
        # Using the bottom center of the map.
        y0 = self.margin
        x0 = self.map_size / 2
        ss = np.arange(y0, (len(kappas) * frenet_step), frenet_step)

        # Transforming the frenet points to cartesian
        (xs, ys) = frenet.frenet_to_cartesian(x0, y0, theta0, ss, kappas)
        road_points = self.reframe_road(xs, ys)
        return road_points

    def execute_frenet_test(self, kappas, method='random', frenet_step=10, theta0=1.57,  parent_info={}, extra_info={}):
        extra_info['kappas'] = kappas
        road_points = self.kappas_to_road_points(kappas, frenet_step=frenet_step, theta0=theta0)
        if road_points:
            self.recent_count += 1
            return self.execute_test(road_points, method=method, parent_info=parent_info, extra_info=extra_info)
        else:
            return 'CANNOT_REFRAME', None

    def reframe_road(self, xs, ys):
        """
        Args:
            xs: cartesian x coordinates
            ys: cartesian y coordinates
        Returns:
            A representation of the road that fits the map size (when possible).
        """
        min_xs = min(xs)
        min_ys = min(ys)
        road_width = 4 # According to the doc
        if (max(xs) - min_xs + road_width > self.map_size - self.margin) \
                or (max(ys) - min_ys + road_width > self.map_size - self.margin):
            log.info("Skip: Road won't fit")

            xs, ys = self.rotate_road(xs, ys)
            if (max(xs) - min(xs) + road_width > self.map_size - self.margin) \
                or (max(ys) - min(ys) + road_width > self.map_size - self.margin):
                log.info("Both failed!")
                return None
            else:
                log.info("I MADE IT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # TODO: Fail the entire test and start over
        xs = list(map(lambda x: x - min_xs + road_width, xs))
        ys = list(map(lambda y: y - min_ys + road_width, ys))
        return list(zip(xs, ys))

    def rotate_road(self, xs, ys):
        # 1. Find diagonal and centroid
        n = len(xs)
        pts = np.random.rand(n,2)
        pts[:,[0]] = xs.reshape([n, 1])
        pts[:,[1]] = ys.reshape([n, 1])
        candidates = pts[spatial.ConvexHull(pts).vertices]
        dist_mat = spatial.distance_matrix(candidates,candidates)
        i,j = np.unravel_index(dist_mat.argmax(),dist_mat.shape)
        p1 = candidates[i].reshape([2,1])
        p2 = candidates[j].reshape([2,1])

        ## find centroid
        P = Polygon([[max(xs), max(ys)], [max(xs), min(ys)], [min(xs), max(ys)], [min(xs), min(ys)]])
        p_center = np.array([P.centroid.x, P.centroid.y]).reshape([2,1])

        # 2. Translate the centroid to the origin point 
        o = np.array([0,0]).reshape([2,1])
        # p_middle = (p1+p2)/2
        p_middle = p_center
        dx = o[0][0] - p_middle[0][0]
        dy = o[1][0] - p_middle[1][0]
        T = np.array([[1,0,dx],[0,1,dy],[0,0,1]])

        # 3. (Rotation) Make a point of the diagonal of the road 
        #    in the diagonal of the map
        p1_t = T.dot(np.vstack((p1,[1])))
        p = np.array([p1_t[0][0], p1_t[1][0]])
        target = np.array([1, 1])
        cosangle = p.dot(target)/np.abs((np.linalg.norm(p) * np.linalg.norm(target)))
        angle = np.arccos(cosangle)
        if p[0] - p[1] < 0:
            angle = np.pi - angle
        R = np.array([[np.cos(angle), -1*np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0,0,1]])

        # 4. Move back
        T_back = np.array([[1,0,100],[0,1,100],[0,0,1]])
        pts_final = T_back.dot(R.dot(T.dot(np.vstack((pts.T,np.ones(n))))))
        xs = pts_final[[0],:].reshape([n])
        ys = pts_final[[1],:].reshape([n])
        return xs, ys