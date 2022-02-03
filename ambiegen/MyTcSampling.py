import numpy as np
from pymoo.model.sampling import Sampling
from ambiegen.Solution import Solution

import ambiegen.config as cf
from ambiegen.road_gen import RoadGen


class MyTcSampling(Sampling):

    '''
    Module to generate the initial population
    '''
    def _do(self, problem, n_samples, **kwargs):
        generator = RoadGen(
            cf.model["map_size"],
            cf.model["min_len"],
            cf.model["max_len"],
            cf.model["min_angle"],
            cf.model["max_angle"],
        )
        X = np.full((n_samples, 1), None, dtype=np.object)

        for i in range(n_samples):
            states = generator.test_case_generate()
            s = Solution()
            s.states = states
            X[i, 0] = s
        return X
