
from pymoo.optimize import minimize

from ambiegen.MyProblem import MyProblem
from ambiegen.MyTcMutation import MyTcMutation
from ambiegen.MyTcCrossOver import MyTcCrossover
from ambiegen.MyDuplicates import MyDuplicateElimination
from ambiegen.MyTcSampling import MyTcSampling
import time
from pymoo.algorithms.nsga2 import NSGA2
import ambiegen.config as cf

def optimize():

    '''
    In this function the algorithm is launched and
    the Pareto optimal solutions are returned
    '''

    algorithm = NSGA2(
        n_offsprings=50,
        pop_size=cf.ga["population"],
        sampling=MyTcSampling(),
        crossover=MyTcCrossover(cf.ga["cross_rate"]),
        mutation=MyTcMutation(cf.ga["mut_rate"]),
        eliminate_duplicates=MyDuplicateElimination(),
    )

    t = int(time.time() * 1000)
    seed = (
        ((t & 0xFF000000) >> 24)
        + ((t & 0x00FF0000) >> 8)
        + ((t & 0x0000FF00) << 8)
        + ((t & 0x000000FF) << 24)
    )

    res = minimize(
        MyProblem(),
        algorithm,
        ("n_gen", cf.ga["n_gen"]),
        seed=seed,
        verbose=False,
        save_history=True,
        eliminate_duplicates=True,
    )

    print("Best solution found: \nF = %s" % (res.F))
    gen = len(res.history) - 1
    test_cases = {}
    i = 0

    while i < len(res.F):
        result = res.history[gen].pop.get("X")[i]

        road_points = result[0].intp_points
        test_cases["tc" + str(i)] = road_points
        i += 1
    return test_cases

