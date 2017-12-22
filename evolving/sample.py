import os
import numpy as np
from TemporalNetwork import TemporalNetwork
from optparse import OptionParser

"""
T = G.network_lifespan() #69459254
tj - ti >= tau
|Si(v) - Sj(v)| > delta
n = G.number_of_nodes()
eps = 0.01

seed 90, time 1900800, node 214: 0.0186072750794
seed 90, time 1987200, node 214: 0.00811857733038

delta = 0.01
tau = T.SECONDS_PER_DAY
"""

SEED_NODES = ['90', '111', '851', '121']
TPARAM = 75.0
EPS = 0.01
DELTA = 0.01


def map_approx_hkpr(G, start_node, eps, tau, delta, results_dir):
    """Simulate the algorithm
    :param G: temporal graph
    :param start_node:
    :param eps:
    :param tau:
    :param delta:
    :param results_dir:
    :return:
    """
    # sample times
    T = G.network_lifespan()
    n = G.number_of_nodes()
    R1 = (3*np.log(n))/(delta*(eps**2))
    R2 = (2*T*np.log(1.0/eps))/(tau*(eps**2))

    print("Sampling R1 =", str(R1), "random walkers in R2 =", str(R2), "times!")

    sampled_times = np.random.choice(T, int(np.floor(R2)))  # with replacement
    for t in sampled_times:
        try:
            Gt = G.get_static_graph_at_t(t)
            try:
                scores = Gt.approx_hkpr(t, start_node, R1)
                file_name = "seed" + str(s) + "_time" + str(t) + ".txt"
                output_file = os.path.join(results_dir, "approx", file_name)
                save_vector(scores, output_file)
            except(KeyError):
                print "Node", s, "is not present in graph G", str(t)
        except:
            print "G", str(t), "is empty..."


def save_vector(vec, output_file):
    fo = open(output_file, 'w')
    for k, v in vec.items():
        fo.write(k + '\t' + str(v) + '\n')
    fo.close()


def run():
    parser = OptionParser()
    parser.add_option("--edgelist",
                      dest="edgeList",
                      action="store",
                      help="Edgelist file",
                      default=None)
    parser.add_option("--tparam",
                      dest="tParam",
                      help="temperature parameter",
                      default=TPARAM)
    parser.add_option("--seednodes",
                      dest="seedNodes",
                      action="store",
                      default=SEED_NODES)
    parser.add_option("--eps",
                      dest="eps",
                      help="approximation parameter",
                      default=EPS)
    parser.add_option("--tau",
                      dest="tau",
                      help="window size parameter",
                      default=None)
    parser.add_option("--delta",
                      dest="delta",
                      help="score change parameter",
                      default=DELTA)
    parser.add_option("--resultsdir",
                      dest="resultsDir",
                      action="store",
                      help="directory for saving results",
                      default="results")

    (options, args) = parser.parse_args()

    edge_list = options.edgeList
    tparam = options.tParam
    seed_nodes = options.seedNodes
    eps = options.eps
    tau = options.tau
    delta = options.delta
    results_dir = options.resultsDir

    # read in edge list
    temporal_edge_list = []
    for row in open(edge_list, 'r').readlines():
        source, dest, time = row.strip().split(' ')
        temporal_edge_list.append([source, dest, int(time)])

    G = TemporalNetwork(temporal_edge_list, tparam)

    if tau is None:
        tau = G.SECONDS_PER_DAY

    # for s in seed_nodes:
    for s in [seed_nodes[0]]:
        map_approx_hkpr(G, s, eps, tau, delta, results_dir)


if __name__ == "__main__":
    run()
