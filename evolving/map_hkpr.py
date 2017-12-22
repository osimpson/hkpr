import os
from TemporalNetwork import TemporalNetwork
from optparse import OptionParser


"""
candidate seeds:
node 90: has highest betweenness centrality in flat (undirected) graph, degree 345
node 111: random, degree 33
node 851: random, degree 103
node 121: random, degree 1
"""


SEED_NODES = ['90', '111', '851', '121']
TPARAM = 75.0


def map_exp_hkpr(temporal_network, seed_nodes, output_dir):
    """Compute the HKPR vectors with the set of seed nodes for each day of the
    temporal network

    :param temporal_network:
    :param seed_nodes:
    :return:
    """
    day_ranges = range(0, temporal_network.network_lifespan(), temporal_network.SECONDS_PER_DAY)

    for t in day_ranges:
        print "computing HKPR for static Graph G", str(t)
        try:
            G_static = temporal_network.get_static_graph_at_t(t)
            for s in seed_nodes:
                try:
                    hkpr_exact = G_static.exp_hkpr(temporal_network.t, s)
                    file_name = "seed"+str(s)+"_time"+str(t)+".txt"
                    output_file = os.path.join(output_dir, "exact", file_name)
                    save_vector(hkpr_exact, output_file)
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
    parser.add_option("--resultsdir",
                      dest="resultsDir",
                      action="store",
                      help="directory for saving results",
                      default="results")

    (options, args) = parser.parse_args()

    edge_list = options.edgeList
    tparam = options.tParam
    seed_nodes = options.seedNodes
    results_dir = options.resultsDir

    # read in edge list
    temporal_edge_list = []
    for row in open(edge_list, 'r').readlines():
        source, dest, time = row.strip().split(' ')
        temporal_edge_list.append([source, dest, int(time)])

    G = TemporalNetwork(temporal_edge_list, tparam)

    map_exp_hkpr(G, seed_nodes, results_dir)


if __name__ == "__main__":
    print "this is the log file!"
    run()