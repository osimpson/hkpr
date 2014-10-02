import Network
import networkx as nx
import numpy as np
import pydot

def approx_hkpr(Net, t, start_node=None, seed_vec=None, eps=0.1,
                verbose=False):
    '''
    Outputs an eps-approximate heat kernel pagerank vector computed
    with random walks.
    '''
    # initialize 0-vector of size n
    approxhkpr = np.zeros(Net.size) 

    # r = (16.0/eps**3)*math.log(n)
    r = (16.0/eps)*math.log(n)
    K = (math.log(1.0/eps))/(math.log(math.log(1.0/eps)))
    # K = t

    if verbose:
        print 'r: ', r
        print 'K: ', K

    for iter in range(int(r)):
        k = np.random.poisson(lam=t)
        k = int(min(k,K))

        v = Net.random_walk(k, start_node=start_node, seed_vec=seed_vec, verbose=False)
        
        approxhkpr[Net.node_to_index[v]] += 1

    return approxhkpr/r