import Network
import networkx as nx
import numpy as np
import pydot
import math


def approx_hkpr_matmult(Net, t, seed_vec=None, eps=0.01, verbose=False):
    """
    Use matrix multiplication to estimate a heat kernel pagerank vector as an
    expected distribution of random walks
    """
    r = (16.0/eps)*math.log(n)
    k = np.random.poisson(lam=t)
    if verbose: print 'k', k, 'random walk steps'
    appr = np.dot(seed_vec, np.linalg.matrix_power(Net.walk_mat, k))
    for i in range(r-1):
        k = np.random.poisson(lam=t)
        if verbose: print 'k', k, 'random walk steps'
        appr += np.dot(seed_vec, np.linalg.matrix_power(Net.walk_mat, k))
    appr = appr/r
    appr = np.array(appr)[0]

    return appr

def approx_hkpr_rwk(Net, k, seed_vec=None, eps=0.01, verbose=False):
    """
    Tool for checking approximating fP^k with random walks of length k
    """
    # initialize 0-vector of size n
    n = Net.size
    approxhkpr = np.zeros(n) 

    # r = (16.0/eps**3)*math.log(n)
    r = (16.0/eps**2)*math.log(n)
    # r = (16.0/eps)*math.log(n)

    if verbose:
        print 'r: ', r
        print 'k: ', k

    for i in range(int(r)):
        v = Net.random_walk(k, seed_vec=seed_vec, verbose=False)
        approxhkpr[Net.node_to_index[v]] += 1

    return approxhkpr/r


def approx_hkpr(Net, t, seed_vec=None, eps=0.01, verbose=False):
    '''
    Outputs an eps-approximate heat kernel pagerank vector computed
    with random walks.
    '''
    # initialize 0-vector of size n
    n = Net.size
    approxhkpr = np.zeros(n) 

    # r = (16.0/eps**3)*math.log(n)
    r = (16.0/eps**2)*math.log(n)
    # r = (16.0/eps)*math.log(n)
    K = (math.log(1.0/eps))/(math.log(math.log(1.0/eps)))
    # K = t

    if verbose:
        print 'r: ', r
        print 'K: ', K

    for i in range(int(r)):
        k = np.random.poisson(lam=t)
        # k = int(min(k,K))

        v = Net.random_walk(k, seed_vec=seed_vec, verbose=False)
        
        approxhkpr[Net.node_to_index[v]] += 1

    return approxhkpr/r


def approx_hkpr_err(true, appr, eps):
    if true.size != appr.size:
        print 'vector dimensions do not match'
        return
    err = 0
    for i in range(true.size):
        comp_err = (abs(true[i]-appr[i])) - (eps*true[i])
        if comp_err > 0:
            err += comp_err
    return err