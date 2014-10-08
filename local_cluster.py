import networkx as nx
import numpy as np
import pydot
from Network import *

np.set_printoptions(precision=20)

#####################################################################
### Computing heat kernel
#####################################################################


def approx_hkpr_matmult(Net, t, seed_vec=None, eps=0.01, verbose=False, normalized=False):
    """
    Use matrix multiplication to estimate a heat kernel pagerank vector as an
    expected distribution of random walks
    """
    n = Net.size
    r = int((16.0/eps)*np.log(n))
    k = np.random.poisson(lam=t)
    if verbose: print 'k', k, 'random walk steps'
    appr = np.dot(seed_vec, np.linalg.matrix_power(Net.walk_mat, k))
    for i in range(r-1):
        k = np.random.poisson(lam=t)
        if verbose: print 'k', k, 'random walk steps'
        appr += np.dot(seed_vec, np.linalg.matrix_power(Net.walk_mat, k))
    appr = appr/r
    appr = np.array(appr)[0]

    if normalized:
        return get_normalized_node_vector_values(Net, appr)
    else:
        return get_node_vector_values(Net, appr)

def approx_matmult_rwk(Net, k, seed_vec=None, eps=0.01, verbose=False, normalized=False):
    """
    Tool for checking approximating fP^k with random walks of length k
    """
    # initialize 0-vector of size n
    n = Net.size
    approxhkpr = np.zeros(n) 

    # r = (16.0/eps**3)*np.log(n)
    r = (16.0/eps**2)*np.log(n)
    # r = (16.0/eps)*np.log(n)

    if verbose:
        print 'r: ', r
        print 'k: ', k

    for i in range(int(r)):
        v = Net.random_walk(k, seed_vec=seed_vec, verbose=False)
        approxhkpr[Net.node_to_index[v]] += 1
    approxhkpr = approxhkpr/r

    if normalized:
        return get_normalized_node_vector_values(Net, approxhkpr)
    else:
        return get_node_vector_values(Net, approxhkpr)


def approx_hkpr_seed(Net, t, start_node, eps=0.01, verbose=False, normalized=False):
    '''
    Outputs an eps-approximate heat kernel pagerank vector computed
    with random walks.
    '''
    # initialize 0-vector of size n
    n = Net.size
    approxhkpr = np.zeros(n) 

    r = (16.0/eps**3)*np.log(n)
    # r = (16.0/eps**2)*np.log(n)
    # r = (16.0/eps)*np.log(n)

    # K = (np.log(1.0/eps))/(np.log(np.log(1.0/eps)))
    # K = t

    if verbose:
        print 'r: ', r
        # print 'K: ', K

    for i in range(int(r)):
        k = np.random.poisson(lam=t)
        # k = int(min(k,K))
        v = Net.random_walk_seed(k, start_node, verbose=False)
        approxhkpr[Net.node_to_index[v]] += 1
    approxhkpr = approxhkpr/r

    if normalized:
        return get_normalized_node_vector_values(Net, approxhkpr)
    else:
        return get_node_vector_values(Net, approxhkpr)


def approx_hkpr_err(true, appr, eps):
    if true.size != appr.size:
        print 'vector dimensions do not match'
        return
    err = 0
    for i in range(true.size):
        if appr[i] == 0:
            comp_err = appr[i] - eps
        else:
            comp_err = (abs(true[i]-appr[i])) - (eps*true[i])
        if comp_err > 0:
            err += comp_err
    return err


#####################################################################
### Local cluster
#####################################################################


def local_cluster_hkpr(Net, start_node, target_size, target_vol, target_cheeg,
                       approx=False, eps=0.01):

    t = (1./target_cheeg)*np.log( (2*np.sqrt(target_vol))/(1-eps) + 2*eps*target_size )

    if approx == 'matmult':
        dn_heat_val = approx_hkpr_matmult(Net, t, seed_vec=seed_vec, eps=eps, verbose=False, normalized=True)
    elif approx == 'rw':
        dn_heat_val = approx_hkpr_seed(Net, t, start_node, eps=eps, verbose=False, normalized=True)
    else:
        #Net must be a Localnetwork
        dn_heat_val = Net.exp_hkpr(t, seed_vec=seed_vec, normalized=True)
    
    #node ranking (this is a list of nodes!)
    rank = sorted(dn_heat_val, key=lambda k: dn_heat_val[k], reverse=True)

    #perform a sweep
    sweep_set = []
    for j in range(len(rank)):
        sweep_set.append(rank[j])
        vol_ach = Net.volume(subset=sweep_set) #volume of sweep set
        if vol_ach > 2*target_vol:
            print 'NO CUT FOUND'
            return
        cheeg_ach = Net.cheeger_ratio(sweep_set) #cheeger ratio of sweep set
        if vol_ach >= target_vol/2 and cheeg_ach <= np.sqrt(8*target_cheeg):
            sweep_heat_vals = {}
            for nd in Net.graph.nodes():
                if nd in sweep_set:
                    sweep_heat_vals[nd] = dn_heat_val[nd]
                else:
                    sweep_heat_vals[nd] = 0
            return sweep_heat_vals, vol_ach, cheeg_ach

    print 'NO CUT FOUND'
    return None


def local_cluster_hkpr_mincheeg(Net, start_node, target_size=None,
                                target_vol=None, target_cheeg=None,
                                approx=False, eps=0.01):

    if target_size is None:
        target_size = Net.size/2.
    if target_vol is None:
        target_vol = Net.volume()/4.
    if target_cheeg is None:
        target_cheeg = 1./3
    t = (1./target_cheeg)*np.log( (2*np.sqrt(target_vol))/(1-eps) + 2*eps*target_size )

    if approx == 'matmult':
        dn_heat_val = approx_hkpr_matmult(Net, t, seed_vec=seed_vec, eps=eps, verbose=False, normalized=True)
    elif approx == 'rw':
        dn_heat_val = approx_hkpr_seed(Net, t, start_node, eps=eps, verbose=False, normalized=True)
    else:
        #Net must be a Localnetwork
        dn_heat_val = Net.exp_hkpr(t, seed_vec=seed_vec, normalized=True)
    
    #node ranking (this is a list of nodes!)
    rank = sorted(dn_heat_val, key=lambda k: dn_heat_val[k], reverse=True)

    #perform a sweep
    sweep_set = []
    min_sweep = []
    best_vol = 0.0
    min_cheeg = 1.0
    for j in range(len(rank)):
        sweep_set.append(rank[j])
        vol_ach = Net.volume(subset=sweep_set) #volume of sweep set
        if vol_ach > 2*target_vol:
           break 
        cheeg_ach = Net.cheeger_ratio(sweep_set) #cheeger ratio of sweep set
        if cheeg_ach < min_cheeg:
            min_sweep = sweep_set
            best_vol = vol_ach
            min_cheeg = cheeg_ach

    sweep_heat_vals = {}
    for nd in Net.graph.nodes():
        if nd in min_sweep:
            sweep_heat_vals[nd] = dn_heat_val[nd]
        else:
            sweep_heat_vals[nd] = 0
   
    return sweep_heat_vals, best_vol, min_cheeg


def local_cluster_pr(Net, start_node, target_cheeg=None):
    seed_vec = indicator_vector(Net, start_node)
    if target_cheeg is None:
        target_cheeg = 1./3
    alpha = (target_cheeg**2)/(255*np.log(100*np.sqrt(Net.graph.number_of_edges())))
    dn_heat_val = Net.nxpagerank(seed_vec, alpha=alpha, normalized=True)
    
    #node ranking (this is a list of nodes!)
    rank = sorted(dn_heat_val, key=lambda k: dn_heat_val[k], reverse=True)

    #perform a sweep
    sweep_set = []
    for j in range(len(rank)):
        sweep_set.append(rank[j])
        vol_ach = Net.volume(subset=sweep_set) #volume of sweep set
        if vol_ach > (2./3)*Net.volume():
            print 'NO CUT FOUND'
            return
        cheeg_ach = Net.cheeger_ratio(sweep_set) #cheeger ratio of sweep set
        if cheeg_ach <= target_cheeg:
            sweep_heat_vals = {}
            for nd in Net.graph.nodes():
                if nd in sweep_set:
                    sweep_heat_vals[nd] = dn_heat_val[nd]
                else:
                    sweep_heat_vals[nd] = 0
            return sweep_heat_vals, vol_ach, cheeg_ach

    print 'NO CUT FOUND'
    return None


def local_cluster_pr_mincheeg(Net, start_node, target_cheeg=None):
    seed_vec = indicator_vector(Net, start_node)
    if target_cheeg is None:
        target_cheeg = 1./3
    alpha = (target_cheeg**2)/(255*np.log(100*np.sqrt(Net.graph.number_of_edges())))
    dn_heat_val = Net.pagerank(seed_vec, alpha=alpha, normalized=True)
    
    #node ranking (this is a list of nodes!)
    rank = sorted(dn_heat_val, key=lambda k: dn_heat_val[k], reverse=True)

    #perform a sweep
    sweep_set = []
    min_sweep = []
    best_vol = 0.0
    min_cheeg = 1.0
    for j in range(len(rank)):
        sweep_set.append(rank[j])
        vol_ach = Net.volume(subset=sweep_set) #volume of sweep set
        if vol_ach > (2./3)*Net.volume():
           break 
        cheeg_ach = Net.cheeger_ratio(sweep_set) #cheeger ratio of sweep set
        if cheeg_ach < min_cheeg:
            min_sweep = sweep_set
            best_vol = vol_ach
            min_cheeg = cheeg_ach

    sweep_heat_vals = {}
    for nd in Net.graph.nodes():
        if nd in min_sweep:
            sweep_heat_vals[nd] = dn_heat_val[nd]
        else:
            sweep_heat_vals[nd] = 0
   
    return sweep_heat_vals, best_vol, min_cheeg