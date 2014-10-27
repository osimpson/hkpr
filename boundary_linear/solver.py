import networkx as nx
import numpy as np
from scipy.integrate import quad
from Network import *

np.set_printoptions(precision=20)

#####################################################################
### Computing heat kernel
#####################################################################


def approx_hkpr_matmult(Net, t, seed_vec, num_trials=1000, verbose=False):
    """
    Use matrix multiplication to estimate a heat kernel pagerank vector as the
    expected distribution of: fP^k with probability Pois(lam=t).

    Parameters:
        t, temperature parameter
        seed_vec, the vector f
        num_trials, number of independent experiments

    Output:
        a dictionary of node, vector values
    """
    n = Net.size
    k = np.random.poisson(lam=t)
    if verbose: print 'k', k, 'random walk steps'
    appr = np.dot(seed_vec, np.linalg.matrix_power(Net.walk_mat, k))
    for i in range(num_trials-1):
        k = np.random.poisson(lam=t)
        if verbose: print 'k', k, 'random walk steps'
        appr += np.dot(seed_vec, np.linalg.matrix_power(Net.walk_mat, k))
    appr = appr/num_trials
    appr = np.array(appr)[0]

    return appr


# def approx_matmult_rwk(Net, k, seed_vec=None, eps=0.01, verbose=False, normalized=False):
#     """
#     Tool for checking approximating fP^k with random walks of length k.
#     """
#     # initialize 0-vector of size n
#     n = Net.size
#     approxhkpr = np.zeros(n)

#     # r = (16.0/eps**3)*np.log(n)
#     r = (16.0/eps**2)*np.log(n)
#     # r = (16.0/eps)*np.log(n)

#     if verbose:
#         print 'r: ', r
#         print 'k: ', k

#     for i in range(int(r)):
#         v = Net.random_walk(k, seed_vec=seed_vec, verbose=False)
#         approxhkpr[Net.node_to_index[v]] += 1
#     approxhkpr = approxhkpr/r

#     if normalized:
#         return get_normalized_node_vector_values(Net, approxhkpr)
#     else:
#         return get_node_vector_values(Net, approxhkpr)


def approx_hkpr(Net, t, seed_vec, eps=0.01, verbose=False):
    """
    An implementation of the ApproxHKPRseed algorithm using random walks.

    Parameters:
        t, temperature parameter
        start_node, the seed node
        eps, desired error parameter

    Output:
        a dictionary of node, vector values
    """
    #initialize 0-vector of size n
    n = Net.size
    approxhkpr = np.zeros(n)

    #create distribution vectors
    f_plus = np.zeros(n)
    f_minus = np.zeros(n)
    for i in range(n):
        if seed_vec[i] > 0.0:
            f_plus[i] = seed_vec[i]
        elif seed_vec[i] < 0.0:
            f_minus[i] = -seed_vec[i]
    if np.linalg.norm(f_plus, ord=1) > 0:
        f_p = f_plus/np.linalg.norm(f_plus, ord=1)
    else:
        f_p = None
    if np.linalg.norm(f_minus, ord=1) > 0:
        f_m = f_minus/np.linalg.norm(f_minus, ord=1)
    else:
        f_m = None

    r = (16.0/eps**3)*np.log(n)
    # r = (16.0/eps**2)*np.log(n)
    # r = (16.0/eps)*np.log(n)

    K = (np.log(1.0/eps))/(np.log(np.log(1.0/eps)))
    # K = t

    if verbose:
        print 'r: ', r
        print 'expected number of random walk steps: ', t
        print 'K: ', K

    if f_p is not None:
        for i in range(int(r)):
            #positive part
            start_node = draw_node_from_dist(Net, f_p)
            k = np.random.poisson(lam=t)
            k = int(min(k,K))
            v = Net.random_walk_seed(k, start_node, verbose=False)
            approxhkpr[Net.node_to_index[v]] += np.linalg.norm(f_plus, ord=1)
        approxhkpr = approxhkpr/r
    if f_m is not None:
        for i in range(int(r)):
            #negative part
            start_node = draw_node_from_dist(Net, f_m)
            k = np.random.poisson(lam=t)
            k = int(min(k,K))
            v = Net.random_walk_seed(k, start_node, verbose=False)
            approxhkpr[Net.node_to_index[v]] -= np.linalg.norm(f_minus, ord=1)
        approxhkpr = approxhkpr/r

    return approxhkpr


def approx_hkpr_err(true, appr, eps):
    """
    Compute the error according to the definition of component-wise additive
    and multiplicative error for approximate heat kernel pagerank vectors.

    This function outputs the total error beyond what we allow.
    """
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
### Solving linear systems with a boundary condition
#####################################################################


def compute_b1(Net, boundary_vec, subset):
    DS = Net.restricted_mat(Net.deg_mat, subset, subset)
    DS_minushalf = np.linalg.inv(DS)**(0.5)
    boundS = Net.vertex_boundary(subset)
    DboundS = Net.restricted_mat(Net.deg_mat, boundS, boundS)
    DboundS_minushalf = np.linalg.inv(DboundS)**(0.5)
    ASboundS = Net.restricted_mat(Net.adj_mat, subset, boundS)
    _b = [Net.node_to_index[s] for s in boundS]
    bboundS = boundary_vec[_b]

    return np.dot(np.dot(np.dot(DS_minushalf, ASboundS), DboundS_minushalf), bboundS)


def compute_b2(Net, boundary_vec, subset):
    DS = Net.restricted_mat(Net.deg_mat, subset, subset)
    DS_minushalf = np.linalg.inv(DS)**(0.5)
    boundS = Net.vertex_boundary(subset)
    DboundS = Net.restricted_mat(Net.deg_mat, boundS, boundS)
    DboundS_minushalf = np.linalg.inv(DboundS)**(0.5)
    ASboundS = Net.restricted_mat(Net.adj_mat, subset, boundS)
    _b = [Net.node_to_index[s] for s in boundS]
    bboundS = boundary_vec[_b]

    b1 = compute_b1(Net, boundary_vec, subset)
    return np.dot(np.transpose(b1), DS**(0.5))


def restricted_solution(Net, boundary_vec, subset):
    """
    Computes the restricted solution as the matrix vector product:
        xS = (\LS)^{-1} (dot) b1
    as defined in Theorem 1

    Parameters:
        Net, the Network Network (graph)
        boundary_vec, a vector over the nodes of the graph with non-empty support
        subset, a list of nodes in V\supp(boundary_vec)

    Output:
        the restricted solution vector xS over the nodes of the subset
    """
    LS = Net.restricted_mat(Net.normalized_laplacian(), subset, subset)
    LS_inv = np.linalg.inv(LS)
    b1 = compute_b1(Net, boundary_vec, subset)

    return np.dot(LS_inv, b1)


def restricted_solution_riemann(Net, boundary_vec, subset, eps=0.01):
    """
    Computes the restricted solution as the Riemann sum:
        xS = sum_{j=1}^N \H_{jT/N} T/N (dot) b1
    as defined in Lemma 2

    \H_t = exp{-t*(\L)_S}

    Parameters:
        Net, the Network Network (graph)
        boundary_vec, a vector over the nodes of the graph with non-empty support
        subset, a list of nodes in V\supp(boundary_vec)

    Output:
        the restricted solution vector xS over the nodes of the subset
    """
    s = len(subset)
    T = (s**3)*(np.log((s**3)*(1./eps)))
    N = T/eps

    xS = np.zeros((s,1))
    b1 = compute_b1(Net, boundary_vec, subset)
    for j in range(1, int(N)+1):
        xS += eps*np.dot(Net.heat_kernel_symm(subset, j*eps), b1)

    return xS

def err_RSR(Net, boundary_vec, subset, eps=0.01):
    """
    Compute the Riemann sum approximation and output the error beyond what is
    promised in Lemma 2.
    """
    xS_true = restricted_solution(Net, boundary_vec, subset)
    xS_rie = restricted_solution_riemann(Net, boundary_vec, subset, eps=eps)
    b1 = compute_b1(Net, boundary_vec, subset)
    return max(0, np.linalg.norm(xS_true-xS_rie) - eps*(np.linalg.norm(b1)+np.linalg.norm(xS_true)))


def restricted_solution_riemann_sample(Net, boundary_vec, subset, eps=0.01):
    """
    Computes the restricted solution by sampling the Riemann sum:
        xS = sum_{j=1}^N \H_{jT/N} T/N (dot) b1
    as defined in Lemma 3

    \H_t = exp{-t*(\L)_S}

    Parameters:
        Net, the Network Network (graph)
        boundary_vec, a vector over the nodes of the graph with non-empty support
        subset, a list of nodes in V\supp(boundary_vec)

    Output:
        the restricted solution vector xS over the nodes of the subset
    """
    s = len(subset)
    T = (s**3)*(np.log((s**3)*(1./eps)))
    print 'T', T
    N = T/eps
    print 'N', N
    r = eps**(-2)*(np.log(s) + np.log(1/eps))
    print 'r', r

    b1 = compute_b1(Net, boundary_vec, subset)
    xS = np.zeros((s,1))
    for i in range(int(r)):
        j = np.random.randint(int(N))+1
        xS += np.dot(Net.heat_kernel_symm(subset, j*eps), b1)

    return (T/r)*xS

def err_RSRS(Net, boundary_vec, subset, eps=0.01):
    """
    Compute the Riemann sum approximation by sampling and output the error
    beyond what is promised in Theorem 2.
    """
    xS_true = restricted_solution(Net, boundary_vec, subset)
    xS_rie = restricted_solution_riemann(Net, boundary_vec, subset, eps=eps)
    xS_sample = restricted_solution_riemann_sample(Net, boundary_vec, subset, eps=eps)
    b1 = compute_b1(Net, boundary_vec, subset)
    allowable_err = eps*( np.linalg.norm(b1) + np.linalg.norm(xS_true) + np.linalg.norm(xS_rie) )
    return max(0, np.linalg.norm(xS_true - xS_sample) - allowable_err)


def greens_solver_exphkpr(Net, boundary_vec, subset, eps=0.01):
    """
    Computes the restricted solution by sampling the Riemann sum expressed with
    Dirichlet heat kernel pagerank vectors:
        xS = sum_{j=1}^N hkpr_{jT/N, b2} T/N
    as defined in Corollary 2.

    \H_t = exp{-t*(\L)_S}

    Parameters:
        Net, the Network Network (graph)
        boundary_vec, a vector over the nodes of the graph with non-empty support
        subset, a list of nodes in V\supp(boundary_vec)

    Output:
        the restricted solution vector xS over the nodes of the subset
    """
    s = len(subset)
    T = (s**3)*(np.log((s**3)*(1./eps)))
    print 'T', T
    N = T/eps
    print 'N', N
    r = eps**(-2)*(np.log(s) + np.log(1/eps))
    print 'r', r

    b2 = compute_b2(Net, boundary_vec, subset)
    xS = np.zeros((1,s))
    for i in range(int(r)):
        j = np.random.randint(int(N))+1
        xS += Net.exp_hkpr(subset, j*eps, b2)

    DS = Net.restricted_mat(Net.deg_mat, subset, subset)
    DS_minushalf = np.linalg.inv(DS)**(0.5)
    return (T/r)*np.dot(xS, DS_minushalf)

def err_RSRS_exphkpr(Net, boundary_vec, subset, eps=0.01):
    """
    Compute the Riemann sum approximation by sampling and output the error
    beyond what is promised in Theorem 2.
    """
    xS_true = restricted_solution(Net, boundary_vec, subset)
    xS_rie = restricted_solution_riemann(Net, boundary_vec, subset, eps=eps)
    xS_sample = greens_solver_exphkpr(Net, boundary_vec, subset, eps=eps)
    b1 = compute_b1(Net, boundary_vec, subset)
    allowable_err = eps*( np.linalg.norm(b1) + np.linalg.norm(xS_true) + np.linalg.norm(xS_rie) )
    return max(0, np.linalg.norm(xS_true - xS_sample) - allowable_err)


def err_test(Net, boundary_vec, subset, eps=0.01):
    xS_true = restricted_solution(Net, boundary_vec, subset)
    xS_rie = restricted_solution_riemann(Net, boundary_vec, subset, eps=eps)
    xS_sample = greens_solver_exphkpr(Net, boundary_vec, subset, eps=eps)
    b1 = compute_b1(Net, boundary_vec, subset)

    print 'error in reimann method:'
    allowable_err = eps*(np.linalg.norm(b1)+np.linalg.norm(xS_true))
    print max(0, np.linalg.norm(xS_true-xS_rie) - allowable_err)

    print 'error in riemann sampling:'
    allowable_err = eps*( np.linalg.norm(b1) + np.linalg.norm(xS_true) + np.linalg.norm(xS_rie) )
    print max(0, np.linalg.norm(xS_true - xS_sample) - allowable_err)

    print 'error in HKPR sampling:'
    allowable_err = eps*( np.linalg.norm(b1) + np.linalg.norm(xS_true) + np.linalg.norm(xS_rie) )
    print max(0, np.linalg.norm(xS_true - xS_sample) - allowable_err)


def greens_solver(Net, boundary_vec, subset, eps=0.01):
    """
    Full Green's solver algorithm with Dirichlet heat kernel pagerank approximation.

    Parameters:
        Net, the Network Network (graph)
        boundary_vec, a vector over the nodes of the graph with non-empty support
        subset, a list of nodes in V\supp(boundary_vec)

    Output:
        the restricted solution vector xS over the nodes of the subset
    """
    s = len(subset)
    T = (s**3)*(np.log((s**3)*(1./eps)))
    print 'T', T
    N = T/eps
    print 'N', N
    r = eps**(-2)*(np.log(s) + np.log(1/eps))
    print 'r', r

    b2 = compute_b2(Net, boundary_vec, subset)
    xS = np.zeros((1,s))
    for i in range(int(r)):
        j = np.random.randint(int(N))+1
        #####################################
        xS += Net.exp_hkpr(subset, j*eps, b2)
        #####################################

    DS = Net.restricted_mat(Net.deg_mat, subset, subset)
    DS_minushalf = np.linalg.inv(DS)**(0.5)
    return (T/r)*np.dot(xS, DS_minushalf)
