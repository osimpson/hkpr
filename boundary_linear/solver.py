import networkx as nx
import numpy as np
from scipy.integrate import quad
import multiprocessing as mp
from Network import *

np.set_printoptions(precision=20)

#####################################################################
### Computing heat kernel
#####################################################################


def approx_hkpr(Net, subset, t, f, K, eps, verbose=False):
    """
    An implementation of the ApproxHKPR algorithm using random walks.

    Parameters:
        subset, [list] the subset over which we compute
        t, [float] temperature parameter
        f, [nparray of shape (1,s)] the seed vector
        eps, [float < 1.0] desired error parameter

    Output:
        Dirichlet heat kernel pagerank f_S H_t
    """
    #initialize 0-vector of size n
    n = Net.size
    approxhkpr = np.zeros((1,n))

    #create distribution vectors
    # f_plus = np.zeros(n)
    # f_minus = np.zeros(n)
    # for i in range(n):
    #     if f[0][i] > 0.0:
    #         f_plus[i] = f[0][i]
    #     elif f[0][i] < 0.0:
    #         f_minus[i] = -f[0][i]
    # if np.sum(f_plus) > 0:
    #     _f_p = np.sum(f_plus)
    #     f_p = f_plus/_f_p
    # else:
    #     f_p = None
    # if np.sum(f_minus) > 0:
    #     _f_m = np.sum(f_minus)
    #     f_m = f_minus/_f_m
    # else:
    #     f_m = None
    _f_ = np.sum(f)
    f_unit = f/_f_
    f_unit = f_unit.reshape(f_unit.shape[1],)

    r = (16.0/eps**3)*np.log(n)
    # r = (16.0/eps**2)*np.log(n)
    # r = (16.0/eps)*np.log(n)

    if K == 'bound':
        K = (np.log(1.0/eps))/(np.log(np.log(1.0/eps)))
    elif K == 'mean':
        K = 2*t
    elif K == 'unlim':
        K = float("infinity")

    if verbose:
        print 'r: ', r
        print 'expected number of random walk steps: ', t
        print 'K: ', K

    # if f_p is not None:
    #     for i in range(int(r)):
    #         #positive part
    #         start_node = draw_node_from_dist(Net, f_p, subset=subset)
    #         k = np.random.poisson(lam=t)
    #         k = int(min(k,K))
    #         v = Net.random_walk_seed(k, start_node, verbose=False)
    #         approxhkpr[0][Net.node_to_index[v]] += _f_p
    #     approxhkpr = approxhkpr/r
    # if f_m is not None:
    #     for i in range(int(r)):
    #         #negative part
    #         start_node = draw_node_from_dist(Net, f_m, subset=subset)
    #         k = np.random.poisson(lam=t)
    #         k = int(min(k,K))
    #         v = Net.random_walk_seed(k, start_node, verbose=False)
    #         approxhkpr[0][Net.node_to_index[v]] -= _f_m
    #     approxhkpr = approxhkpr/r
    for i in range(int(r)):
        start_node = draw_node_from_dist(Net, f_unit, subset=subset)
        k = np.random.poisson(lam=t)
        k = int(min(k,K))
        v = Net.random_walk_seed(k, start_node, verbose=False)
        approxhkpr[0][Net.node_to_index[v]] += _f_
    approxhkpr = approxhkpr/r

    indx = [Net.node_to_index[s] for s in subset]
    return approxhkpr[:,indx]


def approx_hkpr_mp(Net, subset, t, f, eps, K='bound', verbose=False):
    """
    An implementation of the ApproxHKPR algorithm using random walks.
    Use multiprocessing to launch random walks in parallel

    Parameters:
        subset, [list] the subset over which we compute
        t, [float] temperature parameter
        f, [nparray of shape (1,s)] the seed vector
        eps, [float < 1.0] desired error parameter

    Output:
        epsilon-approximate Dirichlet heat kernel pagerank f_S H_t
    """
    n = Net.size

    #create distribution vectors
    # f_plus = np.zeros(n)
    # f_minus = np.zeros(n)
    # for i in range(n):
    #     if f[0][i] > 0.0:
    #         f_plus[i] = f[0][i]
    #     elif f[0][i] < 0.0:
    #         f_minus[i] = -f[0][i]
    # if np.sum(f_plus) > 0:
    #     _f_p = np.sum(f_plus)
    #     f_p = f_plus/_f_p
    # else:
    #     f_p = None
    # if np.sum(f_minus) > 0:
    #     _f_m = np.sum(f_minus)
    #     f_m = f_minus/_f_m
    # else:
    #     f_m = None
    _f_ = np.sum(f)
    f_unit = f/_f_
    f_unit = f_unit.reshape(f_unit.shape[1],)

    r = (16.0/eps**3)*np.log(n)
    # r = (16.0/eps**2)*np.log(n)
    # r = (16.0/eps)*np.log(n)

    if K == 'bound':
        K = (np.log(1.0/eps))/(np.log(np.log(1.0/eps)))
    elif K == 'mean':
        K = 2*t
    elif K == 'unlim':
        K = float("infinity")

    if verbose:
        print 'r: ', r
        print 'expected number of random walk steps: ', t
        print 'K: ', K

    # if f_p is not None:
    #     for i in range(int(r)):
    #         #positive part
    #         start_node = draw_node_from_dist(Net, f_p, subset=subset)
    #         k = np.random.poisson(lam=t)
    #         k = int(min(k,K))
    #         v = Net.random_walk_seed(k, start_node, verbose=False)
    #         approxhkpr[0][Net.node_to_index[v]] += _f_p
    #     approxhkpr = approxhkpr/r
    # if f_m is not None:
    #     for i in range(int(r)):
    #         #negative part
    #         start_node = draw_node_from_dist(Net, f_m, subset=subset)
    #         k = np.random.poisson(lam=t)
    #         k = int(min(k,K))
    #         v = Net.random_walk_seed(k, start_node, verbose=False)
    #         approxhkpr[0][Net.node_to_index[v]] -= _f_m
    #     approxhkpr = approxhkpr/r

    #split up the sampling over all processors and collect in a queue
    collect_samples = mp.Queue()
    num_processes = mp.cpu_count()
    def generate_samples(collect_samples):
        num_samples = int(np.ceil(r/num_processes))
        steps = np.random.poisson(lam=t, size=num_samples)
        print 'maximum random walk steps', max(steps)
        if verbose:
            approxhkpr_samples = np.zeros((1,n))

        for i in xrange(num_samples):
            start_node = draw_node_from_dist(Net, f_unit, subset=subset)
            k = steps[i]
            k = int(min(k,K))
            v = Net.random_walk_seed(k, start_node)
            approxhkpr_samples[0][Net.node_to_index[v]] += _f_
        collect_samples.put(approxhkpr_samples)

    #set up a list of processes
    processes = [mp.Process(target=generate_samples, args=(collect_samples,))
                 for x in range(num_processes)]

    #run processes
    for p in processes:
        p.start()
    #exit completed processes
    for p in processes:
        p.join()

    #get process results from output queue
    cum_samples = [collect_samples.get() for p in processes]
    approxhkpr = sum(cum_samples)
    approxhkpr = approxhkpr/r

    indx = [Net.node_to_index[s] for s in subset]
    return approxhkpr[:,indx]


def approx_hkpr_err_unit(true, appr, eps):
    """
    Compute the error according to the definition of component-wise additive
    and multiplicative error for approximate heat kernel pagerank vectors.

    This function outputs the total error beyond what we allow.
    """
    if true.shape != appr.shape:
        print 'vector dimensions do not match'
        return
    err = 0
    for i in range(true.size):
        if appr[0][i] == 0:
            comp_err = appr[0][i] - eps
        else:
            comp_err = (abs(true[0][i]-appr[0][i])) - (eps*true[0][i])
        if comp_err > 0:
            err += comp_err
    return err

def approx_hkpr_err(true, appr, f, eps):
    return max(0, approx_hkpr_err_unit(true, appr, eps) - np.linalg.norm(f, axis=1))

def approx_hkpr_err_1norm(true, appr, f, eps):
    return max(0, approx_hkpr_err_unit(true, appr, eps) - np.linalg.norm(f,
ord=1, axis=1))


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


def restricted_solution_riemann(Net, boundary_vec, subset, gamma):
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
    T = (s**3)*(np.log((s**3)*(1./gamma)))
    print '\tT = ',T
    N = T/gamma
    print '\tN = ',N

    xS = np.zeros((s,1))
    b1 = compute_b1(Net, boundary_vec, subset)
    for j in range(1, int(N)+1):
        xS += gamma*np.dot(Net.heat_kernel_symm(subset, j*gamma), b1)

    return xS

def err_RSR(Net, boundary_vec, subset, gamma):
    """
    Compute the Riemann sum approximation and output the error beyond what is
    promised in Lemma 2.
    """
    xS_true = restricted_solution(Net, boundary_vec, subset)
    xS_rie = restricted_solution_riemann(Net, boundary_vec, subset, gamma)
    b1 = compute_b1(Net, boundary_vec, subset)
    return max(0, np.linalg.norm(xS_true-xS_rie) - gamma*(np.linalg.norm(b1)+np.linalg.norm(xS_true)))

def restricted_solution_riemann_sample(Net, boundary_vec, subset, gamma):
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
    T = (s**3)*(np.log((s**3)*(1./gamma)))
    print '\tT', T
    N = T/gamma
    print '\tN', N
    r = gamma**(-2)*(np.log(s) + np.log(1/gamma))
    print '\tr', r

    b1 = compute_b1(Net, boundary_vec, subset)
    # _b1_ = np.sum(b1)
    # b1_unit = b1/_b1_
    xS = np.zeros((s,1))
    for i in range(int(r)):
        j = np.random.randint(int(N))+1
        xS += np.dot(Net.heat_kernel_symm(subset, j*gamma), b1)

    return (T/r)*xS

def err_RSRS(Net, boundary_vec, subset, gamma):
    """
    Compute the Riemann sum approximation by sampling and output the error
    beyond what is promised in Theorem 2.
    """
    xS_true = restricted_solution(Net, boundary_vec, subset)
    xS_rie = restricted_solution_riemann(Net, boundary_vec, subset, gamma)
    xS_sample = restricted_solution_riemann_sample(Net, boundary_vec, subset,
gamma)
    b1 = compute_b1(Net, boundary_vec, subset)
    allowable_err = gamma*( np.linalg.norm(b1) + np.linalg.norm(xS_true) + np.linalg.norm(xS_rie) )
    return max(0, np.linalg.norm(xS_true - xS_sample) - allowable_err)


def greens_solver_exphkpr_riemann(Net, boundary_vec, subset, gamma):
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
    T = (s**3)*(np.log((s**3)*(1./gamma)))
    print '\tT', T
    N = T/gamma
    print '\tN', N

    b2 = compute_b2(Net, boundary_vec, subset)
    # _b2_ = np.sum(b2)
    # b2_unit = b2/_b2_
    xS = np.zeros((1,s))
    for j in range(1, int(N)+1):
        # xS += Net.exp_hkpr(subset, j*eps, b2_unit)
        xS += gamma*Net.exp_hkpr(subset, j*gamma, b2)

    DS = Net.restricted_mat(Net.deg_mat, subset, subset)
    DS_minushalf = np.linalg.inv(DS)**(0.5)
    # return np.dot(xS, DS_minushalf)*_b2_
    return np.dot(xS, DS_minushalf)


def greens_solver_exphkpr(Net, boundary_vec, subset, gamma):
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
    T = (s**3)*(np.log((s**3)*(1./gamma)))
    print '\tT', T
    N = T/gamma
    print '\tN', N
    r = gamma**(-2)*(np.log(s) + np.log(1/gamma))
    print '\tr', r

    b2 = compute_b2(Net, boundary_vec, subset)
    # _b2_ = np.sum(b2)
    # b2_unit = b2/_b2_
    xS = np.zeros((1,s))
    for i in range(int(r)):
        j = np.random.randint(int(N))+1
        xS += Net.exp_hkpr(subset, j*gamma, b2)

    DS = Net.restricted_mat(Net.deg_mat, subset, subset)
    DS_minushalf = np.linalg.inv(DS)**(0.5)
    return (T/r)*np.dot(xS, DS_minushalf)

def err_RSRS_exphkpr(Net, boundary_vec, subset, gamma):
    """
    Compute the Riemann sum approximation by sampling and output the error
    beyond what is promised in Theorem 2.
    """
    xS_true = restricted_solution(Net, boundary_vec, subset)
    xS_rie = restricted_solution_riemann(Net, boundary_vec, subset, gamma)
    xS_sample = greens_solver_exphkpr(Net, boundary_vec, subset, gamma)
    b1 = compute_b1(Net, boundary_vec, subset)
    allowable_err = gamma*( np.linalg.norm(b1) + np.linalg.norm(xS_true) + np.linalg.norm(xS_rie) )
    return max(0, np.linalg.norm(xS_true-xS_sample) - allowable_err)


def greens_solver(Net, boundary_vec, subset, eps, gamma):
    """
    Full Green's solver algorithm with Dirichlet heat kernel pagerank approximation.

    Parameters:
        Net, the Network Network (graph)
        boundary_vec, [nparray of size (n,1)] a vector over the nodes of the graph with non-empty support
        subset, [list] a list of nodes in V\supp(boundary_vec)

    Output:
        [nparray of size (1,s)] the restricted solution vector xS over the nodes of the subset
    """
    s = len(subset)
    T = (s**3)*(np.log((s**3)*(1./gamma)))
    print 'T', T
    N = T/gamma
    print 'N', N
    r = gamma**(-2)*(np.log(s) + np.log(1/gamma))
    print 'r', r

    b2 = compute_b2(Net, boundary_vec, subset)
    xS = np.zeros((1,s))
    ts = np.random.randint(1, int(N)+1, size=int(np.ceil(r)))
    for i in xrange(int(r)):
        j = ts[i] 
        xS += approx_hkpr_mp(Net, subset, j*gamma, b2, eps=eps)

    DS = Net.restricted_mat(Net.deg_mat, subset, subset)
    DS_minushalf = np.linalg.inv(DS)**(0.5)
    return (T/r)*np.dot(xS, DS_minushalf)

def err_RSRS_exphkpr(Net, boundary_vec, subset, eps, gamma):
    """
    Compute the error beyond what is promised in Theorem 2 by the
    Green's Solver algorithm.
    """
    xS_true = restricted_solution(Net, boundary_vec, subset)
    xS_rie = restricted_solution_riemann(Net, boundary_vec, subset, gamma)
    xS_sample = greens_solver_exphkpr(Net, boundary_vec, subset, gamma)
    b1 = compute_b1(Net, boundary_vec, subset)
    b2 = compute_b2(Net, boundary_vec, subset)
    allowable_err = gamma*( np.linalg.norm(b1) + np.linalg.norm(xS_true) +
np.linalg.norm(xS_rie) ) + eps*(np.linaglg.norm(b2, ord=1, axis=1))
    return max(0, np.linalg.norm(xS_true-xS_sample) - allowable_err)
