import networkx as nx
import numpy as np
from scipy.integrate import quad
import scipy.misc
import multiprocessing as mp
from Network import *

np.set_printoptions(precision=20)

#####################################################################
### Computing heat kernel
#####################################################################


def approx_dir_hkpr(Net, subset, t, f, eps, verbose=False):
    """
    An implementation of the ApproxHKPR algorithm using random walks.
    Use multiprocessing to launch random walks in parallel

    Parameters:
        Net, [Network] the network object
        subset, [list] the subset over which we compute
        t, [float] temperature parameter
        f, [nparray of shape (1,s)] the seed vector
        eps, [float < 1.0] desired error parameter

    Output:
        epsilon-approximate Dirichlet heat kernel pagerank f_S H_t
    """
    n = Net.size
    s = len(subset)

    _f_ = np.sum(f)
    f_unit = f/_f_
    f_unit = f_unit.reshape(f_unit.shape[1],)

    r = (16.0/eps**3)*np.log(n)
    K = 2*t

    if verbose:
        print 'R: ', r
        print 't: ', t
        print 'K: ', K

    #split up the sampling over all processors and collect in a queue
    collect_samples = mp.Queue()
    num_processes = mp.cpu_count()
    def generate_samples(collect_samples):
        num_samples = int(np.ceil(r/num_processes))
        steps = np.random.poisson(lam=t, size=num_samples)
        if verbose:
            print 'maximum random walk steps', max(steps)
        approxhkpr_samples = np.zeros((1,n))
        for i in xrange(num_samples):
            start_node = draw_node_from_dist(Net, f_unit, subset=subset)
            k = steps[i]
            k = int(min(k,K))
            v = Net.dir_random_walk(k, start_node, subset)
            if v is not None:
                approxhkpr_samples[0][Net.node_to_index[v]] += 1.0

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

    approxhkpr = approxhkpr*(_f_/r)

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
    return max(0, approx_hkpr_err_unit(true, appr, eps) - np.sum(f))


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
    b1 = compute_b1(Net, boundary_vec, subset)
    DS = Net.restricted_mat(Net.deg_mat, subset, subset)
    return np.dot(np.transpose(b1), DS**(0.5))


def restricted_solution(Net, boundary_vec, subset):
    """
    Computes the restricted solution as the matrix vector product:
        xS = (\LS)^{-1} (dot) b1
    as defined in Theorem 1

    Parameters:
        Net, [Network] the network object
        boundary_vec, [nparray of shape (n,1)] a vector over the nodes of the graph with non-empty support
        subset, [list] the subset of nodes over which we compute

    Output:
        the restricted solution vector xS over the nodes of the subset
    """
    LS = Net.restricted_mat(Net.normalized_laplacian(), subset, subset)
    LS_inv = np.linalg.inv(LS)
    b1 = compute_b1(Net, boundary_vec, subset)

    return np.dot(LS_inv, b1)


def restricted_solution_riemann(Net, boundary_vec, subset, gamma, verbose=False):
    """
    Computes the restricted solution as the Riemann sum:
        xS = sum_{j=1}^N \H_{jT/N} T/N (dot) b1
    as defined in Lemma 2

    \H_t = exp{-t*(\L)_S}

    Parameters:
        Net, [Network] the network object
        boundary_vec, [nparray of shape (n,1)] a vector over the nodes of the graph with non-empty support
        subset, [list] the subset of nodes over which we compute
        gamma, [float] the solver error parameter

    Output:
        the restricted solution vector xS over the nodes of the subset
    """
    s = len(subset)
    b1 = compute_b1(Net, boundary_vec, subset)
    T = (s**3)*(np.log((s**3)*(1./gamma)))
    N = T/gamma
    if verbose:
        print 'gamma', gamma
        print 'T', T
        print 'N', N

    xS = np.zeros((s,1))
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

def restricted_solution_riemann_sample(Net, boundary_vec, subset, gamma, verbose=False):
    """
    Computes the restricted solution by sampling the Riemann sum:
        xS = sum_{j=1}^N \H_{jT/N} T/N (dot) b1
    as defined in Lemma 3

    \H_t = exp{-t*(\L)_S}

    Parameters:
        Net, [Network] the network object
        boundary_vec, [nparray of shape (n,1)] a vector over the nodes of the graph with non-empty support
        subset, [list] the subset of nodes over which we compute
        gamma, [float] the solver error parameter

    Output:
        the restricted solution vector xS over the nodes of the subset
    """
    s = len(subset)
    T = (s**3)*(np.log((s**3)*(1./gamma)))
    N = T/gamma
    r = gamma**(-2)*(np.log(s) + np.log(1/gamma))
    if verbose:
        print 'gamma', gamma
        print 'T', T
        print 'N', N
        print 'r', r

    b1 = compute_b1(Net, boundary_vec, subset)
    xS = np.zeros((s,1))
    for i in xrange(int(r)):
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
    xS_sample = restricted_solution_riemann_sample(Net, boundary_vec, subset, gamma)
    b1 = compute_b1(Net, boundary_vec, subset)
    allowable_err = gamma*( np.linalg.norm(b1) + np.linalg.norm(xS_true) + np.linalg.norm(xS_rie) )
    return max(0, np.linalg.norm(xS_true - xS_sample) - allowable_err)


def greens_solver_exphkpr_riemann(Net, boundary_vec, subset, gamma, verbose=False):
    """
    Computes the restricted solution by sampling the Riemann sum expressed with
    Dirichlet heat kernel pagerank vectors:
        xS = sum_{j=1}^N hkpr_{jT/N, b2} T/N
    as defined in Corollary 2.

    \H_t = exp{-t*(\L)_S}

    Parameters:
        Net, [Network] the network object
        boundary_vec, [nparray of shape (n,1)] a vector over the nodes of the graph with non-empty support
        subset, [list] the subset of nodes over which we compute
        gamma, [float] the solver error parameter

    Output:
        the restricted solution vector xS over the nodes of the subset
    """
    s = len(subset)
    T = (s**3)*(np.log((s**3)*(1./gamma)))
    N = T/gamma
    if verbose:
        print 'gamma', gamma
        print 'T', T
        print 'N', N

    b2 = compute_b2(Net, boundary_vec, subset)
    xS = np.zeros((1,s))
    for j in range(1, int(N)+1):
        xS += gamma*Net.exp_hkpr(subset, j*gamma, b2)

    DS = Net.restricted_mat(Net.deg_mat, subset, subset)
    DS_minushalf = np.linalg.inv(DS)**(0.5)
    return np.dot(xS, DS_minushalf)


def greens_solver_exphkpr(Net, boundary_vec, subset, gamma, verbose=False):
    """
    Computes the restricted solution by sampling the Riemann sum expressed with
    Dirichlet heat kernel pagerank vectors:
        xS = sum_{j=1}^N hkpr_{jT/N, b2} T/N
    as defined in Corollary 2.

    \H_t = exp{-t*(\L)_S}

    Parameters:
        Net, [Network] the network object
        boundary_vec, [nparray of shape (n,1)] a vector over the nodes of the graph with non-empty support
        subset, [list] the subset of nodes over which we compute
        gamma, [float] the solver error parameter

    Output:
        the restricted solution vector xS over the nodes of the subset
    """
    s = len(subset)
    T = (s**3)*(np.log((s**3)*(1./gamma)))
    # T_thresh = (s**3)*(np.log(1./eps))
    T_thresh = float('infinity')
    N = T/gamma
    r = gamma**(-2)*(np.log(s) + np.log(1/gamma))
    if verbose:
        print 'gamma', gamma
        print 'T', T
        print 'T_thresh', T_thresh
        print 'N', N
        print 'r', r

    b2 = compute_b2(Net, boundary_vec, subset)
    xS = np.zeros((1,s))
    ts = np.random.randint(1, int(N)+1, size=int(np.ceil(r)))
    pos_samples = 0
    zero_samples = 0
    for i in range(int(r)):
        j = ts[i]
        if j*gamma < T_thresh:
            pos_samples += 1
            xS += Net.exp_hkpr(subset, j*gamma, b2)
        else:
            zero_samples += 1
            pass

    if verbose:
        print pos_samples, 'positive samples'
        print zero_samples, 'zero samples'
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


def greens_solver(Net, boundary_vec, subset, eps, gamma, verbose=False):
    """
    Full Green's solver algorithm with Dirichlet heat kernel pagerank approximation.

    Parameters:
        Net, [Network] the network object
        boundary_vec, [nparray of shape (n,1)] a vector over the nodes of the graph with non-empty support
        subset, [list] the subset of nodes over which we compute
        eps, [float] the heat kernel pagerank approximation error
        gamma, [float] the solver error parameter

    Output:
        [nparray of size (1,s)] the restricted solution vector xS over the nodes of the subset
    """
    s = len(subset)
    T = (s**3)*(np.log((s**3)*(1./gamma)))
    T_thresh = 500
    # T_thresh = (s**3)*(np.log(1./eps))
    # T_thresh = float('infinity')
    N = T/gamma
    r = gamma**(-2)*(np.log(s) + np.log(1/gamma))
    if verbose:
        print 'eps', eps
        print 'gamma', gamma
        print 'T', T
        print 'T_thresh', T_thresh
        print 'N', N
        print 'r', r

    b2 = compute_b2(Net, boundary_vec, subset)
    xS = np.zeros((1,s))
    ts = np.random.randint(1, int(N)+1, size=int(np.ceil(r)))
    for i in xrange(int(r)):
        j = ts[i]
        if j*gamma < T_thresh:
            xS += approx_dir_hkpr(Net, subset, j*gamma, b2, eps)
        #otherwise this vector doesn't contribute
        else:
            pass
    DS = Net.restricted_mat(Net.deg_mat, subset, subset)
    DS_minushalf = np.linalg.inv(DS)**(0.5)
    return (T/r)*np.dot(xS, DS_minushalf)

def err_RSRS_apprhkpr(Net, boundary_vec, subset, eps, gamma):
    """
    Compute the error beyond what is promised in Theorem 2 by the
    Green's Solver algorithm.
    """
    xS_true = restricted_solution(Net, boundary_vec, subset)
    xS_rie = restricted_solution_riemann(Net, boundary_vec, subset, gamma)
    xS_sample = greens_solver(Net, boundary_vec, subset, eps, gamma)
    b1 = compute_b1(Net, boundary_vec, subset)
    b2 = compute_b2(Net, boundary_vec, subset)
    allowable_err = gamma*( np.linalg.norm(b1) + np.linalg.norm(xS_true) +
                            np.linalg.norm(xS_rie) ) + eps*(np.linaglg.norm(b2, ord=1, axis=1))
    return max(0, np.linalg.norm(xS_true-xS_sample) - allowable_err)
