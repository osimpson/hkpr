import numpy as np
import math
from Network import *
import local_cluster as lc

execfile('/home/olivia/UCSD/projects/datasets/datasets.py')


def avg_L1_error(true, appr):
    delta = true - appr
    return sum(([math.fabs(x) for x in delta]))/len(delta) # L1 error

def component_error(true, appr, eps):
    num_nodes = 0
    total_over_err = 0
    for u in range(len(true)):
        u_bound = appr[u] - (1+eps)*true[u]
        l_bound = (1-eps)*true[u] - eps - appr[u]
        if u_bound > 0 or l_bound > 0:
            num_nodes += 1
        if u_bound > 0:
           total_over_err += u_bound
        if l_bound > 0:
           total_over_err += l_bound
    return num_nodes, total_over_err

# def test_k(data_file, temp, eps, k_max, out_file, num_seeds, num_tests):
#     '''
#     Test accuracy of approximation algorithm over various values of K
#     '''

#     net = hkpr.Network(gml_file=data_file)

#     f = open(out_file, 'w')
#     f.write('Accuracy test over thresholds for length of walks')
#     f.write('\nTesting different seed nodes in network of size '+str(net.size))

#     for j in range(num_seeds):

#         # choose start node according to dv/vol(G)
#         total = sum(net.deg_vec)
#         p = net.deg_vec/total
#         sn = np.random.choice(net.graph.nodes(),p=p)

#         f.write('\n\nseed node: ' + str(sn))
#         f.write('\ntemp: ' + str(temp))
#         f.write('\neps: ' + str(eps))
#         f.write('\nmax K: ' + str(k_max))
#         f.write('\n\navg K'       # average number of rw steps taken
#                 +'\tavg L1 error' # L1 error/number of nodes
#                 +'\texcess'       # excess componenent-wise error
#                 +'\tbad nodes')   # number of nodes that exceeded allowed error

#         true = net.exp_hkpr(t=temp, start_node=sn)

#         # trials of randomized approximation algorithm
#         for i in range(num_tests):
#             avg_length = 0

#             r, avg_length, nolim = net.approx_hkpr_testK(temp, K=k_max,
#                                                          start_node=sn, eps=eps)

#             f.write('\n'+str(avg_length))
#             L1 = avg_L1_error(true, nolim)
#             f.write('\t\t'+str(L1))
#             num_nodes, over_err = component_error(true, nolim, eps)
#             f.write('\t'+str(over_err))
#             f.write('\t\t'+str(num_nodes))

#     f.close()

def test_k(Net, t, eps, K, out_file, num_seeds, num_tests):
    '''
    Test accuracy of approximation algorithm over various values of K
    '''
    f = open(out_file, 'w')
    f.write('Accuracy test over thresholds for length of walks')
    f.write('\nTesting different seed nodes in network of size '+str(Net.size))

    for j in range(num_seeds):

        # choose start node according to dv/vol(G)
        total = sum(Net.deg_vec)
        p = Net.deg_vec/total
        start_node = np.random.choice(Net.graph.nodes(),p=p)

        f.write('\n\nseed node: ' + str(start_node))
        f.write('\nt: ' + str(t))
        f.write('\neps: ' + str(eps))
        f.write('\nmax K: ' + str(K))
        f.write('\n\navg L1 error' # L1 error/number of nodes
                +'\texcess'       # excess componenent-wise error
                +'\tbad nodes')   # number of nodes that exceeded allowed error

        seed_vec = indicator_vector(Net, start_node)
        true = Net.exp_hkpr(t, seed_vec)
        true_vec = np.array(true.values())

        # trials of randomized approximation algorithm
        for i in range(num_tests):
            appr = lc.approx_hkpr_seed_mp(Net, t, start_node, K=K, eps=eps)
            appr_vec = np.array(appr.values())
            L1 = avg_L1_error(true_vec, appr_vec)
            f.write('\n'+str(L1))
            num_nodes, over_err = component_error(true_vec, appr_vec, eps)
            f.write('\t'+str(over_err))
            f.write('\t\t'+str(num_nodes))

    f.close()

def compute_K(t, eps):
    num = t*(2+math.log(t)) + math.log(2/eps)
    den = math.log(num)

    return num/den

def compute_K_noT(eps):
    num = math.log(1/eps)
    den = math.log(num)

    return num/den

#file_pre = '/home/olivia/UCSD/projects/data/random_walk_data/power_test_'
# data_file='/home/olivia/UCSD/projects/data/power_data/power.gml'

# dolphins = Localnetwork.fullgraph(GRAPH_DATASETS['dolphins'])
power = Localnetwork.fullgraph(GRAPH_DATASETS['power'])
t=15
eps=0.05
n_seeds=2
n_tests=5

print 'testing for bounded walk length...'
# test_k(dolphins, t, eps, None, 'dolphins_nolim_0.05.txt', n_seeds, n_tests)
test_k(power, t, eps, None, 'power_nolim_0.05.txt', n_seeds, n_tests)

K = int((math.log(1.0/eps))/(math.log(math.log(1.0/eps))))

for k in range(K, 2*t, 4):
    print 'testing for', str(k)
    # test_k(dolphins, t, eps, k, 'dolphins_'+str(k)+'_0.05.txt', n_seeds, n_tests)
    test_k(power, t, eps, k, 'power_'+str(k)+'_0.05.txt', n_seeds, n_tests)

print 'testing for', str(2*t)
# test_k(dolphins, t, eps, 2*t, 'dolphins_'+str(2*t)+'_0.05.txt', n_seeds, n_tests)
test_k(power, t, eps, 2*t, 'power_'+str(2*t)+'_0.05.txt', n_seeds, n_tests)
