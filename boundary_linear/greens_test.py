from Network import *
import solver
import cProfile

execfile('/home/olivia/UCSD/projects/datasets/datasets.py')
# execfile('/home/olivia/UCSD/projects/datasets/nxexample.py')


def test_network(Net, eps=0.01, randfactor=1):
    subset_size = min(int((1./5)*Net.size), 100)
    f.write('\nsubset size: '+str(subset_size))
    subset = Net.random_hop_cluster_size(subset_size)
    boundary_vec = np.zeros((Net.size, 1))
    vertex_boundary = Net.vertex_boundary(subset)
    f.write('\nboundary vector mult factor: '+str(randfactor))
    for nd in vertex_boundary:
        boundary_vec[Net.node_to_index[nd]] = np.random.random()*randfactor
    f.write('\n1-norm of boundary vector: '+str(np.sum(boundary_vec)))
    f.write('\n2-norm of boundary vector: '+str(np.linalg.norm(boundary_vec)))

    print 'computing true restricted solution...'
    xS_true = solver.restricted_solution(Net, boundary_vec, subset)
    print 'computing restricted solution with Riemann sum...'
    xS_rie = solver.restricted_solution_riemann(Net, boundary_vec, subset, eps=eps)
    print 'computing restricted solution by sampling the Riemann sum...'
    xS_rieSample = solver.restricted_solution_riemann_sample(Net, boundary_vec, subset, eps=eps)
    print 'computing restricted solution by summing heat kernel pagerank vectors...'
    xS_hkpr = solver.greens_solver_exphkpr_riemann(Net, boundary_vec, subset, eps=eps)
    print 'computing restricted solution by sampling heat kernel pagerank...'
    xS_hkprSample = solver.greens_solver_exphkpr(Net, boundary_vec, subset, eps=eps)
    b1 = solver.compute_b1(Net, boundary_vec, subset)

    f.write('\nerror in riemann method:\t')
    allowable_err = eps*(np.linalg.norm(b1)+np.linalg.norm(xS_true))
    f.write(str(max(0, np.linalg.norm(xS_true-xS_rie) - allowable_err)))

    print '\terror in riemann method:'
    print '\t\t', max(0, np.linalg.norm(xS_true-xS_rie) - allowable_err)

    f.write('\nerror in riemann sampling:\t')
    allowable_err_sample = eps*( np.linalg.norm(b1) + np.linalg.norm(xS_true) + np.linalg.norm(xS_rie) )
    f.write(str(max(0, np.linalg.norm(xS_true-xS_rieSample) - allowable_err_sample)))

    print '\terror in riemann sampling:'
    print '\t\t', max(0, np.linalg.norm(xS_true-xS_rieSample) - allowable_err_sample)

    f.write('\nerror in HKPR summing:\t')
    allowable_err = eps*(np.linalg.norm(b1)+np.linalg.norm(xS_true))
    f.write(str(max(0, np.linalg.norm(xS_true-np.transpose(xS_hkpr)) - allowable_err)))

    print '\terror in HKPR summing:'
    print '\t\t', max(0, np.linalg.norm(xS_true-np.transpose(xS_hkpr)) - allowable_err)

    f.write('\nerror in HKPR sampling:\t')
    f.write(str(max(0, np.linalg.norm(xS_true-np.transpose(xS_hkprSample)) - allowable_err_sample)))

    print '\terror in HKPR sampling:'
    print '\t\t', max(0, np.linalg.norm(xS_true-np.transpose(xS_hkprSample)) - allowable_err_sample)

# f = open('sampling_tests.txt', 'w')
# f.write('example\n')
# test_network(example)
# f.close()

f = open('sampling_tests_dolphins4.txt', 'w')
print '\ngenerating dolphins network...'
dolphins = Network(GRAPH_DATASETS['dolphins'])
f.write('dolphins\n')
test_network(dolphins)
test_network(dolphins, randfactor=10)
test_network(dolphins, randfactor=1000)
f.close()

f = open('sampling_tests_lesmis4.txt', 'w')
print '\ngenerating lesmis network...'
lesmis = Network(GRAPH_DATASETS['lesmis'])
f.write('lesmis\n')
test_network(lesmis)
test_network(lesmis, randfactor=10)
test_network(lesmis, randfactor=1000)
f.close()

f = open('sampling_tests_power4.txt', 'w')
print '\ngenerating power network...'
power = Network(GRAPH_DATASETS['power'])
f.write('power\n')
test_network(power)
test_network(power, randfactor=10)
test_network(power, randfactor=1000)
f.close()

f = open('sampling_tests_facebook4.txt', 'w')
print '\ngenerating facebook network...'
facebook = Network(edge_list=GRAPH_DATASETS['facebook'])
f.write('facebook\n')
test_network(facebook)
test_network(facebook, randfactor=10)
test_network(facebook, randfactor=100)
f.close()
