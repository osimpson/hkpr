from Network import *
import solver

execfile('/home/olivia/UCSD/projects/datasets/datasets.py')


def test_hkpr(Net, t=15.0, eps=0.01, randfactor=1):
    subset_size = min(int((1./5)*Net.size), 100)
    f.write('\nsubset size: '+str(subset_size))
    subset = Net.random_hop_cluster_size(subset_size)
    fvec = np.random.random(Net.size)*randfactor

    print 'computing true restricted solution...'
    hkpr_true = Net.exp_hkpr(subset, t, fvec)
    print 'computing approximate solution with unlimited walk length...'
    hkpr_unlim = solver.approx_hkpr(Net, subset, t, fvec, K='unlim', eps=eps, verbose=True)
    print 'computing approximate solution with mean walk length...'
    hkpr_mean = solver.approx_hkpr(Net, subset, t, fvec, K='mean', eps=eps, verbose=True)
    print 'computing approximate solution with bounded walk length...'
    hkpr_bound = solver.approx_hkpr(Net, subset, t, fvec, K='bound', eps=eps, verbose=True)

    f.write('\n\nerror in unlimited walk length:')
    f.write('\n\tunit error: '+str(solver.approx_hkpr_err_unit(hkpr_true, hkpr_unlim, eps)))
    f.write('\n\tgeneral error: '+str(solver.approx_hkpr_err(hkpr_true, hkpr_unlim, fvec, eps)))

    f.write('\n\nerror in mean walk length:')
    f.write('\n\tunit error: '+str(solver.approx_hkpr_err_unit(hkpr_true, hkpr_mean, eps)))
    f.write('\n\tgeneral error: '+str(solver.approx_hkpr_err(hkpr_true, hkpr_mean, fvec, eps)))

    f.write('\n\nerror in bounded walk length:')
    f.write('\n\tunit error: '+str(solver.approx_hkpr_err_unit(hkpr_true, hkpr_bound, eps)))
    f.write('\n\tgeneral error: '+str(solver.approx_hkpr_err(hkpr_true, hkpr_bound, fvec, eps)))

# f = open('sampling_tests.txt', 'w')
# f.write('example\n')
# test_network(example)
# f.close()

f = open('hkpr_tests_dolphins.txt', 'w')
print '\ngenerating dolphins network...'
dolphins = Network(GRAPH_DATASETS['dolphins'])
f.write('dolphins\n')
test_hkpr(dolphins)
test_hkpr(dolphins, randfactor=10)
test_hkpr(dolphins, randfactor=100)
f.close()

f = open('hkpr_tests_lesmis.txt', 'w')
print '\ngenerating lesmis network...'
lesmis = Network(GRAPH_DATASETS['lesmis'])
f.write('lesmis\n')
test_hkpr(lesmis)
test_hkpr(lesmis, randfactor=10)
test_hkpr(lesmis, randfactor=100)
f.close()

f = open('hkpr_tests_power.txt', 'w')
print '\ngenerating power network...'
power = Network(GRAPH_DATASETS['power'])
f.write('power\n')
test_hkpr(power)
test_hkpr(power, randfactor=10)
test_hkpr(power, randfactor=100)
f.close()

# f = open('hkpr_tests_facebook.txt', 'w')
# print '\ngenerating facebook network...'
# facebook = Network(edge_list=GRAPH_DATASETS['facebook'])
# f.write('facebook\n')
# test_hkpr(facebook)
# test_hkpr(facebook, randfactor=10)
# test_hkpr(facebook, randfactor=100)
# f.close()
