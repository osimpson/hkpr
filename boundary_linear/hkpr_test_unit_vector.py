"""
Test hkpr approximation with varying walk lengths for unit vectors, and
different values of t.
"""

from Network import *
import solver

execfile('/home/olivia/UCSD/projects/datasets/datasets.py')


def test_hkpr(Net, t, eps=0.01):
    subset_size = min(int((1./5)*Net.size), 100)
    subset = Net.random_hop_cluster_size(subset_size)

    f.write('\n\nt='+str(t))
    f.write('\nsubset size: '+str(subset_size))
    #unit vector
    fvec = np.random.random((1,Net.size))
    fvec_unit = fvec/np.linalg.norm(fvec, ord=1, axis=1)
    f.write('\nseed vector of 1-norm: '+str(np.linalg.norm(fvec,ord=1)))

    print 'computing true restricted solution...'
    hkpr_true = Net.exp_hkpr(subset, t, fvec_unit)
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

f = open('hkpr_tests_dolphins_unit_overt.txt', 'w')
print '\ngenerating dolphins network...'
dolphins = Network(GRAPH_DATASETS['dolphins'])
f.write('dolphins\n')
subset_size = min(int((1./5)*dolphins.size), 100)
t = (subset_size**3)*(np.log((subset_size**3)*(1./eps)))
test_hkpr(dolphins, t)
t = t/2
test_hkpr(dolphins, t)
t = t/2
test_hkpr(dolphins, t)
t = t/2
test_hkpr(dolphins, t)
f.close()

f = open('hkpr_tests_lesmis_unit_overt.txt', 'w')
print '\ngenerating lesmis network...'
lesmis = Network(GRAPH_DATASETS['lesmis'])
f.write('lesmis\n')
subset_size = min(int((1./5)*lesmis.size), 100)
t = (subset_size**3)*(np.log((subset_size**3)*(1./eps)))
test_hkpr(lesmis, t)
t = t/2
test_hkpr(lesmis, t)
t = t/2
test_hkpr(lesmis, t)
t = t/2
test_hkpr(lesmis, t)
f.close()

f = open('hkpr_tests_power_unit_overt.txt', 'w')
print '\ngenerating power network...'
power = Network(GRAPH_DATASETS['power'])
f.write('power\n')
subset_size = min(int((1./5)*power.size), 100)
t = (subset_size**3)*(np.log((subset_size**3)*(1./eps)))
test_hkpr(power, t)
t = t/2
test_hkpr(power, t)
t = t/2
test_hkpr(power, t)
t = t/2
test_hkpr(power, t)
f.close()

f = open('hkpr_tests_facebook_unit_overt.txt', 'w')
print '\ngenerating facebook network...'
facebook = Network(edge_list=GRAPH_DATASETS['facebook'])
f.write('facebook\n')
subset_size = min(int((1./5)*facebook.size), 100)
t = (subset_size**3)*(np.log((subset_size**3)*(1./eps)))
test_hkpr(facebook, t)
t = t/2
test_hkpr(facebook, t)
t = t/2
test_hkpr(facebook, t)
t = t/2
test_hkpr(facebook, t)
f.close()
