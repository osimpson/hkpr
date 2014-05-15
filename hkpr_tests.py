import hkpr
import numpy as np
import math


def avg_L1_error(true, appr):
    delta = true - appr 
    return sum(([math.fabs(x) for x in delta]))/dolphins.size # L1 error

def component_error(true, appr, eps):
    num_nodes = 0
    total_over_err = 0
    for u in range(dolphins.size):
        u_bound = appr[u] - (1+eps)*true[u]
        l_bound = (1-eps)*true[u] - eps - appr[u]
        if u_bound > 0 or l_bound > 0:
            num_nodes += 1 
        if u_bound > 0:
           total_over_err += u_bound
        if l_bound > 0:
           total_over_err += l_bound
    return num_nodes, total_over_err

def test_k(temp, eps, k_max, out_file, num_tests):
    '''
    Test accuracy of approximation algorithm over various values of K
    '''

    f = open(out_file, 'w')
    f.write('Accuracy test over thresholds for length of walks')
    f.write('\nTesting different seed nodes in dolphins network of size 61')

    # choose start node according to dv/vol(G)
    total = sum(dolphins.deg_vec)
    p = dolphins.deg_vec/total
    sn = np.random.choice(dolphins.graph.nodes(),p=p)
    
    f.write('\nseed node: ' + str(sn)) #print 'start node: ', str(sn)
    f.write('\ntemp: ' + str(temp))    #print 'temp: ', temp
    f.write('\neps: ' + str(eps))      #print 'eps: ', eps
    f.write('\n\nmax K'       # maximum walk length
            +'\tavg L1 error' # L1 error/number of nodes
            +'\texcess'       # excess componenent-wise error
            +'\tbad nodes')   # number of nodes that exceeded allowed error
    
    true = dolphins.exp_hkpr(t=temp, start_node=sn)

    # trials of randomized approximation algorithm
    for i in range(num_tests):    
        r, K, nolim = dolphins.approx_hkpr_testK(temp, K=k_max, start_node=sn,
                                                 eps=eps)
        
        f.write('\n'+str(K))
        L1 = avg_L1_error(true, nolim)
        f.write('\t'+str(L1))
        num_nodes, over_err = component_error(true, nolim, eps)
        f.write('\t'+str(over_err))
        f.write('\t'+str(num_nodes))
    
    f.close()


dolphins = hkpr.Network(gml_file='dolphins.gml')

t=15
eps=0.01

test_k(t, eps, None, 'test_nolim_0.01.txt', 5)

K = int((math.log(1.0/eps))/(math.log(math.log(1.0/eps))))

for k in range(K, 2*t, 3):
    test_k(t, eps, k, 'test_'+str(k)+'_0.01.txt', 5)
