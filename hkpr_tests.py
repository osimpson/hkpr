import hkpr
import numpy as np
import math

dolphins = hkpr.Network(gml_file='dolphins.gml')

def approx_error(true, appr):
    delta = true - appr 
    return sum(([math.fabs(x) for x in delta])) # L1 error

def test_k(temp, eps, out_file):
    '''
    Test accuracy of approximation algorithm over various values of K
    '''

    f = open('out_file', 'w')
    f.write('Accuracy test over thresholds for length of walks\n\n')
    f.write('dolphins network\nt = '+str(temp)+'\nepsilon = '+str(eps)+'\n')

    # choose start node according to dv/vol(G)
    total = sum(dolphins.deg_vec)
    p = dolphins.deg_vec/total
    sn = np.random.choice(dolphins.graph.nodes(),p=p)

    f.write('seed node = '+str(sn)+'\n\n')

    true = dolphins.exp_hkpr(t=temp, start_node=sn)
    nolim = dolphins.approx_hkpr_testK(temp, None, start_node=sn, eps=eps)
    error = approx_error(true, nolim)
 
    f.write('K threshold\tL1 error\n\n')
    f.write('no limit\t'+str(error)+'\n')

    K = (math.log(1.0/eps))/(math.log(math.log(1.0/eps)))

    for k in range(K,5):
        appr = dolphins.approx_hkpr_testk(temp, K=k, start_node=sn, eps=eps)
        error = approx_error(true, appr)
        f.write(str(k)+'\t'+str(error)+'\n')

    f.close()
