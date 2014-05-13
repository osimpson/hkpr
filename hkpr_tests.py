import hkpr
import numpy as np

dolphins = hkpr.Network(gml_file='dolphins.gml')

def test_k(temp):
    '''
    Test accuracy of approximation algorithm over various values of K
    '''

    # choose start node according to dv/vol(G)
    total = sum(dolphins.deg_vec)
    p = dolphins.deg_vec/total
    sn = np.random.choice(dolphins.graph.nodes(),p=p)

    true = dolphins.exp_hkpr(t=temp, start_node=sn)
