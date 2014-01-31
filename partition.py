import networkx as nx
import numpy as np
import pydot
import hkpr
import math

def vol(network, subset=None):
    if subset is None:
        vol = np.trace(network.deg_mat)
    else:
        vol = 0
        for node in subset:
            vol += network.graph.degree(node)
    return vol

def edge_bound(network, subset):
    '''
    Outputs the size of the edge boundary of the subset in the network.
    '''
    edge_bound = 0
    for node in subset:
        for v in network.graph.neighbors(node):
            if v not in subset:
                edge_bound += 1
    return edge_bound

def cheeg(network, subset):
    cheeg = float(edge_bound(network, subset))/vol(network, subset=subset)
    return cheeg

def partition_hkpr(network, start_node, target_vol, target_cheeg, approx=False, eps=0.1):
    t = (1/target_cheeg**2)*math.log(4*target_vol*(7*eps/2 + 1)**2)

    if approx:
        heat = network.approx_hkpr(t, start_node=start_node, eps=eps)
    else:
        heat = network.exp_hkpr(t, start_node=start_node)
    
    # perform a sweep
    dn_heat_val = {} #dic of probability per degree for each node
    for node in network.graph.nodes():
        dn_heat_val[node] = heat[network.node_to_index[node]]/network.graph.degree(node)
    rank = sorted(dn_heat_val, key=lambda k: dn_heat_val[k], reverse=True) #node ranking (this is a list of nodes!)
    values = list(dn_heat_val.values()) #XXX we might lose indices here...

    sweep_set = []
    for i in range(network.size):
        sweep_set.append(rank[i])
        vol_ach = vol(network, subset=sweep_set) #volume of sweep set
        if vol_ach > 2*target_vol:
            print 'no cut found'
            break
        cheeg_ach = cheeg(network, sweep_set) #cheeger ratio of sweep set
        if vol_ach >= target_vol/2 and cheeg_ach <= math.sqrt(6)*target_cheeg:
            return sweep_set, vol_ach, cheeg_ach, values 

    return None

def main():
    dolphins = hkpr.Network(gml_file='dolphins.gml')
    start_node = 0
    target_vol = 224 
    target_chg = 0.026785714285714284

    (set_true, vol_true, cheeg_true, heat_true) = partition_hkpr(dolphins, start_node, target_vol, target_chg)
#    (set_appr, vol_appr, cheeg_appr, heat_appr) = partition_hkpr(dolphins, start_node, target_vol, target_chg, approx=True, eps=0.1)

    print 'true vector results:'
    print 'set: ', set_true
    print 'volume: ', vol_true, '(target = ', target_vol, ')'
    print 'ratio: ', cheeg_true, '(target = ', target_chg, ')'
    print heat_true

    dolphins.draw_hkpr(heat_true, 'dolphins_cut_0.png')

if __name__ == 'main':
    main()

main()
