import networkx as nx
import numpy as np
import pydot
import hkpr
import math

np.set_printoptions(precision=20)

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
        heat = network.approx_hkpr(t, start_node=start_node, eps=eps)[-1]
    else:
        heat = network.exp_hkpr(t, start_node=start_node)
    
    # perform a sweep
    dn_heat_val = {} #dic of probability per degree for each node
    dn_heat_val_vec = np.zeros(network.size) #vector of probability per degree for each node
    for node in network.graph.nodes():
        dn_heat_val[node] = heat[network.node_to_index[node]]/network.graph.degree(node)
        dn_heat_val_vec[network.node_to_index[node]] = dn_heat_val[node]
    rank = sorted(dn_heat_val, key=lambda k: dn_heat_val[k], reverse=True) #node ranking (this is a list of nodes!)

    sweep_set = []
    for i in range(network.size):
        sweep_set.append(rank[i])
        vol_ach = vol(network, subset=sweep_set) #volume of sweep set
        if vol_ach > 2*target_vol:
            print 'no cut found'
            break
        cheeg_ach = cheeg(network, sweep_set) #cheeger ratio of sweep set
        if vol_ach >= target_vol/2 and cheeg_ach <= math.sqrt(6)*target_cheeg:
            return sweep_set, vol_ach, cheeg_ach, dn_heat_val_vec

    return None

def min_partition_hkpr(network, start_node, v, phi, approx=False, eps=0.1):
    '''
    Find a cut near a specified starting vertex of minimum conductance
    ''' 
    t = (2*math.log(v))/phi**2

    if approx:
        heat = network.approx_hkpr(t, start_node=start_node, eps=eps)[-1]
    else:
        heat = network.exp_hkpr(t, start_node=start_node)

    # perform a min sweep
    dn_heat_val = {} #dic of probability per degree for each node
    dn_heat_val_vec = np.zeros(network.size) #vector of probability per degree for each node
    for node in network.graph.nodes():
        dn_heat_val[node] = heat[network.node_to_index[node]]/network.graph.degree(node)
        dn_heat_val_vec[network.node_to_index[node]] = dn_heat_val[node]
    rank = sorted(dn_heat_val, key=lambda k: dn_heat_val[k], reverse=True) #node ranking (this is a list of nodes!)

    sweep_set = []
    min_sweep = []
    best_vol = 0.0
    min_cheeg = 1.0
    for i in range(network.size):
        sweep_set.append(rank[i])
        vol_ach = vol(network, subset=sweep_set) #volume of sweep set
        if vol_ach > 2*v:
            break
        cheeg_ach = cheeg(network, sweep_set) #cheeger ratio of sweep set
        if cheeg_ach < min_cheeg: #best cheeger ratio, perform updates
            min_sweep = sweep_set
            best_vol = vol_ach
            min_cheeg = cheeg_ach
    return sweep_set, vol_ach, cheeg_ach, dn_heat_val_vec

def main():
    dolphins = hkpr.Network(gml_file='dolphins.gml')
    start_node = 0
    target_vol = 224 
    target_chg = 0.026785714285714284

#    # approximate hkpr results
#    (set_appr, vol_appr, cheeg_appr, heat_vec_appr) = partition_hkpr(dolphins, start_node, target_vol, target_chg, approx=True, eps=0.01)
#
#    print 'approx vector results:'
#    print 'set: ', set_appr
#    print 'volume: ', vol_appr, '(target = ', target_vol, ')'
#    print 'ratio: ', cheeg_appr, '(target = ', target_chg, ')'
#    print 'vector values:\n', heat_vec_appr

#    # normalize range of vector values to map to a unit interval
#    min_v = min(heat_vec_appr)
#    max_v = max(heat_vec_appr)
#    norm_v = [(x-min_v)/(max_v-min_v) for x in heat_vec_appr]
#    print 'normalized:\n', norm_v

#    # true hkpr results
#    (set_true, vol_true, cheeg_true, heat_vec_true) = partition_hkpr(dolphins, start_node, target_vol, target_chg)
#    print 'true vector results:'
#    print 'set: ', set_true
#    print 'volume: ', vol_true, '(target = ', target_vol, ')'
#    print 'ratio: ', cheeg_true, '(target = ', target_chg, ')'
#    print 'vector values:\n', heat_vec_true

#    # normalize range of vector values to map to a unit interval
#    min_v = min(heat_vec_true)
#    max_v = max(heat_vec_true)
#    norm_v = [(x-min_v)/(max_v-min_v) for x in heat_vec_true]
#    print 'normalized:\n', norm_v

    # true results with min cheeger ratio
    (best_set, best_vol, best_cheeg, heat_val_vec) = min_partition_hkpr(dolphins, start_node, approx=False, eps=0.1)

    print 'min cut results:'
    print 'set: ', best_set
    print 'volume: ', best_vol
    print 'ratio: ', best_cheeg
    print 'vector values:\n', heat_val_vec

    # normalize range of vector values to map to a unit interval 
    min_v = min(heat_val_vec)
    max_v = max(heat_val_vec)
    norm_v = [(x-min_v)/(max_v-min_v) for x in heat_val_vec]
    print 'normalized:\n', norm_v

    dolphins.draw_hkpr(norm_v, 'dolphins_cut_0.png')

if __name__ == 'main':
    main()
