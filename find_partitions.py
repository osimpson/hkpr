import numpy as np
import partition
import hkpr
import pickle


net = hkpr.Network(gml_file='dolphins.gml')

for i in range(10):
    # choose start node according to dv/vol(G)
    total = sum(net.deg_vec)
    p = net.deg_vec/total
    start_node = np.random.choice(net.graph.nodes(), p=p)
    
    target_vol = len(net.graph.edges())/2
    print target_vol
    target_chg = 0.03
    
    (set_true, vol_true, cheeg_true, heat_vec_true) = partition.partition_hkpr(net, start_node, target_vol, target_chg)
    
    f = open('dolphins_'+str(i)+'.txt', 'w')
    
    f.write('start node:'+str(start_node)+'\n')
    f.write('true vector results:\n')
    f.write('set: [')
    for s in set_true:
        f.write(str(s)+', ')
    f.write(']\n')
    f.write('volume: '+str(vol_true)+'\n')
    f.write('ratio: '+str(cheeg_true)+'\n')
    f.write('vector values:\n')
    for i in range(len(heat_vec_true)):
        f.write(str(i)+'\t'+str(heat_vec_true[i])+'\n')
    
    # normalize range of vector values to map to a unit interval
    min_v = min(heat_vec_true)
    max_v = max(heat_vec_true)
    norm_v = [(x-min_v)/(max_v-min_v) for x in heat_vec_true]
    
    net.draw_hkpr(norm_v, 'dolphins_cut_test_'+str(start_node)+'.png')
