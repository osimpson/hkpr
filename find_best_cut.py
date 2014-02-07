import numpy as np
import partition
import hkpr
import pickle
import sys

net_name = sys.argv[1] #name for output files e.g. 'dolphins'
net_file = sys.argv[2] #file path for network data e.g. './dolphins.gml'
fformat = sys.argv[3] #file type: gml, nx, or edgelist

if fformat == 'gml':
    net = hkpr.Network(gml_file=net_file)
elif fformat == 'nx':
    net = hkpr.Network(netx=net_file)
elif fformat == 'edgelist':
    net = hkpr.Network(edge_list=net_file)
else:
    sys.exit('unknown file format')
    

for i in range(10):
    # choose start node according to dv/vol(G)
    total = sum(net.deg_vec)
    p = net.deg_vec/total
    start_node = np.random.choice(net.graph.nodes(), p=p)

    v = len(net.graph.edges())/2
    phi = 0.2
    
    (best_set, best_vol, best_cheeg, heat_val_vec) = partition.min_partition_hkpr(net, start_node, v, phi, approx=False, eps=0.1)
    
    f = open(net_name+'_best'+str(i)+'.txt', 'w')
    
    f.write('start node:'+str(start_node)+'\n')
    f.write('true vector results:\n')
    f.write('set: [')
    for s in best_set:
        f.write(str(s)+', ')
    f.write(']\n')
    f.write('volume: '+str(best_vol)+'\n')
    f.write('ratio: '+str(best_cheeg)+'\n')
    f.write('vector values:\n')
    for i in range(len(heat_val_vec)):
        f.write(str(i)+'\t'+str(heat_val_vec[i])+'\n')
    
    # normalize range of vector values to map to a unit interval
    min_v = min(heat_val_vec)
    max_v = max(heat_val_vec)
    norm_v = [(x-min_v)/(max_v-min_v) for x in heat_val_vec]
    
    net.draw_hkpr(norm_v, net_name+'_best_cut_test_'+str(start_node)+'.png')
