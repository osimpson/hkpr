import numpy as np
import partition
import hkpr
import pickle
import sys

net_name = sys.argv[1] #name for output files e.g. 'dolphins'
net_file = sys.argv[2] #file path for network data e.g. './dolphins.gml'
fformat = sys.argv[3] #file type: gml, nx, edgelist, or pickle
pkl = sys.argv[4] #True for pickling, False for file writing
start_node = int(sys.argv[5])

print 'building network...'
if fformat == 'gml':
    net = hkpr.Network(gml_file=net_file)
elif fformat == 'nx':
    net = hkpr.Network(netx=net_file)
elif fformat == 'edgelist':
    net = hkpr.Network(edge_list=net_file)
elif fformat == 'pickle':
    net = pickle.load(open(net_file,'r'))
else:
    sys.exit('unknown file format')

print 'start node: ', start_node
print start_node in net.graph.nodes()
if start_node not in net.graph.nodes():
    sys.exit('unknown node')
 
v = len(net.graph.edges())/2
phi = 0.2

print 'performing a sweep...'
(best_set, best_vol, best_cheeg, heat_val_vec) = partition.min_partition_hkpr(net, start_node, v, phi, approx=False, eps=0.1)

print 'normalizing for rendering...'
# normalize range of vector values to map to a unit interval
min_v = min(heat_val_vec)
max_v = max(heat_val_vec)
norm_v = [(x-min_v)/(max_v-min_v) for x in heat_val_vec]

if not pkl:    
    f = open(net_name+'_best_'+str(start_node)+'.txt', 'w')
    
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

    f.write('range normalized for drawing:\n')
    for i in range(len(norm_v)):
        f.write(str(i)+'\t'+str(norm_v[i])+'\n')

    f.close()
    
else:
    print 'pickling'
    # order: network, volume, cheeger, heat kernel pagerank vector, vector used for rendering
    f = open(net_name+'_best_'+str(start_node)+'.pck', 'wb')

    pickle.dump(best_vol, f)
    pickle.dump(best_cheeg, f)
    pickle.dump(heat_val_vec, f)
    pickle.dump(norm_v, f)

    f.close()

net.draw_hkpr(norm_v, net_name+'_best_cut_test_'+str(start_node)+'.png')
