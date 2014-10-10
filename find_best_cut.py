import numpy as np
import local_cluster as lc
from Network import *
import pickle
from optparse import OptionParser

execfile('/home/olivia/UCSD/projects/datasets/datasets.py')

parser = OptionParser()
parser.add_option("-f", "--fformat", dest="fformat", action="store",
                  help="file format of dataset")
parser.add_option("-d", "--dataset", dest="dataset", action="store",
                  help="dataset")
parser.add_option("-a", "--approx", dest="approx", action="store", type="string", default=False,
                  help="type of vector approximation")
parser.add_option("-s", "--startnode", dest="start_node", action="store", default=None,
                  help="start node if specified")
parser.add_option("-r", "--iters", dest="r", action="store", type=int, default=10,
                  help="number of random iterations")
parser.add_option("-o", "--outfile", dest="outfile", action="store")
parser.add_option("-p", "--pngout", dest="pngout", action="store", default=None)

(options, args) = parser.parse_args()

if options.approx is False:
    if options.fformat == 'gml':
        Net = Localnetwork(gml_file=GRAPH_DATASETS[options.dataset])
    elif options.fformat == 'nx':
        Net = Localnetwork(netx=GRAPH_DATASETS[options.dataset])
    elif options.fformat == 'edgelist':
        Net = Localnetwork(edge_list=GRAPH_DATASETS[options.dataset])
    else:
        sys.exit('unknown file format')
else:
    if options.fformat == 'gml':
        Net = Network(gml_file=GRAPH_DATASETS[options.dataset])
    elif options.fformat == 'nx':
        Net = Network(netx=GRAPH_DATASETS[options.dataset])
    elif options.fformat == 'edgelist':
        Net = Network(edge_list=GRAPH_DATASETS[options.dataset])
    else:
        sys.exit('unknown file format')


f = open(options.outfile, 'w')

best_start = None
best_cheeg = 1.0
best_vol = 0.0
for i in range(options.r):
    start_node = options.start_node
    if start_node is None:
        start_node = np.random.choice(Net.graph.nodes())
        # # choose start node according to dv/vol(G)
        # total = Net.volume()
        # p = Net.deg_vec/total
        # start_node = np.random.choice(Net.graph.nodes(), p=p)
    print 'start node: ', start_node
    # try:
    #     seed = indicator_vector(Net, start_node)
    # except KeyError:
    #     seed = indicator_vector(Net, int(start_node))

    print 'performing a sweep...'
    heat_vals, vol, cheeg = lc.local_cluster_hkpr_mincheeg(Net, start_node, approx=options.approx)

    if cheeg < best_cheeg:
        best_cheeg = cheeg
        best_vol = vol
        best_start = start_node

    if options.pngout is not None:
        print 'rendering...'
        draw_vec(Net, heat_vals, options.pngout+str(start_node)+'.png')

    f.write('start node:'+str(start_node)+'\n')
    # f.write('true vector results:\n')

    # f.write('set: [')
    # for s in best_set:
        # f.write(str(s)+', ')
    # f.write(']\n')

    f.write('volume: '+str(vol)+'\n')

    f.write('ratio: '+str(cheeg)+'\n\n')

    # f.write('vector values:\n')
    # for i in range(len(heat_val_vec)):
        # f.write(str(i)+'\t'+str(heat_val_vec[i])+'\n')

print '\nbest seed node:', best_start
print 'cheeger ratio:', best_cheeg
print 'cluster volume:', best_vol

f.write('best seed node: '+str(best_start))
f.write(' cheeger ratio: '+str(best_cheeg))
f.write(' cluster volume: '+str(best_vol))
f.close()

    # print 'pickling...'
    # f = open(net_name+'_best_'+str(start_node)+'.pck', 'wb')

    # pickle.dump(best_vol, f)
    # pickle.dump(best_cheeg, f)
    # pickle.dump(heat_val_vec, f)
    # pickle.dump(norm_v, f)
