execfile('Network.py')
import local_cluster as lc

f = open('gowalla_pagerank_best_cut.txt', 'w')

gowalla = Network(edge_list=GRAPH_DATASETS['gowalla'])
prval, prvol, prcheeg = lc.local_cluster_pr_mincheeg(gowalla, '307')

f.write('gowalla pagerank\n')
f.write('start node: 307\n')
f.write('volume: '+str(prvol)+'\n')
f.write('ratio: '+str(prcheeg)+'\n')

f.close()

f = open('dblp_best_cut.txt', 'w')

dblp = Network(edge_list=GRAPH_DATASETS['dblp'])
hkval, hkvol, hkcheeg = lc.local_cluster_hkpr_mincheeg(dblp, '38868', approx='rw')

f.write('dblp heat kernel\n')
f.write('start node: 38868\n')
f.write('volume: '+str(hkvol)+'\n')
f.write('ratio: '+str(hkcheeg)+'\n')

prval, prvol, prcheeg = lc.local_cluster_pr_mincheeg(dblp, '38868')

f.write('dblp pagerank\n')
f.write('start node: 38868\n')
f.write('volume: '+str(prvol)+'\n')
f.write('ratio: '+str(prcheeg)+'\n')

f.close()

f = open('webUND_best_cut.txt', 'w')

webUND = Network(edge_list=GRAPH_DATASETS['webUND'])
hkval, hkvol, hkcheeg = lc.local_cluster_hkpr_mincheeg(web, '12129', approx='rw')

f.write('web heat kernel\n')
f.write('start node: 12129\n')
f.write('volume: '+str(hkvol)+'\n')
f.write('ratio: '+str(hkcheeg)+'\n')

prval, prvol, prcheeg = lc.local_cluster_pr_mincheeg(web, '12129')

f.write('dblp pagerank\n')
f.write('start node: 38868\n')
f.write('volume: '+str(prvol)+'\n')
f.write('ratio: '+str(prcheeg)+'\n')

f.close()
