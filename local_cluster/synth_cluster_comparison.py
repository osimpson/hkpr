from sys import argv
from Network import *
import local_cluster as lc
import networkx as nx
import numpy as np

"""
Test quality of clusters computed with various vectors on synthetic graphs.

Generate random graph resembling networks, one of:
    connected_watts_strogatz_graph(n, k, p)
    barabasi_albert_graph(n, k)
    powerlaw_cluster_graph(n, k, p)
and report the cheeger ratio computed with 3 different vectors for a number of
different random start nodes.

Write results to file.
"""

numgraphs = 5
numtrials = 5

#python synth_rank_comparison.py generator n d p outfile
generator = argv[1]
n = int(argv[2])
d = int(argv[3])
p = float(argv[4])
target_size = float(argv[5])
outfile = argv[6]

target_cheeg = 0.1
eps = 0.1

target_vol = d*target_size
T = (1./target_cheeg)*np.log( (2*np.sqrt(target_vol))/(1-eps) + 2*eps*target_size )

volume_e = []
volume_a = []
volume_pr = []
ratio_e = []
ratio_a = []
ratio_pr = []

for i in range(numgraphs):
    if generator == 'smallworld':
        print 'generating a connected Watts-Strogatz random graph...'
        netx = nx.connected_watts_strogatz_graph(n, d, p)
    elif generator == 'pref':
        print 'generating a Barabasi-Albert random graph...'
        netx = nx.barabasi_albert_graph(n, d)
    elif generator == 'powercluster':
        print 'generating a Holme-Kim random graph...'
        netx = nx.powerlaw_cluster_graph(n, d, p)
    else:
        print 'generator type not recognized...'
        break

    net = Network(netx=netx)
    for j in range(numtrials):
        start_node = net.random_node()
        sweep_set_exact, _, vol_e, min_cheeg_e = lc.local_cluster_hkpr_mincheeg(net, start_node, target_size=target_size, target_vol=target_vol, target_cheeg=target_cheeg, approx='exact', eps=eps, verbose=True)
        sweep_set_apprx, _, vol_a, min_cheeg_a = lc.local_cluster_hkpr_mincheeg(net, start_node, target_size=target_size, target_vol=target_vol, target_cheeg=target_cheeg, approx='rw', eps=eps, verbose=True)
        sweep_set_pr, _, vol_pr, min_cheeg_pr = lc.local_cluster_pr_mincheeg(net, start_node, target_cheeg=target_cheeg)

        volume_e.append(vol_e)
        volume_a.append(vol_a)
        volume_pr.append(vol_pr)
        ratio_e.append(min_cheeg_e)
        ratio_a.append(min_cheeg_a)
        ratio_pr.append(min_cheeg_pr)

print 'results over trials:'
print 'volume_e\tvolume_a\tvolume_pr\tratio_e\tratio_a\tratio_pr'
for i in range(len(volume_e)):
    print volume_e[i], '\t', volume_a[i], '\t', volume_pr[i], '\t', ratio_e[i], '\t', ratio_a[i], '\t', ratio_pr[i]
print

f = open(outfile, 'w')
f.write('volume_e\tvolume_a\tvolume_pr\tratio_e\tratio_a\tratio_pr\n')
for i in range(len(volume_e)):
    f.write(str(volume_e[i])+'\t'+str(volume_a[i])+'\t'+str(volume_pr[i])+'\t'+
            str(ratio_e[i])+'\t'+str(ratio_a[i])+'\t'+str(ratio_pr[i])+'\n')
f.close()
