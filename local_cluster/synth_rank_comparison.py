from sys import argv
from Network import *
import local_cluster as lc
import networkx as nx
import numpy as np

"""
Test rank similarity of heat kernel pagerank approximation on synthetic graphs.

Generate 20 random graph resembling networks, one of:
    connected_watts_strogatz_graph(n, k, p)
    barabasi_albert_graph(n, k)
    powerlaw_cluster_graph(n, k, p)
and report the average itersection distance between a real heat kernel pagerank
vector and an approximation using a random vertex (drawn according to degree)
as a seed.

Write results to file.
"""

numk = 10
numgraphs = 10
numtrials = 2

#python synth_rank_comparison.py generator n d p outfile
generator = argv[1]
n = int(argv[2])
d = int(argv[3])
p = float(argv[4])
outfile = argv[5]

eps = 0.1
target_cheeg = 0.05
target_size = 100
target_vol = d*target_size
T = (1./target_cheeg)*np.log( (2*np.sqrt(target_vol))/(1-eps) + 2*eps*target_size )

ks = np.linspace(1, T, num=numk)

isim = []
isim_10 = []
isim_100 = []
eps_err = []
l1_err = []

for k in ks:
    _isim = 0
    _isim10 = 0
    _isim100 = 0
    _eps_err = 0
    _l1_err = 0
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
        start_node = net.random_node()
        print '\tcomputing exact pagerank...'
        exact = net.exp_hkpr(T, start_node, normalized=False)
        ranked_e = sorted(exact, key=exact.get, reverse=True)
        for j in range(numtrials):
            print '\tcomputing approximate pagerank...'
            apprx = lc.approx_hkpr_seed_mp_dict(net, T, start_node, K=k, eps=eps, normalized=False, verbose=True)
            ranked_a = sorted(apprx, key=apprx.get, reverse=True)

            _isim += lc.isim(ranked_e, ranked_a, n)
            _isim10 += lc.isim(ranked_e, ranked_a, 10)
            _isim100 += lc.isim(ranked_e, ranked_a, 100)
            _eps_err += lc.approx_hkpr_err_dict(exact, apprx, eps)
            _l1_err += lc.avg_l1(exact, apprx)
    isim.append( (_isim*1.0)/(numtrials*numgraphs) )
    isim_10.append( (_isim10*1.0)/(numtrials*numgraphs) )
    isim_100.append( (_isim100*1.0)/(numtrials*numgraphs) )
    eps_err.append( (_eps_err*1.0)/(numtrials*numgraphs) )
    l1_err.append( (_l1_err*1.0)/(numtrials*numgraphs) )

print 'results over k:'
print 'k\tisim\tisim10\tisim100\teps\tl1'
for i in range(len(ks)):
    print ks[i], '\t', isim[i], '\t', isim_10[i], '\t', isim_100[i], '\t', eps_err[i], '\t', l1_err[i]
print

f = open(outfile, 'w')
f.write('k\tisim\tisim10\tisim100\teps\tl1\n')
for i in range(len(ks)):
    f.write(str(ks[i])+'\t'+str(isim[i])+'\t'+str(isim_10[i])+'\t'+str(isim_100[i])
            +'\t'+str(eps_err[i])+'\t'+str(l1_err[i])+'\n')
f.close()
