import networkx as nx
import numpy as np
import math
import scipy
from scipy import linalg
import pydot
import operator

class Network(object):
    """
    Network object initialized from a gml file or networkx graph.
    Base object for heat kernel pagerank computations.
    """

    def __init__(self, gml_file=None, netx=None, edge_list=None):
        if gml_file:
            self.graph = nx.read_gml(gml_file)
        elif netx:
            self.graph = netx
        elif edge_list:
            self.graph = nx.read_edgelist(edge_list)
        else:
            self.graph = None
        self.size = self.graph.number_of_nodes()

        # node-index dictionaries
        self.node_to_index = {}
        self.index_to_node = {}
        i = 0
        for node in sorted(self.graph.nodes()):
            self.node_to_index[node] = i
            self.index_to_node[i] = node
            i += 1

    def draw_vec(self, vec, file_name):

        G = pydot.Dot(graph_type='graph')

        for n in self.graph.nodes():
            node = pydot.Node(str(n))
            node.set_style('filled')
            color = 255 - (vec[self.node_to_index[n]]/max(vec)*255)
            node.set_fillcolor('#ff%02x%02x' % (color,color))
            G.add_node(node)

        for (u,v) in self.graph.edges():
            edge = pydot.Edge(str(u),str(v))
            G.add_edge(edge)

        G.write_png(file_name, prog='neato')


class Localnetwork(Network):

    def __init__(self, network, subset):
        netxsg = network.graph.subgraph(subset)
        Network.__init__(self, netx=netxsg)
        self.adj_mat = nx.to_numpy_matrix(self.graph, nodelist=sorted(self.graph.nodes()))
        d = np.sum(self.adj_mat, axis=1)
        self.deg_vec = np.zeros(self.size)
        self.deg_mat = np.zeros((self.size, self.size))
        for i in range(self.size):
            self.deg_vec[i] = d[i,0]
            self.deg_mat[i,i] = d[i,0]
        self.walk_mat = np.linalg.inv(self.deg_mat)*self.adj_mat


    def laplacian_combo(self):
        return self.deg_mat - self.adj_mat


    def heat_ker(self, t):
        '''
        Exact computation of H_t = exp{-t(I-P)} with the matrix exponential.
        '''

        # \Delta = I - P
        laplace = np.eye(self.size) - self.walk_mat

        # heat kernel
        return scipy.linalg.expm(-t*laplace) 


    def exp_hkpr(self, t, start_node=None, seed_vec=None):
        '''
        Exact computation of hkpr(t,f) = f^T H_t.
        '''

        # get seed vector
        if start_node is not None:
            f = np.zeros(self.size)
            f[self.node_to_index[start_node]] = 1.
        elif seed_vec is not None:
            f = seed_vec
        else:
            print 'no seed vector given'
            return
 
        heat_ker = self.heat_ker(t)
        return np.transpose(f).dot(heat_ker)


    ##APPROXIMATIONS

    def random_walk(self, k, start_node=None, seed_vec=None, verbose=False):
        '''
        Outputs the last node visited in a k-step random walk on the graph.
        If start_node given, walk starts from start_node.
        If seed_vec given, walk starts from a node drawn from seed_vec.
        If neither are given, walk starts from a node drawn from
        p(v) = d(v)/vol(G).
        '''

        if start_node is not None:
            cur_node = start_node
        elif seed_vec is not None:
            cur_node = np.random.choice(self.graph.nodes(), p=seed_vec)
        else:
            # choose start node according to dv/vol(G)
            total = sum(self.deg_vec)
            p = self.deg_vec/total
            cur_node = np.random.choice(self.graph.nodes(), p=p)
        if verbose:
            print 'start:', cur_node
        for steps in range(k):
            p = np.asarray(self.walk_mat)[self.node_to_index[cur_node]]
            next_node = np.random.choice(self.graph.nodes(), p=p)
            cur_node = next_node
            if verbose:
                print cur_node
        if verbose:
            print 'stop:', cur_node
        return cur_node

    def approx_hkpr(self, t, start_node=None, seed_vec=None, eps=0.1,
                    verbose=False):
        '''
        Outputs an eps-approximate heat kernel pagerank vector computed
        with random walks.
        '''
        n = self.graph.size()

        # initialize 0-vector of size n
        approxhkpr = np.zeros(self.size) 

        # r = (16.0/eps**3)*math.log(n)
        r = (16.0/eps)*math.log(n)
        # K = (math.log(1.0/eps))/(math.log(math.log(1.0/eps)))
        K = t

        if verbose:
            print 'r: ', r
            print 'K: ', K

        for iter in range(int(r)):
            k = np.random.poisson(lam=t)
            k = int(min(k,K))

            v = self.random_walk(k, start_node=start_node, seed_vec=seed_vec, verbose=False)
            
            approxhkpr[self.node_to_index[v]] += 1

        return approxhkpr/r

