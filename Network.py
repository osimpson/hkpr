import networkx as nx
import numpy as np
import pydot
import operator
import scipy

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

    
    def random_hop_cluster(self, hops, node=None):
        if node is None:
            node = np.random.choice(self.graph.nodes())

        cluster = [node]
        for i in range(hops):
            add_cluster = []
            for n in cluster:
                add_cluster.extend(self.graph.neighbors(n))
            cluster.extend(add_cluster)

        return cluster

    def random_hop_cluster_size(self, size, node=None):
        if node is None:
            node = np.random.choice(self.graph.nodes())

        cluster = [node]
        while len(cluster) < size:
            add_cluster = []
            for n in cluster:
                add_cluster.extend(self.graph.neighbors(n))
            cluster.extend(add_cluster)

        return cluster


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


    def exp_hkpr(self, t, seed_vec=None):
        '''
        Exact computation of hkpr(t,f) = f^T H_t.
        '''

        # get seed vector
        if seed_vec is not None:
            f = seed_vec
        else:
            print 'no seed vector given'
            return
 
        heat_ker = self.heat_ker(t)
        return np.transpose(f).dot(heat_ker)


    ##APPROXIMATIONS

    def random_walk(self, k, seed_vec=None, verbose=False):
        '''
        Outputs the last node visited in a k-step random walk on the graph.
        If start_node given, walk starts from start_node.
        If seed_vec given, walk starts from a node drawn from seed_vec.
        If neither are given, walk starts from a node drawn from
        p(v) = d(v)/vol(G).
        '''
        if seed_vec is not None:
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


def indicator_vector(Net, node):
    chi = np.zeros(Net.size)
    chi[Net.node_to_index[node]] = 1.0
    return chi
