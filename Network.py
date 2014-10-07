import networkx as nx
import numpy as np
import pydot
import operator
import scipy

execfile('/home/olivia/UCSD/projects/datasets/datasets.py')

class Network(object):
    """
    Network object initialized from a gml file or networkx graph.
    Base object for heat kernel pagerank computations.
    """

    def __init__(self, gml_file=None, netx=None, edge_list=None):
        if gml_file:
            self.graph = nx.read_gml(gml_file)
        elif edge_list:
            self.graph = nx.read_edgelist(edge_list)
        elif netx:
            self.graph = netx
        else:
            self.graph = None
        self.size = self.graph.number_of_nodes()

        # node-index dictionaries
        self.node_to_index = {}
        self.index_to_node = {}
        i = 0
        for nd in sorted(self.graph.nodes()):
            self.node_to_index[nd] = i
            self.index_to_node[i] = nd
            i += 1


    def volume(self, subset=None):
        if subset is None:
            vol = 2*self.graph.number_of_edges()
        else:
            vol = 0
            for nd in subset:
                vol += self.graph.degree(nd)
        return vol


    def edge_boundary(self, subset):
        '''
        Outputs the size of the edge boundary of the subset in the network.
        '''
        edge_bound = 0
        for nd in subset:
            for v in self.graph.neighbors(nd):
                if v not in subset:
                    edge_bound += 1
        return edge_bound


    def cheeger_ratio(self, subset):
        cheeg = float(self.edge_boundary(subset))/self.volume(subset=subset)
        return cheeg
   

    def random_hop_cluster_hops(self, hops, start_node=None):
        if start_node is None:
            start_node = np.random.choice(self.graph.nodes())

        cluster = [start_node]
        for i in range(hops):
            add_cluster = []
            for nd in cluster:
                add_cluster.extend(self.graph.neighbors(nd))
            cluster.extend(add_cluster)

        return cluster
    
    def random_hop_cluster_size(self, size, start_node=None):
        if size > self.size:
            print 'cluster size exceeds graph size, returning full graph set'
            return self.graph.nodes()

        if start_node is None:
            start_node = np.random.choice(self.graph.nodes())

        cluster = [start_node]
        while len(cluster) < size:
            add_cluster = []
            for nd in cluster:
                add_cluster.extend(self.graph.neighbors(nd))
            cluster.extend(add_cluster)
            cluster = list(set(cluster))

        return cluster


    def random_walk(self, k, seed_vec=None, verbose=False):
        '''
        Outputs the last node visited in a k-step random walk on the graph.
        If start_node given, walk starts from start_node.
        If seed_vec given, walk starts from a node drawn from seed_vec.
        If neither are given, walk starts from a node drawn from
        p(v) = d(v)/vol(G).
        '''
        if seed_vec is not None:
            cur_node = draw_node_from_dist(self, seed_vec)
        else:
            # choose start node according to dv/vol(G)
            total = sum(self.deg_vec)
            p = self.deg_vec/total
            cur_node = np.random.choice(self.graph.nodes(), p=p)
        if verbose:
            print 'start:', cur_node
        for steps in range(k):
            next_node = np.random.choice(self.graph.neighbors(cur_node))
            cur_node = next_node
            if verbose:
                print cur_node
        if verbose:
            print 'stop:', cur_node
        return cur_node



class Localnetwork(Network):

    def __init__(self, network, subset):
        netxsg = network.graph.subgraph(subset)
        Network.__init__(self, netx=netxsg)
        self.adj_mat = nx.to_numpy_matrix(self.graph, nodelist=sorted(self.graph.nodes()))
        d = np.sum(self.adj_mat, axis=1)
        self.deg_vec = np.zeros(self.size)
        self.deg_mat = np.zeros((self.size, self.size))
        for n in self.graph.nodes():
            i = self.node_to_index[n]
            self.deg_vec[i] = d[i,0]
            self.deg_mat[i,i] = d[i,0]
        # for i in range(self.size):
        #     self.deg_vec[i] = d[i,0]
        #     self.deg_mat[i,i] = d[i,0]
        self.walk_mat = np.linalg.inv(self.deg_mat)*self.adj_mat


    def __init__(self, gml_file=None, netx=None, edge_list=None):
        Network.__init__(self, gml_file=gml_file, netx=netx, edge_list=edge_list)
        self.adj_mat = nx.to_numpy_matrix(self.graph, nodelist=sorted(self.graph.nodes()))
        d = np.sum(self.adj_mat, axis=1)
        self.deg_vec = np.zeros(self.size)
        self.deg_mat = np.zeros((self.size, self.size))
        for n in self.graph.nodes():
            i = self.node_to_index[n]
            self.deg_vec[i] = d[i,0]
            self.deg_mat[i,i] = d[i,0]
        # for i in range(self.size):
        #     self.deg_vec[i] = d[i,0]
        #     self.deg_mat[i,i] = d[i,0]
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


    def exp_hkpr(self, t, seed_vec):
        '''
        Exact computation of hkpr(t,f) = f^T H_t.
        '''
        f = seed_vec
        heat_ker = self.heat_ker(t)
        hkpr = np.dot(np.transpose(f), heat_ker)
        return get_node_vector_values(self, hkpr)


    def random_walk(self, k, seed_vec=None, verbose=False):
        '''
        Outputs the last node visited in a k-step random walk on the graph.
        If start_node given, walk starts from start_node.
        If seed_vec given, walk starts from a node drawn from seed_vec.
        If neither are given, walk starts from a node drawn from
        p(v) = d(v)/vol(G).
        '''
        if seed_vec is not None:
            cur_node = draw_node_from_dist(self, seed_vec)
            # cur_node = np.random.choice(self.graph.nodes(), p=seed_vec)
        else:
            # choose start node according to dv/vol(G)
            total = sum(self.deg_vec)
            p = self.deg_vec/total
            cur_node = np.random.choice(self.graph.nodes(), p=p)
        if verbose:
            print 'start:', cur_node
        for steps in range(k):
            p = np.asarray(self.walk_mat)[self.node_to_index[cur_node]]
            next_node = draw_node_from_dist(self, p)
            # next_node = np.random.choice(self.graph.nodes(), p=p)
            cur_node = next_node
            if verbose:
                print cur_node
        if verbose:
            print 'stop:', cur_node
        return cur_node


    def pagerank(self, seed_vec, alpha=0.85):
        I = np.identity(self.size)
        Z = 0.5*(I + self.walk_mat)
        lhs = I - (1.0-alpha)*Z
        rhs = alpha*seed_vec

        pr = np.linalg.solve(lhs, rhs)
        return get_node_vector_values(self, pr)

    def nxpagerank(self, seed_vec, alpha=0.85):
        #build personalization dict from seed_vec
        pref = {}
        for nd in self.graph.nodes():
            pref[nd] = seed_vec[self.node_to_index[nd]]

        return nx.pagerank(self.graph, alpha=alpha, personalization=pref)


def indicator_vector(Net, node=None):
    if node is None:
        node = np.random.choice(Net.graph.nodes())
    chi = np.zeros(Net.size)
    chi[Net.node_to_index[node]] = 1.0
    return chi


def draw_node_from_dist(Net, dist_vec):
    indx = np.random.choice(Net.index_to_node.keys(), p=dist_vec)
    node = Net.index_to_node[indx]
    return node


def get_node_vector_values(Net, vec):
    vals = {}
    for i in range(vec.size):
        vals[Net.index_to_node[i]] = vec[i]
    return vals

def get_normalized_node_vector_values(Net, vec):
    vals = {}
    for i in range(vec.size):
        node = Net.index_to_node[i]
        vals[node] = vec[i]/Net.graph.degree(node)
    return vals


def draw_vec(self, vec, file_name, label_names=True):

    G = pydot.Dot(graph_type='graph')

    # normalize range of vector values to map to a unit interval
    min_v = min(vec)
    max_v = max(vec)
    norm_v = [(x-min_v)/(max_v-min_v) for x in vec]

    for n in self.graph.nodes():
        if label_names:
            node = pydot.Node(str(n))
        else:
            node = pydot.Node("")
        node.set_style('filled')
        color = 255 - (norm_v[self.node_to_index[n]]/max(norm_v)*255)
        node.set_fillcolor('#ff%02x%02x' % (color,color))

        G.add_node(node)

    for (u,v) in self.graph.edges():
        edge = pydot.Edge(str(u),str(v))
        G.add_edge(edge)

    G.write_png(file_name, prog='neato')

def draw_vec(self, dic, file_name, label_names=True):

    G = pydot.Dot(graph_type='graph')

    # normalize range of vector values to map to a unit interval
    min_v = min(dic.values())
    max_v = max(dic.values())
    norm_d = {}
    for nd in dic:
        norm_d[nd] = (dic[nd]-min_v)/(max_v-min_v)
    
    maxval = max(norm_d.values())
    for n in self.graph.nodes():
        node = pydot.Node(str(n))
        node.set_style('filled')
        node.set_shape('circle')
        if not label_names:
            node.set_label(" ")
        color = 255 - (norm_d[n]/maxval*255)
        node.set_fillcolor('#ff%02x%02x' % (color,color))

        G.add_node(node)

    for (u,v) in self.graph.edges():
        edge = pydot.Edge(str(u),str(v))
        G.add_edge(edge)

    G.write_png(file_name, prog='neato')