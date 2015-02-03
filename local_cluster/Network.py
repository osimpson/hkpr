import networkx as nx
import numpy as np
import pydot
import operator
import scipy
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import linalg

execfile('/home/olivia/UCSD/projects/datasets/datasets.py')

class Network(object):
    """
    Network object initialized from a gml file, edgelist, or networkx graph.
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

        #matrices
        # try:
        #     self.adj_mat = csc_matrix(nx.to_numpy_matrix(self.graph, nodelist=sorted(self.graph.nodes())))
        # except MemoryError:
        self.adj_mat = lil_matrix((self.size, self.size))
        for node in self.graph.nodes():
            for neighbor in self.graph.neighbors(node):
                self.adj_mat[self.node_to_index[node], self.node_to_index[neighbor]] = 1.0
        d = self.adj_mat.sum(axis=1)
        # self.deg_vec = np.zeros(self.size)
        # self.deg_mat = np.zeros((self.size, self.size))
        self.deg_mat = lil_matrix((self.size, self.size))
        self.deg_mat_inv = lil_matrix((self.size, self.size))
        for n in self.graph.nodes():
            i = self.node_to_index[n]
            # self.deg_vec[i] = d[i]
            self.deg_mat[i,i] = d[i]
            self.deg_mat_inv[i,i] = 1./d[i]
        self.adj_mat = csc_matrix(self.adj_mat)
        self.deg_mat = csc_matrix(self.deg_mat)
        self.deg_mat_inv = csc_matrix(self.deg_mat_inv)


    # def combinatorial_laplacian(self):
    #     return self.deg_mat - self.adj_mat


    # def normalized_laplacian(self):
    #     D_minushalf = np.linalg.inv(self.deg_mat)**(0.5)
    #     L = np.dot(D_minushalf, np.dot(self.combinatorial_laplacian(), D_minushalf))
    #     return L


    def walk_mat(self):
        return self.deg_mat_inv.dot(self.adj_mat)


    def laplace_operator(self):
        return sparse.eye(self.size, format="csc") - self.walk_mat()


    def heat_kernel(self, t):
        """
        Exact computation of H_t = exp{-t(I-P)} with the matrix exponential.
        Returns a numpy matrix.
        """
        # \Delta = I - P
        laplace = self.laplace_operator()

        # heat kernel
        return linalg.expm(-t*laplace)


    def exp_hkpr(self, t, start_node, normalized=False):
        """
        Exact computation of hkpr(t,f) = f^T H_t.

        Parameters:
            t, temperature parameters
            start_node, the seed node
            normalized, if set to true, output vector values normalized by
                node degree

        Output:
            a dictionary of node, vector values
        """
        f = indicator_vector(self, node=start_node)
        heat_kernel = self.heat_kernel(t)
        # hkpr = np.dot(np.transpose(f), heat_kernel)
        hkpr = np.transpose(heat_kernel.transpose().dot(f))

        if normalized:
            return get_normalized_node_vector_values(self, hkpr)
        else:
            return get_node_vector_values(self, hkpr)


    def volume(self, subset=None):
        """
        Compute the volume of the entire graph if no subset is provided,
        or of the subset of nodes.
        """
        if subset is None:
            vol = 2*self.graph.number_of_edges()
        else:
            vol = 0
            for nd in subset:
                vol += self.graph.degree(nd)
        return vol


    def edge_boundary(self, subset):
        """
        Compute the size of the edge boundary of the subset in the network.
        """
        edge_bound = 0
        for nd in subset:
            for v in self.graph.neighbors(nd):
                if v not in subset:
                    edge_bound += 1
        return edge_bound


    def cheeger_ratio(self, subset):
        """
        Compute the Cheeger ratio of the subset in the network.
        Assume the volume of the subset is < vol(network)/2.
        """
        cheeg = float(self.edge_boundary(subset))/self.volume(subset=subset)
        return cheeg


    def random_hop_cluster_hops(self, hops, start_node=None):
        """
        Compute a random cluster in the network based on hops.

        Parameters:
            hops, number of hops from the starting node used to compute the
                    random cluster
            start_node, the seed of the cluster.  If none provided, this is
                    chosen uniformly at random.

        Output:
            a list of nodes
        """
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
        """
        Compute a random cluster of specified size in the network based on hops.

        Parameters:
            size, the desired size of the cluster
            start_node, the seed of the cluster.  If none provided, this is
                    chosen uniformly at random.

        Output:
            a list of nodes
        """
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


    def random_walk_seed(self, k, start_node, verbose=False):
        """
        Outputs the last node visited in a k-step random walk on the graph
        starting from start_node.
        """

        def random_neighbor(node):
            neighbs = self.graph[node].keys()
            d = len(neighbs)
            tmp = np.random.random()
            for j in range(d):
                if tmp < (j+1)*(1./d):
                    return self.graph[node].keys()[j]

        cur_node = start_node
        if verbose:
            print 'start:', cur_node
        for steps in xrange(k):
            # next_node = np.random.choice(self.graph.neighbors(cur_node))
            # next_node = np.random.choice(self.graph[cur_node].keys())
            next_node = random_neighbor(cur_node)
            cur_node = next_node
            if verbose:
                print cur_node
        if verbose:
            print 'stop:', cur_node
        return cur_node


    def random_walk_seed_matmult(self, k, start_node):
        seed_vec = indicator_vector(self, start_node)
        prod = np.dot(seed_vec, self.walk_mat**k)
        dist = np.ravel(prod)

        return np.random.choice(self.graph.nodes(), p=dist)


    def random_node(self):
        # choose start node according to dv/vol(G)
        total = sum(self.deg_vec)
        p = self.deg_vec/total
        cur_node = np.random.choice(self.graph.nodes(), p=p)

    # def random_walk(self, k, seed_vec=None, verbose=False):
    #     """
    #     Outputs the last node visited in a k-step random walk on the graph
    #     using seed_vec as a starting distribution.
    #     """
    #     if seed_vec is not None:
    #         cur_node = draw_node_from_dist(self, seed_vec)
    #     else:
    #         # choose start node according to dv/vol(G)
    #         total = sum(self.deg_vec)
    #         p = self.deg_vec/total
    #         cur_node = np.random.choice(self.graph.nodes(), p=p)
    #     if verbose:
    #         print 'start:', cur_node
    #     for steps in range(k):
    #         next_node = np.random.choice(self.graph.neighbors(cur_node))
    #         cur_node = next_node
    #         if verbose:
    #             print cur_node
    #     if verbose:
    #         print 'stop:', cur_node
    #     return cur_node


    def nxpagerank(self, seed_vec, alpha=0.85, normalized=False):
        """
        Use networkx provided function for computing the pagerank of the
        network.

        Parameters:
            seed_vec, preference vector
            alpha, reset probability, default to 0.85
            normalized, if set to true, output vector values normalized by
                node degree

        Output:
            a dictionary of node, vector values
        """
        #build personalization dict from seed_vec
        pref = {}
        for nd in self.graph.nodes():
            pref[nd] = seed_vec[self.node_to_index[nd]]

        pr = nx.pagerank(self.graph, alpha=alpha, personalization=pref)

        if normalized:
            prvec = np.array(pr.values())
            return get_normalized_node_vector_values(self, prvec)
        else:
            return pr


def indicator_vector(Net, node=None):
    """
    Compute the indicator vector for the given network and node.
    Output as a numpy array.
    """
    if node is None:
        node = np.random.choice(Net.graph.nodes())
    chi = np.zeros(Net.size)
    chi[Net.node_to_index[node]] = 1.0
    return chi


def draw_node_from_dist(Net, dist_vec):
    """
    Draw a random node from the network based on probability distribution
    given by dist_vec.
    """
    indx = np.random.choice(Net.index_to_node.keys(), p=dist_vec)
    node = Net.index_to_node[indx]
    return node


def get_node_vector_values(Net, vec):
    """
    Return a dictionary of node, vector values for list/vector vec.
    """
    vals = {}
    for i in range(vec.size):
        vals[Net.index_to_node[i]] = vec[i]
    return vals

def get_normalized_node_vector_values(Net, vec):
    """
    Return a dictionary of node, vector values for list/vector vec
    and normalize each value by node degree.
    """
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
