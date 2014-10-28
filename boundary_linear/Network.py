import networkx as nx
import numpy as np
import pydot
import operator
from scipy import linalg

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

        #node-index dictionaries
        self.node_to_index = {}
        self.index_to_node = {}
        i = 0
        for nd in sorted(self.graph.nodes()):
            self.node_to_index[nd] = i
            self.index_to_node[i] = nd
            i += 1

        #matrices
        self.adj_mat = np.array(nx.to_numpy_matrix(self.graph, nodelist=sorted(self.graph.nodes())))
        d = np.sum(self.adj_mat, axis=1)
        self.deg_vec = np.zeros(self.size)
        self.deg_mat = np.zeros((self.size, self.size))
        for n in self.graph.nodes():
            i = self.node_to_index[n]
            self.deg_vec[i] = d[i]
            self.deg_mat[i,i] = d[i]

    # def volume(self, subset=None):
    #     """
    #     Compute the volume of the entire graph if no subset is provided,
    #     or of the subset of nodes.
    #     """
    #     if subset is None:
    #         vol = 2*self.graph.number_of_edges()
    #     else:
    #         vol = 0
    #         for nd in subset:
    #             vol += self.graph.degree(nd)
    #     return vol


    def combinatorial_laplacian(self):
        return self.deg_mat - self.adj_mat


    def normalized_laplacian(self):
        D_minushalf = np.linalg.inv(self.deg_mat)**(0.5)
        L = np.dot(D_minushalf, np.dot(self.combinatorial_laplacian(), D_minushalf))
        return L


    def walk_mat(self):
        return np.dot(np.linalg.inv(self.deg_mat), self.adj_mat)


    def laplace_operator(self):
        return np.eye(self.size) - self.walk_mat()


    def vertex_boundary(self, subset):
        """
        Compute the vertex boundary of the subset in the network.
        """
        vertex_boundary = []
        for nd in subset:
            for v in self.graph.neighbors(nd):
                if v not in subset:
                    vertex_boundary.append(v)
        return list(set(vertex_boundary))


    def heat_kernel_symm(self, subset, t):
        """
        Exact computation of \H_t = exp{-t(\L)_S} with the matrix exponential.
        Returns a numpy matrix.
        """
        LS = self.restricted_mat(self.normalized_laplacian(), subset, subset)
        return linalg.expm(-t*LS)


    def heat_kernel(self, subset, t):
        """
        Exact computation of H_t = exp{-t(I-P)_S} with the matrix exponential.
        Returns a numpy matrix.
        """
        DeltaS = self.restricted_mat(self.laplace_operator(), subset, subset)
        return linalg.expm(-t*DeltaS)


    def exp_hkpr(self, subset, t, f):
        """
        Exact computation of Dirichlet hkpr(t,f) = f^T H_t.

        Parameters:
            t, temperature parameters
            seed_vec, the preference vector
            normalized, if set to true, output vector values normalized by
                node degree

        Output:
            a dictionary of node, vector values
        """
        heat_kernel = self.heat_kernel(subset, t)
        if f.shape[0] != heat_kernel.shape[1]:
            indx = [self.node_to_index[s] for s in subset]
            f = f[indx]
        hkpr = np.dot(f, heat_kernel)

        return hkpr


    def restricted_mat(self, mat, row_subset, column_subset):
        rwindx = [self.node_to_index[s] for s in row_subset]
        clmindx = [self.node_to_index[s] for s in column_subset]
        return mat[[[s] for s in rwindx], clmindx]


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

        cluster = cluster[:size]

        return cluster


    def random_walk_seed(self, k, start_node, verbose=False):
        """
        Outputs the last node visited in a k-step random walk on the graph
        starting from start_node.
        """
        cur_node = start_node
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


#####################################################################
### matrix functions
#####################################################################


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
