import networkx as nx
import numpy as np
import math
import random
import scipy
from scipy import linalg
import pydot
import operator

class Network(object):
    """
    Network object initialized from a gml file or networkx graph.
    Base object for heat kernel pagerank computations.
    """

    def __init__(self, gml_file=None, netx=None):
        if gml_file:
            self.graph = nx.read_gml(gml_file)
        elif netx:
            self.graph = netx
        else:
            self.graph = None
        self.size = self.graph.number_of_nodes()
        self.adj_mat = nx.to_numpy_matrix(self.graph)
        d = np.sum(self.adj_mat, axis=1)
        self.deg_vec = np.zeros(self.size)
        self.deg_mat = np.zeros((self.size, self.size))
        for i in range(self.size):
            self.deg_vec[i] = d[i,0]
            self.deg_mat[i,i] = d[i,0]
        self.walk_mat = np.linalg.inv(self.deg_mat)*self.adj_mat

        # node-index dictionaries
        self.node_to_index = {}
        self.index_to_node = {}
        i = 0
        for node in self.graph.nodes():
            self.node_to_index[node] = i
            self.index_to_node[i] = node
            i += 1


    ##BUILDING RESTRICTED MATRICES

    def subset_deg_mat(self, subset):
        '''
        D_S = degree matrix restricted to S 
        '''

        s_ix = [self.node_to_index[node] for node in subset]

        sub_deg_mat = self.deg_mat[np.ix_(s_ix, s_ix)]
        return sub_deg_mat

    def boundary_deg_mat(self, subset):
        '''
        D_bS = degree matrix restricted to the boundary of S
        '''

        v_bound = [] # vertex boundary of subset
        for v in subset:
            Nv = self.graph.neighbors(v)
            v_bound.extend([u for u in Nv if u not in subset and u not in v_bound])
        v_bound_ix = [self.node_to_index[node] for node in v_bound]

        boundary_deg_mat = self.deg_mat[np.ix_(v_bound_ix, v_bound_ix)]
        return boundary_deg_mat       

    def subset_adj_mat(self, subset):
        '''
        A_S = adjacency matrix restricted to S 
        '''

        s_ix = [self.node_to_index[node] for node in subset]

        sub_adj_mat = self.adj_mat[np.ix_(s_ix, s_ix)]
        return sub_adj_mat

    def sub_bound_adj_mat(self, subset):
        '''
        A_bS = adjacency matrix with columns of rows corresponding to S
               and columns corresponding to the vertex boundary of S in G
        '''

        v_bound = [] # vertex boundary of subset
        for v in subset:
            Nv = self.graph.neighbors(v)
            v_bound.extend([u for u in Nv if u not in subset and u not in v_bound])
 
        s_ix = [self.node_to_index[node] for node in subset]
        v_bound_ix = [self.node_to_index[node] for node in v_bound]

        boundary_adj_mat = self.adj_mat[np.ix_(s_ix, v_bound_ix)]
        return boundary_adj_mat


    ##EXACT COMPUTATIONS

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
            f[start_node - 1] = 1.
        elif seed_vec is not None:
            f = seed_vec
        else:
            print 'no seed vector given'
            return
 
        heat_ker = self.heat_ker(t)
        return np.transpose(f).dot(heat_ker)

    #TODO confirm submatrices
    def dir_heat_ker(self, subset, t):
        '''
        Exact computation of H_{S,t} = exp{-t(I_S-P_S)}.
        Here, subset is a subset of nodes in the Network instance,
        given in ID sorted order.
        '''

        s = len(subset)

        # \Delta_S = I_S - D_S^{-1}A_S
        sub_deg_mat = self.subset_deg_mat(subset)
        sub_adj_mat = self.subset_adj_mat(subset)

        #node_to_index = {}
        #index_to_node = {}
        #i = 0
        #for node in subset:
        #    node_to_index[node] = i
        #    index_to_node[i] = node
        #    i += 1

        ## \Delta_S = I_S - D_S^{-1}A_S
        #ind_sub = self.graph.subgraph(subset)        
        #sub_adj_mat = np.zeros((s,s))
        #sub_deg_mat = np.zeros((s,s))
        #for node in subset:
        #    i = node_to_index[node]
        #    sub_deg_mat[i,i] = float(ind_sub.degree(node))
        #    for n in ind_sub.neighbors(node):
        #        j = node_to_index[n]
        #        sub_adj_mat[i,j] = 1.0
        sub_walk_mat = np.dot(np.linalg.inv(sub_deg_mat),sub_adj_mat)
        dir_laplace = np.eye(s) - sub_walk_mat

        # Dirichlet heat kernel
        return scipy.linalg.expm(-t*dir_laplace)

    def exp_dir_hkpr(self, subset, t, start_node=None, seed_vec=None):
        '''
        Exact computation of hkpr(S,t,f) = f^T H_{S,t} over a subset.
        Here, subset is a subset of nodes in the Network instance,
        given in ID sorted order.
        '''

        s = len(subset)

        # get seed vector
        if start_node is not None:
            if start_node not in subset:
                print 'start_node not in subset'
                return
            f = np.zeros(s)
            i = subset.index(start_node)
            f[i] = 1.
        elif seed_vec is not None:
            if len(seed_vec) != s:
                print 'seed vector not over subset of nodes'
                return
            f = seed_vec
        else:
            print 'no seed vector given'
            return

        dir_heat_ker = self.dir_heat_ker(subset, t)
        return np.transpose(f).dot(dir_heat_ker)


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
            p = np.asarray(self.walk_mat)[cur_node-1]
            next_node = np.random.choice(self.graph.nodes(), p=p)
            cur_node = next_node
            if verbose:
                print cur_node
        if verbose:
            print 'stop:', cur_node
        return cur_node

    def approx_hkpr(self, t, start_node=None, seed_vec=None, eps=0.1, verbose=False):
        '''
        Outputs an eps-approximate heat kernel pagerank vector computed
        with random walks.
        '''

        n = self.graph.size()

        # initialize 0-vector of size n
        approxhkpr = np.zeros(self.size) 

        r = (16.0/eps**3)*math.log(n)
        K = (math.log(1.0/eps))/(math.log(math.log(1.0/eps)))

        if verbose:
            print 'r', r
            print 'K', K

        for iter in range(int(r)):
            k = np.random.poisson(lam=t)
            k = int(min(k,K))

            v = self.random_walk(k, start_node=start_node, seed_vec=seed_vec, verbose=False)
            
            approxhkpr[v-1] += 1

        return r, K, approxhkpr/r

    ##Dirichlet

    def res_random_walk(self, k, subset, start_node=None, seed_vec=None, verbose=False):
        '''
        subset is a subset of nodes given in id sorted order.
        Outputs the last node visited in a k-step random walk on the subset.
        If start_node given, walk starts from start_node in the subset.
        If seed_vec given, walk starts from a node drawn from seed_vec over
        the subset.
        If neither are given, walk starts from a node drawn from
        p(v) = d_S(v)/vol(S).
        '''

        s = len(subset)

        node_to_index = {}
        index_to_node = {}
        i = 0
        for node in subset:
            node_to_index[node] = i
            index_to_node[i] = node
            i += 1

        # P_S = D_S^{-1}A_S
        #TODO right now we are using the random walk matrix for the induced
        # subgraph, not the submatrix P_S = D_S^{-1}A_S, in order to maintain
        # probabilities  
        ind_sub = self.graph.subgraph(subset)        
        sub_adj_mat = np.zeros((s,s))
        sub_deg_mat = np.zeros((s,s))
        for node in subset:
            i = node_to_index[node]
            sub_deg_mat[i,i] = float(ind_sub.degree(node))
            for n in ind_sub.neighbors(node):
                j = node_to_index[n]
                sub_adj_mat[i,j] = 1.0
        #sub_deg_mat = np.array(self.subset_deg_mat(subset))
        #sub_adj_mat = np.array(self.subset_adj_mat(subset))
        sub_walk_mat = np.dot(np.linalg.inv(sub_deg_mat),sub_adj_mat)

        if seed_vec is not None and len(seed_vec) != s:
            print 'seed vector not over subset of nodes'
            return

        
        ## Choose start node.  If start node is given, this is the start node.
        ## If a seed vector is given, assume the vector indexes the vertices in
        ## an increasing ordering.  If nothing is provided, choose a start node
        ## proportional to node degree in the induced subgraph.
        
        if start_node is not None:
            if start_node not in subset:
                print 'start_node not in subset'
                return
            cur_node = start_node
        elif seed_vec is not None:
            cur_node = np.random.choice(subset, p=seed_vec)
        else:
            # choose start node according to dv/vol(S)
            vol_subset = 0
            subset_deg_vec = np.zeros(s)
            i = 0
            for node in subset:
                vol_subset += self.graph.degree(node)
                subset_deg_vec[i] = self.graph.degree(node)
                i+= 1
            p = subset_deg_vec/vol_subset
            cur_node = np.random.choice(subset, p=p)
        if verbose:
            print 'start:', cur_node
        for steps in range(k):
            p = sub_walk_mat[node_to_index[cur_node]]
            next_node = np.random.choice([index_to_node[i] for i in range(s)], p=p)
            cur_node = next_node
            if verbose:
                print cur_node
        if verbose:
            print 'stop:'
        return cur_node

    def approx_dir_hkpr(self, subset, t, start_node=None, seed_vec=None, eps=0.1, verbose=False):
        '''
        subset is a subset of nodes in sorted order
        '''

        s = len(subset)

        # sort nodes in increasing id order
        node_to_index = {}
        i = 0
        for node in subset:
            node_to_index[node] = i
            i += 1

        # initialize a 0-vector of size s
        approxhkpr = np.zeros(s)

        r = (16.0/eps**3)*math.log(s)
        K = (math.log(1.0/eps))/(math.log(math.log(1.0/eps)))

        if verbose:
            print 'r', r
            print 'K', K

        for iter in range(int(r)):
            k = np.random.poisson(lam=t)
            k = int(min(k,K))

            v = self.res_random_walk(k, subset, start_node=start_node, seed_vec=seed_vec, verbose=False)
            
            approxhkpr[node_to_index[v]] += 1

        return approxhkpr/r

    ## Draw hkpr

    def draw_hkpr(self, vec, file_name):

        G = pydot.Dot(graph_type='graph')

        for n in self.graph.nodes():
            node = pydot.Node(str(n))
            node.set_style('filled')
            color = 255 - (vec[n-1]/max(vec)*255)
            node.set_fillcolor('#ff%02x%02x' % (color,color))
            G.add_node(node)

        for (u,v) in self.graph.edges():
            edge = pydot.Edge(str(u),str(v))
            G.add_edge(edge)

        G.write_png(file_name)

def main():
    karate = Network(gml_file='karate.gml')
    subset = [5,6,7,11,17]
    t, eps = 35.0, 0.01
     
    # choose start node according to dv/vol(G)
    total = sum(karate.deg_vec)
    p = karate.deg_vec/total
    start_node = np.random.choice(karate.graph.nodes(), p=p)

    print 't=',t,'\neps=',eps,'\nstart node=',start_node,'\n'

    #karate.res_random_walk(6,subset,verbose=True)
    #karate.random_walk(6,verbose=True)
    h = karate.approx_hkpr(t=t, start_node=start_node, eps=eps, verbose=True)
    print h
    print 'sum=',sum(h)

    true = karate.exp_hkpr(t=t, start_node=start_node)
    print '\n true hkpr:'
    print true
    print 'sum=',sum(true)

    # probability-per-degree ranking
    h_dic = {}
    true_dic = {}
    for i in range(karate.size):
        node = karate.index_to_node[i]
        h_dic[node] = h[i]/karate.graph.degree(node)
        true_dic[node] = true[i]/karate.graph.degree(node)

    sorted_h = sorted(h_dic.iteritems(), key=operator.itemgetter(1), reverse=True)
    sorted_true = sorted(true_dic.iteritems(), key=operator.itemgetter(1), reverse=True)

    print '\napprox ranking','\t\t','true ranking'
    for i in range(len(sorted_h)):
        print sorted_h[i],'\t',sorted_true[i]
 
if __name__ == 'main':
    main()
