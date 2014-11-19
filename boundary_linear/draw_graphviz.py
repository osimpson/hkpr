import networkx as nx
import numpy as np
import pydot
from Network import *


def draw_subset(Net, subset, file_name, vertex_boundary=None):

    G = pydot.Dot(graph_type='graph')

    for n in Net.graph.nodes():
        node = pydot.Node(str(n))
        if n in subset:
            node.set_style('filled')
            node.set_fillcolor('red')
        if vertex_boundary is not None:
            if n in vertex_boundary:
                node.set_style('filled')
                node.set_fillcolor('orchid3')
        G.add_node(node)

    for (u,v) in Net.graph.edges():
        edge = pydot.Edge(str(u),str(v))
        G.add_edge(edge)

    G.write_png(file_name, prog='neato')
