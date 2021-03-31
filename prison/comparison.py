import networkx as nx
import numpy as np
import math 
from networkx.algorithms.community import greedy_modularity_communities
import pygraphviz as pgv

block = np.array(
        [[1, 0, 0],
 [0, 1, 0],
 [0, 0, 1]]

        )

gr = nx.from_numpy_array(block, create_using=nx.DiGraph)

def draw(g, name, partitions=None, edges=None):

    cols = ["red", "green", "blue", "orange", "cyan", "magenta", "aliceblue", "gold", "greenyellow"]
    agr = nx.nx_agraph.to_agraph(g)
    agr.layout(prog="dot")
    agr.node_attr["style"] = "filled"
    if(partitions):
        col = -1
        for partition in partitions:
            col += 1
            for p in partition:
                n = agr.get_node("{}".format(p))
                n.attr['fillcolor'] = cols[col]
    if(edges):
        col = -1
        for edge in edges:
            for ed in edge:
                col += 1
                e = agr.get_edge(*ed)
                e.attr['color'] = cols[col]

    agr.draw("{}.png".format(name))

edges = [set([(3, 1)])]

draw(gr, "3_roles", None)


