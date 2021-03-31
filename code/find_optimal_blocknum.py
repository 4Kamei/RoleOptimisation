import networkx as nx
import numpy as np
import random as rand
import math

#given a vector of community assignments, computes a state object 
def init_state(sigma, Npart, b, adj):
    N = len(sigma)
    parts = [set() for i in range(Npart)]
    for i in range(N):
        parts[sigma[i]].add(i)

    state = {}
    state["part"] = parts
    e = compute_e(parts, adj)
    Ee = compute_e(parts, b)
    state["e"] = e
    state["Ee"] = Ee
    state["adj"] = adj
    state["b"] = b

    k_in =  [[0 for j in range(Npart)] for i in range(N)]
    k_out = [[0 for j in range(Npart)] for i in range(N)]

    state["k_in"]  = k_in
    state["k_out"] = k_out

    for i in range(N):
        for r in range(Npart):
            for j in parts[r]:
                k_in[i][r] += adj[i, j] 
                k_out[i][r] += adj[j, i]

    for key in state.keys():
        print("")
        print(key)
        print(state[key])

    return state
 
#given a state vector 
def update_state(old_state, changed_node, from_community, to_community):
    print("test")

def get_b(adj):
    """
    in_deg = [0 for i in range(N)]
    out_deg = [0 for i in range(N)]

    for i in range(N):
        for j in range(N):
            in_deg[i] += adj[j, i]
            out_deg[i] += adj[i, j]
    """
    b = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            b[i, j] = g.out_degree(i) * g.in_degree(j) / M
    return b

def compute_e(part, matrix):
    Npart = len(part)
    e = np.zeros((Npart, Npart))
    for r in range(Npart):
        for s in range(Npart):
            for i in part[r]:
                for j in part[s]:
                    e[r, s] += matrix[i, j]
    return e/M
  
def state_cost(state):
    e = compute_e(state, adj, N, Nb)
    Ee = compute_Ee(state, adj, N, Nb, b)
    cost = 0
    for r in range(Nb):
        for s in range(Nb):
            cost += abs(e[r][s] - Ee[r][s])
    return cost / 2




#get a graph, any graph
"""
adj = np.matrix(
        [[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]])
"""
adj = np.matrix(
       [[0, 1, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]])

print(adj)

g = nx.from_numpy_array(adj, create_using=nx.DiGraph)

M = len(g.edges())

N = len(g.nodes())

b = get_b(g)

rand.seed(234)

Npart = 3

sigma = [0, 2, 0, 1]

#state is a dictionary containing all the 

state = init_state(sigma, Npart, b, adj)

