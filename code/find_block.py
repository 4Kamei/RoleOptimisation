import networkx as nx
import numpy as np
import random as rand
import math
import pygraphviz as pgv

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



"""
adj = np.matrix(
       [[0, 1, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]])
"""


g = nx.read_gml("celegansneural.gml")

adj = nx.to_numpy_matrix(g)

print(adj)

g = nx.from_numpy_array(adj, create_using=nx.DiGraph)

M = len(g.edges())

N = len(g.nodes())

Nb = 5

b = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        b[i, j] = g.in_degree(i) * g.out_degree(j) / M

def compute_optimal_roles(adj, N, Nb, b):    
    def state_cost(state):
        e = compute_e(state, adj, N, Nb)
        Ee = compute_Ee(state, adj, N, Nb, b)
        cost = 0
        for r in range(Nb):
            for s in range(Nb):
                cost += abs(e[r][s] - Ee[r][s])
        return cost / 2

    def perturb(state):
        i = rand.randrange(N)
        r = rand.randrange(Nb)
        s = state.copy()
        s[i] = r
        return s
    print(" --- start --- ")
    #generate a random initialisation vector
    max_state = [rand.randrange(Nb) for i in range(N)]
    
    max_cost = state_cost(max_state)

    T = 5
    alpha = 0.95
    max_iter = 10000
    min_T = 0.1
    curr_state = max_state.copy()
    curr_cost = max_cost

    while(T > min_T):
    
        print("adjusting temperature T = {}".format(T))

        curr_state = max_state.copy()
        curr_cost = max_cost
    
        for ite in range(max_iter):
            new_s = perturb(curr_state)
            c = state_cost(new_s)
            delta = curr_cost - c
            if(rand.random() < math.exp(delta  / T)):
                curr_state = new_s
                curr_cost = c

                if(curr_cost > max_cost):
                    max_state = curr_state
                    max_cost = state_cost(max_state)
                    print("new max found T = {} distance to maximum".format(T))
                    print("     {} : {}".format(max_cost, str(max_state)))


        T = T * alpha
         
    return max_state, max_cost

    #run simulated annealing 

def compute_e(roles, adj, N, Nb):
    e = np.zeros((Nb, Nb))
    for i in range(N):
        for j in range(N):
            e[roles[i], roles[j]] += adj[i, j]
    return e/M

def compute_Ee(roles, adj, N, Nb, b):
    Ee = np.zeros((Nb, Nb))
    for i in range(N):
        for j in range(N):
            Ee[roles[i], roles[j]] += b[i, j]
    return Ee/M

def derive_best_block(opt_roles, adj, N, Nb, b):

    block = np.zeros((Nb, Nb))

    e = compute_e(opt_roles, adj, N, Nb)
    Ee = compute_Ee(opt_roles, adj, N, Nb, b)

    for r in range(Nb):
        for s in range(Nb):
            desc = e[r][s] - Ee[r][s]
            if(desc > 0):
                print("({}, {}) = {}".format(r, s, desc))
                block[r, s] = 1
    
    return block


def compute_optimal_partition(adj, N, Nb, b, B):
    
    def cost_delta(node, oldC, newC):

        nonlocal curr_state

        k_in = [0 for i in range(Nb)]
        k_out = [0 for i in range(Nb)]

        ek_in = [0 for i in range(Nb)]
        ek_out = [0 for i in range(Nb)]

        for j in range(N):
            k_in[curr_state[j]] += adj[j, node]
            k_out[curr_state[j]] += adj[node, j]
            ek_in[curr_state[j]] += b[j][node]
            ek_out[curr_state[j]] += b[node][j]

        delta = 0
        for s in range(Nb):
            f1 = k_out[s] - ek_out[s]
            f2 = B[newC][s] - B[oldC][s]
            delta += f1 * f2

        for r in range(Nb):
            f3 = k_in[r] - ek_in[r]
            f4 = B[r][newC] - B[r][oldC]
            delta += f3 * f4
        return delta/M

    def perturb(state):
        i = rand.randrange(N)
        r = rand.randrange(Nb)
        return i, r

    max_state = [rand.randrange(Nb) for i in range(N)]
    max_cost = 0

    T = 1
    Tmin = 0.01
    alpha = 0.99
    max_iter = N * Nb
    rand.seed(4234)

    curr_state = max_state.copy()
    curr_cost = max_cost

    while(T > Tmin):
        if(curr_cost < max_cost):
            curr_cost = max_cost
            curr_state = max_state.copy()

        for it in range(max_iter):

            node, newC = perturb(curr_state)
            delta = cost_delta(node, curr_state[node], newC)
            if(rand.random() < math.exp(delta / T)):
                curr_cost += delta
                curr_state[node] = newC

        if(curr_cost > max_cost):
            max_cost = curr_cost
            max_state = curr_state.copy()
            print("new maximum found T = {}".format(T))
            print(" {} : {}".format(max_cost, str(max_state)))
        
        T = T * alpha
    return max_state
 
def get_random_state(cur_state, N, Nb):
    i = rand.randrange(N)
    s = rand.randrange(Nb)
    return i, s

def init(seed, N, Nb):
    return [rand.randrange(Nb) for i in range(N)]
  
def draw(g, name, partitions=None):

    agr = nx.nx_agraph.to_agraph(g)
    agr.layout(prog="dot")
    agr.node_attr["style"] = "filled"
    if(partitions):
        cols = ["red", "green", "blue", "orange", "cyan", "magenta", "aliceblue", "gold", "greenyellow"]
        col = -1
        for partition in partitions:
            col += 1
            if(col == len(cols)):
                col -= 1
            for p in partition:
                n = agr.get_node("{}".format(p))
                n.attr['fillcolor'] = cols[col]
    agr.draw("{}.png".format(name))


"""
Q = [0 for i in range(N - 1)]
Opts = [[] for i in range(N -1)]

for i in range(N -1):
    Opts[i], Q[i] = compute_optimal_roles(adj, N, i + 1, b)
    
for i in range(N -1):
    print("{} - {} : {}".format(i, str(Opts[i]), Q[i]))
"""

index = input("Nb = ")

Nb = int(index)

opt_roles = compute_optimal_roles(adj, N, Nb, b)

block = derive_best_block(opt_roles, adj, N, Nb, b)

print(block)

opt_partition = compute_optimal_partition(adj, N, Nb, b, block)

print("found optimal partition")

print(opt_partition)

partitions = [set([]) for i in range(Nb)]

for i in range(N):
    partitions[opt_partition[i]].add(i)

draw(g, "test", partitions)
