import json
import numpy as np
import random as r 

def gen_erdos(size, p, no_diag = True):
    
    mat = np.zeros(size)
    
    for i in range(size[0]):
        for j in range(size[1]):
            if(not(i == j and no_diag)):        
                mat[i, j] = 1 if r.random() <= p else 0
    return mat

def rand(n):
    if(n == 1):
        return 1
    return r.randrange(n + 1)

def generate_block(block_matrix, block_size, bsdelta, fill_in, fill_out):
    size = block_matrix.shape[0]
    widths = [block_size + rand(2 * bsdelta) - bsdelta for i in range(size)]
 
    d = [[0 for i in range(size)] for j in range(size)]
    for i in range(size):
        for j in range(size):
            fill = fill_out
            if(block_matrix[i, j] == 1):
                fill = fill_in
            no_diag = i == j
            d[i][j] =  gen_erdos((widths[i], widths[j]), fill, no_diag)
    
    rows = [0 for i in range(size)]
    for i in range(size):
        rows[i] = np.column_stack(tuple(d[i])).transpose()
    return np.column_stack(tuple(rows)).transpose()

def to_python_array(np_array):
    sh = np.shape(np_array)
    s0 = sh[0]
    s1 = sh[1]

    out = [[0 for i in range(s1)] for j in range(s0)]
    

    for i in range(s0):
        for j in range(s1):
            out[i][j] = np_array[i, j]
    
    return out

"""
block_matrix = np.array([
    [1.0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 1]])

"""

block_matrix = np.array([
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0.0]])

size = block_matrix.shape[0] ** 2
ones = sum(map(sum, block_matrix))
zeros = size - ones

totFill = 1/size * (ones)


max_fill = 1
min_fill = totFill
steps = 50
delta = (max_fill - min_fill)/steps

fill_steps = [min_fill + delta * i for i in range(steps + 1)]

for fill_in in fill_steps:

    outmat = generate_block(block_matrix, 20, 5, fill_in, (size * totFill - ones * fill_in)/zeros)

    di = {"adj":to_python_array(outmat), "block":to_python_array(block_matrix)}

    s = json.dumps(di)
    
    f = open("{}.json".format(fill_in), "w")
    f.write(s)
    f.close()
    print()
    for row in outmat:
        print()
        for i in row:        
            print("X" if i == 1 else " " , end="")


