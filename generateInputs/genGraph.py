from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import math

def check_sym(adj):
    for i in range(len(adj)):
        for j in range(len(adj)):
            assert adj[i][j] == adj[j][i]
    print("Adjacency matrix is symmetric")

def gen_adj_matrix(V, D, S):

    positions = np.random.rand(V, D)
    differences = positions[:, None, :] - positions[None, :, :]
    distances = np.sqrt(np.sum(differences**2, axis=-1)) # euclidean
    distances = distances.tolist()

    for i in range(V):
        for j in range(V):
            distances[i][j] = truncate(distances[i][j] * 100, 5)
            if random.random() < S:
                distances[i][j] = 'x'
            if i > j:
                distances[i][j] = distances [j][i]

    return distances

def truncate(n, d):
    stepper = 10.0 ** d
    return math.trunc(stepper * n) / stepper

if __name__ == "__main__":
    
    V = 200 # number of nodes
    D = 2 # dimensionality
    S = 0.7 # sparsity
    num_TAs = 93 # number of houses to visit
  
    # Randomly choose TA houses
    house_list = []
    house_list_copy = []
    TA_house_list = []
    for i in range(V):
        house_list.append('node' + str(i + 1))
        house_list_copy.append('node' + str(i + 1))
    for _ in range(num_TAs):
        i = random.randint(0, len(house_list) - 1)
        TA_house_list.append(house_list.pop(i))
    start_node = house_list.pop(random.randint(0, len(house_list) - 1))

    # *********************
    # Generate random adjacency matrix and check validity
    # *********************
    
    adj_mat = gen_adj_matrix(V, D, S)
    check_sym(adj_mat)
   

    # *********************
    # Write to file
    # *********************

    # Write #nodes and #TAs
    f = open(str(V) + '.in', 'w')
    f.write(str(V) + '\n')
    f.write(str(num_TAs) + '\n')
    
    # Write all node names
    for i in range(V):
        f.write(house_list_copy[i])
        if i + 1 < V:
            f.write(' ')
    f.write('\n')

    # Write TA home locations
    for i in range(num_TAs):
        f.write(TA_house_list[i])
        if i + 1 < num_TAs:
            f.write(' ')
    f.write('\n')

    # Write starting location
    f.write(start_node + '\n')

    # Write adjacency matrix
    for i in range(V):
        for j in range(V):
            f.write(str(adj_mat[i][j]))
            if j + 1 < V:
                f.write(' ')
        if i + 1 < V:
            f.write('\n')
    f.close()
    
    # *********************
    # Debugging Methods
    # *********************

    # create a weighted, directed graph in networkx
    #graph = nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph())

    #print(nx.adjacency_matrix(graph).todense())
    
    #nx.draw(graph)
    #plt.show()


