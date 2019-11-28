import networkx as nx
import numpy as np
import random

class graphSolver:

    def __init__(self, node_names, house_names, start, adj_mat):

        self.node_names = node_names
        self.house_names = house_names
        self.start = start
        self.adj_mat = adj_mat
        self.G = nx.Graph()
        
        # Dictionary used to go from name to index number
        # The reverse can be done by indexing self.node_names
        self.node_indices = {}
        for i in range(len(node_names)):
            self.node_indices[self.node_names[i]] = i

        # Convert adjacency matrix to nx graph
        # TODO: currently adding edges one-by-one. Not sure how to feed in and adjacency
        # matrix that has 'x' elements that means no path.
        # TODO: do we still need the adjacency matrix? Currently converting all weight
        # elements to floats, but potentially can remove
        for i in range(len(adj_mat)):
            for j in range(len(adj_mat)):
                if adj_mat[i][j] != 'x':
                    adj_mat[i][j] = float(adj_mat[i][j])
                    if i > j:
                        self.G.add_edge(self.node_names[i], self.node_names[j], weight=adj_mat[i][j])

    def fitness(self, path):
        """
        Calculate fitness or score of a path to be maximized (inverse of the energy
        expended).

        Parameters
        ----------
        path: list of nodes in the order of traversal

        Return
        ------
        -1 if path is invalid
        inverse of total energy expended for the path
        """
        energy = 0

        for i in range(len(path) - 1):
            if self.G.has_edge(path[i], path[i + 1]):
                energy += self.G[path[i]][path[i + 1]]['weight']
            else:
                return -1
        
        '''
        for h in self.house_names:
            # find the shortest length from h to a node in path and add it to energy
            continue
        '''

        raise NotImplementedError

    # TODO: Nick working on for generating intial population of graphs
    #def generate_random_cycle():

    # TODO: @Steven @Jeffrey
    def shortest_path_to_cyle(self, path, node):
        """
        Calculate shortest distance from a node to the closest node on the path

        Parameters
        ----------
        path: list of nodes in the path taken by the car
        node: the house of a TA

        Return
        ------
        dist: float weight of the smallest cost of travelling from a node on the
        path to the node
        """


        raise NotImplementedError

        
def readInput(filename):
    """
    Read from an input file

    Parameters
    ----------
    filename: relative path to the input file

    Return
    ------
    node_names:     list of the names of all nodes
    house_names:    list of the names of houses
    start_node:     starting node
    adj_mat:        adjacency matrix
    """

    f = open(filename, 'r')
    
    num_nodes = int(f.readline().strip())
    num_houses = int(f.readline().strip())
    node_names = f.readline().strip().split(' ')
    house_names = f.readline().strip().split(' ')
    start_node = f.readline()

    adj_mat = []

    for _ in range(num_nodes):
        adj_mat.append(f.readline().strip().split(' '))

    f.close()

    return node_names, house_names, start_node, adj_mat

def main():
    
    node_names, house_names, start, adj_mat = readInput('../inputs/99_50.in')
    solver = graphSolver(node_names, house_names, start, adj_mat)

    temp_path = ['1', '3', '5', '1']

    print(solver.fitness(temp_path))
    
    '''
    for row in adj_mat:
        print(row)
    '''

if __name__  == "__main__":
    main()
