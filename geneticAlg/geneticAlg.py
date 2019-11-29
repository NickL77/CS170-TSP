import networkx as nx
import numpy as np
import random
import util

class graphSolver:

    def __init__(self, node_names, house_names, start, adj_mat):

        # Genetic Algo Hyperparameters
        self.population_size = 100
        self.mutation_rate = 0.03

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

        for h in self.house_names:
            # find the shortest length from h to a node in path and add it to energy
            energy += self.shortest_path_to_cycle(path, h)

        return energy

    def generate_random_cycle(self):
        """
        Generate a valid random cycle within G

        Return
        ------
        rand_path: random cycle for vehicle to travel. Length of path correlates to
        a normally generated random number
        """

        # Normal Distribution Parameters
        mu = len(self.node_names) / 2
        std_dev = len(self.node_names) / 10

        # Generate a list of random nodes to visit
        node_names_copy = self.node_names[:]
        random.shuffle(node_names_copy)

        psuedo_path = [self.start]
        for _ in range(int(random.gauss(mu, std_dev))):
            psuedo_path.append(node_names_copy.pop())
        psuedo_path.append(self.start)

        # Connect random paths with shortest paths
        rand_path = [self.start]
        for i in range(len(psuedo_path) - 1):
            node1 = psuedo_path[i]
            node2 = psuedo_path[i + 1]

            connection = nx.shortest_path(self.G, source=node1, target=node2)
            rand_path.extend(connection[1:])

        return rand_path

    def shortest_path_to_cycle(self, path, node):
        """
        Calculate shortest distance from a TA's house to the closest dropoff point on the path the vehicle takes

        Parameters
        ----------
        path: list of nodes in the path taken by the car
        node: the house of a TA

        Return
        ------
        dist: float weight of the smallest cost of travelling from a node on the
        path to the input node
        """
        visited = set()
        fringe = util.PriorityQueue()
        goal = None
        fringe.push(node, 0)
        foundPath = False
        final_cost = float('inf')
        while not foundPath:
            if fringe.isEmpty():
                return None
            curr_node, final_cost = fringe.pop()
            if curr_node in path:
                goal = curr_node
                foundPath = True
            elif curr_node not in visited:
                visited.add(curr_node)
                for child in list(self.G.neighbors(curr_node)):
                    cost = final_cost + self.G[curr_node][child]['weight']
                    fringe.update(child, cost)
        return final_cost

def main():

    #node_names, house_names, start, adj_mat = readInput('../inputs/99_50.in')
    #node_names, house_names, start, adj_mat = readInput('../inputs/99_200.in')
    
    solver = graphSolver(node_names, house_names, start, adj_mat)
    #temp_path = ['1', '3', '5', '1']
    temp_path = solver.generate_random_cycle()

    print(solver.fitness(temp_path))

    '''
    for row in adj_mat:
        print(row)
    '''

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
    start_node = f.readline().strip()

    adj_mat = []

    for _ in range(num_nodes):
        adj_mat.append(f.readline().strip().split(' '))

    f.close()

    return node_names, house_names, start_node, adj_mat

if __name__  == "__main__":
    main()
