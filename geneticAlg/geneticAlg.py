import networkx as nx
import numpy as np
import random
import util
import sys

class graphSolver:

    def __init__(self, node_names, house_names, start, adj_mat):

        # Genetic Algo Hyperparameters
        self.default_iterations = 100
        self.population_size = 100
        self.elite_size = int(self.population_size * 0.01)
        self.mutation_rate = 0.05

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

        '''
        self.starting_population = self.get_initial_population()
        population = self.starting_population
        for _ in range(100):
            population = self.next_generation(population)
        #self.next_generation(self.starting_population)
        '''

    def run_evolution(self, iterations=100, population=None):
        if population == None:
            population = self.get_initial_population()
        for _ in range(iterations):
            population = self.next_generation(population)
        return population[0]

    def next_generation(self, population):
        
        curr_population = []
        next_population = []
        total = 0
        for c in population:
            f = self.fitness(c)
            total += 1/f
            curr_population.append((c, f))
        
        curr_population = sorted(curr_population, key=lambda x: x[1])
        #print(str(int(curr_population[0][1])))
        print(str(int(curr_population[0][1])) + ' ' + str(int(curr_population[10][1])) + ' ' + str(int(curr_population[25][1])))

        def select_parents():
            s1 = random.uniform(0, total)
            s2 = random.uniform(0, total)
            parent1, parent2 = None, None
            curr = 0
            for p in curr_population:
                if parent1 and parent2:
                    break
                curr += 1/p[1]
                if not parent1 and curr > s1:
                    parent1 = p
                if not parent2 and curr > s2:
                    parent2 = p
            return parent1, parent2

        for i in range(self.elite_size):
            next_population.append(curr_population[i][0])

        while len(next_population) < self.population_size:
            p1, p2 = None, None
            while p1 == p2:
                p1, p2 = select_parents()
            c1, c2 = self.breed(p1[0], p2[0])
            if c1 == None:
                continue
            next_population.extend([c1, c2])

        return next_population

    def breed(self, path1, path2):
        
        s = set(path1)
        intersection = [v for v in path2 if v in s and v != self.start]
       
        if len(intersection) == 0:
            return None, None

        node = random.choice(intersection)

        i1 = path1.index(node)
        i2 = path2.index(node)

        child1 = path1[:i1] + path2[i2:]
        child2 = path2[:i2] + path1[i1:]

        if random.random() < self.mutation_rate * 1:
            child1 = self.mutate(child1)
        
        if random.random() < self.mutation_rate * 1:
            child2 = self.mutate(child2)

        return [child1, child2]


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

        energy *- 2.0/3.0

        for h in self.house_names:
            # find the shortest length from h to a node in path and add it to energy
            e, _ = self.shortest_path_to_cycle(path, h)
            energy += e

        return energy

    def get_pedestrian_walks(self, path):
        dropoff_locations = {}
        for h in self.house_names:
            _, n = self.shortest_path_to_cycle(path, h)
            dropoff_locations[n] = dropoff_locations.get(n, []) + [h]
        return dropoff_locations

    def generate_random_cycle(self):
        """
        Generate a valid random cycle within G

        Return
        ------
        rand_path: random cycle for vehicle to travel. Length of path correlates to
        a normally generated random number
        """

        # Normal Distribution Parameters
        mu = len(self.node_names) * 3 / 4
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

    def mutate(self, path):
        new_visit = int(random.uniform(0, len(self.node_names)))
        new_visit = self.node_names[new_visit]
        cut = int(random.uniform(0, len(path) - 1))

        if new_visit == path[cut] or new_visit == path[cut + 1]:
            return path
        
        path1 = nx.shortest_path(self.G, source=path[cut], target=new_visit)
        path2 = nx.shortest_path(self.G, source=new_visit, target=path[cut + 1])

        new_path = path[:cut] + path1 + path2[1:-1] + path[cut + 1:]
        return new_path

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
        return final_cost, goal

    def get_initial_population(self):
        initial_population = []
        for _ in range(self.population_size):
            initial_population.append(self.generate_random_cycle())
        return initial_population

    def solve(self, filename):
        output_file = '../outputs/' + filename + '.out'

        best_path = None
        best_score = sys.maxint

        for _ in range(3):
            result_path = self.run_evolution(10000000)
            fit = self.fitness(result_path)
            print(result_path)
            print(fit)
            if fit < best_score:
                best_score = fit
                best_path = result_path

        best_score = self.fitness(best_path)
        print(best_score)

        result_dropoff = self.get_pedestrian_walks(best_path)
        writeOutput(output_file, best_path, result_dropoff)

def main():

    #node_names, house_names, start, adj_mat = readInput('../inputs/9_50.in')
    #node_names, house_names, start, adj_mat = readInput('../inputs/7_100.in')
    #node_names, house_names, start, adj_mat = readInput('../inputs/7_200.in')
    

    #solver = graphSolver(node_names, house_names, start, adj_mat)
    #result_path = solver.run_evolution(100)
    #print(result_path)
    
    '''
    best_path = None
    best_score = 900

    for _ in range(10):
        result_path = solver.run_evolution(20)
        print(result_path)
        print(solver.fitness(result_path))
        if solver.fitness(result_path) < best_score:
            best_score = solver.fitness(result_path)
            best_path = result_path
    result_path = best_path
    result_dropoff = solver.get_pedestrian_walks(result_path)

    writeOutput('../outputs/9_100.out', result_path, result_dropoff)
    '''

    filename = '../inputs/1_50.in'
    node_names, house_names, start, adj_mat = readInput(filename)

    path = ['1', '28', '19', '16', '23', '32', '15', '24', '41', '22', '37', '26', '34', '3', '11', '46', '44', '12', '25', '35', '17', '20', '45', '39', '31', '1']
    solver = graphSolver(node_names, house_names, start, adj_mat)
    result_dropoff = solver.get_pedestrian_walks(path)
    writeOutput('1_50.out', path, result_dropoff)
    print(solver.fitness(path))

    '''
    for i in range(1, 8):
        filename = str(i) + '_50'
        print(filename)
        input_file = '../inputs/' + filename + '.in'
        #input_file = 'small_8.in'
        node_names, house_names, start, adj_mat = readInput(input_file)
        solver = graphSolver(node_names, house_names, start, adj_mat)
        solver.solve(filename)
    '''

    '''
    for row in adj_mat:
        print(row)
    '''

def writeOutput(filename, path, dropoff):
    f = open(filename, 'w')

    for i in range(len(path)):
        f.write(path[i])
        if i + 1 < len(path):
            f.write(' ')
    f.write('\n')

    f.write(str(len(dropoff)) + '\n')

    for k in dropoff:
        f.write(k + ' ')
        for i in range(len(dropoff[k])):
            f.write(dropoff[k][i])
            if i + 1 < len(dropoff[k]):
                f.write(' ')
        f.write('\n')


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
