from itertools import product
from sys import stdout as out
from mip import Model, xsum, minimize, BINARY
import util
import networkx as nx

class graphSolver:

    def __init__(self, node_names, house_names, start, adj_mat):
        self.node_names = node_names
        self.house_names = house_names
        self.start = start
        self.adj_mat = adj_mat
        self.G = nx.Graph()

        for i in range(len(adj_mat)):
            for j in range(len(adj_mat)):
                if adj_mat[i][j] == 'x':
                    adj_mat[i][j] = -1
                else:
                    adj_mat[i][j] = float(adj_mat[i][j])
                    if i > j:
                        self.G.add_edge(self.node_names[i], self.node_names[j], weight=adj_mat[i][j])

    def solve(self):

        n, V = len(self.adj_mat), set(range(len(self.adj_mat)))

        model = Model()

        """
        Define Parameters
        d: {0, 1} whether or not an edge is selected
        y: for connectivity constraint
        s: whether or not a node is visited
        """
        d = {}
        for i in V:
            for j in V:
                if self.adj_mat[i][j] > 0:
                    d[(i, j)] = model.add_var(var_type=BINARY)

        y = [model.add_var() for i in V]
        s = [model.add_var() for i in V]

        """
        Define optimization function
        """
        def cost_func():
            total = 0
            for i in V:
                for j in V:
                    if self.adj_mat[i][j] > 0:
                        total += self.adj_mat[i][j] * d[(i,j)]
            return total
        model.objective = minimize(cost_func())

        """
        Add constraints
        """
        house_set = set(self.house_names)
        house_set.add(self.start)
        for i in range(len(self.node_names)):
            if self.node_names[i] in house_set:
                model += s[i] == 1

        for i in V:
            model += xsum(d[(i,j)] for j in V -{i} if self.adj_mat[i][j] > 0) == 1 * s[i]

        for i in V:
            model+= xsum(d[(j,i)] for j in V - {i} if self.adj_mat[i][j] > 0) == 1 * s[i]

        for (i, j) in product(V - {0}, V - {0}):
            if i != j and self.adj_mat[i][j] > 0:
                model += y[i] - (n+1)*d[(i,j)] >= y[j] - n

        model.optimize(max_seconds = 30)

        if model.num_solutions:

            nc = self.node_names.index(self.start)
            solution = []
            for _ in range(500):
                solution.append(self.node_names[nc])
                for i in V:
                    if (nc, i) in d and d[(nc, i)].x > 0.99:
                        nc = i
                        break
                if nc == self.node_names.index(self.start):
                    solution.append(self.start)
                    break
            return solution
        else:
            return None


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

def main():
    for i in range(8, 50):
        filename = str(i) + '_50'
        input_file = '../inputs/' + filename + '.in'
        output_file = '../outputs/' + filename + '.out'

        node_names, house_names, start_node, adj_mat = util.readInput(input_file)
        solver = graphSolver(node_names, house_names, start_node, adj_mat)
        path = solver.solve()
        if path and solver.fitness(path) > 0:
            dropoff = solver.get_pedestrian_walks(path)
            util.writeOutput(output_file, path, dropoff)
        else:
            print('FAILED: ' + input_file)


if __name__  == "__main__":
    main()
