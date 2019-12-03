from itertools import product
from sys import stdout as out
from mip import Model, xsum, minimize, BINARY, OptimizationStatus
import util
import networkx as nx
import copy

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

        self.full_mat, self.implicit_edge_map = self.make_complete_graph();

    def solve(self, max_runtime = 120):

        n, V = len(self.adj_mat), set(range(len(self.adj_mat)))
        H = set(range(len(self.house_names)))
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
                #if self.adj_mat[i][j] > 0:
                if self.full_mat[i][j] > 0:
                    d[(i, j)] = model.add_var(var_type=BINARY)

        z = [model.add_var() for i in V]
        y = [model.add_var(var_type=BINARY) for i in V]

        # i is node, j is house/pedestrian
        c = []
        s = []
        for i in V:
            c.append(self.closest_nodes(10, self.node_names[i]))
            temp_row = []
            for j in H:
                temp_row.append(model.add_var(var_type=BINARY))
            s.append(temp_row)

        """
        Define optimization function
        """
        def cost_func():
            total = 0
            for i in V:
                for j in V:
                    if self.full_mat[i][j] > 0:
                        total += self.full_mat[i][j] * d[(i,j)] * 2.0 / 3.0
            for i in V:
                for j in H:
                    total += c[i][j] * s[i][j]
            return total
        model.objective = minimize(cost_func())

        """
        Add constraints
        """
        model += y[self.node_names.index(self.start)] == 1

        # Will need if skimp on distance calulation with closest_nodes
        #for i in H:
        #    model += xsum(s[j][i] * c[j][i] for j in V) >= 0.2

        for i in H:
            model += xsum(s[j][i] for j in V) == 1

        for i in V:
            for j in H:
                model += s[i][j] <= y[i]

        for i in V:
            model += xsum(d[(i,j)] for j in V -{i} if self.full_mat[i][j] > 0) == 1 * y[i]

        for i in V:
            model+= xsum(d[(j,i)] for j in V - {i} if self.full_mat[i][j] > 0) == 1 * y[i]

        for (i, j) in product(V - {0}, V - {0}):
            if i != j and self.full_mat[i][j] > 0:
                model += z[i] - (n+1)*d[(i,j)] >= z[j] - n

        status = model.optimize(max_seconds = max_runtime)

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
            solution = self.make_valid_path(solution)
            print(solution)
            return solution, status, model.gap
        else:
            return None, status, model.gap

    def make_complete_graph(self):

        full_mat = copy.deepcopy(self.adj_mat)
        implicit_edge_map = {}

        for i in range(len(full_mat)):
            for j in range(len(full_mat)):
                if full_mat[i][j] == -1 and i != j:
                    source = self.node_names[i]
                    target = self.node_names[j]
                    weight, shortest_path = nx.single_source_dijkstra(self.G, source=source, target=target)
                    full_mat[i][j] = weight
                    implicit_edge_map[(source, target)] = shortest_path
        return full_mat, implicit_edge_map

    def make_valid_path(self, path):
        valid_path = []
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i+1]
            valid_path.extend(self.implicit_edge_map.get((source, target), [source, target])[:-1])
        valid_path.append(path[-1])
        return valid_path


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

    def closest_nodes(self, n, node):
        """
        Calculates n closest nodes from `node` to offer as items to purchase (pedestrians
        walking home)

        Parameters
        ----------
        n: num closest nodes to find
        node: the house of a TA

        Return
        ------
        closest: list of n closest nodes
        """
        visited = set()
        fringe = util.PriorityQueue()
        fringe.push(node, 0)
        closest_nodes = {}
        while not fringe.isEmpty():
            if fringe.isEmpty():
                return None
            curr_node, final_cost = fringe.pop()
            if curr_node not in visited:
                closest_nodes[curr_node] = final_cost
                visited.add(curr_node)
                for child in list(self.G.neighbors(curr_node)):
                    cost = final_cost + self.G[curr_node][child]['weight']
                    fringe.update(child, cost)

        available_dropoffs = []
        for n in self.house_names:
            if n in closest_nodes:
                available_dropoffs.append(closest_nodes[n])
            else:
                print('phat L')
                available_dropoffs.append(-1)
        return available_dropoffs

def main():

    for i in range(1, 366):
        try:
            filename = str(i) + '_100'
            input_file = '../inputs/' + filename + '.in'
            #output_file = '../outputs/optimal/' + filename + '.out'
            output_file = '../outputs/100test_new/' + filename + '.out'

            print('Solving: ', filename)

            node_names, house_names, start_node, adj_mat = util.readInput(input_file)
            solver = graphSolver(node_names, house_names, start_node, adj_mat)

            path, status, gap = solver.solve(120)
            if gap > 100:
                continue
            gap = int(gap * 100)

            if path and solver.fitness(path) > 0:
                dropoff = solver.get_pedestrian_walks(path)
                if status == OptimizationStatus.OPTIMAL:
                    util.writeOutput(output_file, path, dropoff)
                else:
                    #suboptimal_output_file = '../outputs/suboptimal/' + filename + '_gap_' + str(gap) + '.out'
                    suboptimal_output_file = output_file
                    util.writeOutput(suboptimal_output_file, path, dropoff)
            else:
                print('FAILED: ' + input_file)
        except (ValueError, FileNotFoundError, IndexError, nx.NetworkXError) as e:
            print('FAILED', e)
    '''
    input_file = '../inputs/7_50.in'
    node_names, house_names, start_node, adj_mat = util.readInput(input_file)
    solver = graphSolver(node_names, house_names, start_node, adj_mat)
    print(solver.closest_nodes(10, '1'))
    '''




if __name__  == "__main__":
    main()
