class graphSolver:

    def __init__(self, node_names, house_names, start, adj_mat):

        self.node_names = node_names
        self.house_names = house_names
        self.start = start
        self.adj_mat = adj_mat
        
        self.node_indices = {}
        for i in range(len(node_names)):
            self.node_indices[self.node_names[i]] = i

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
            node1 = self.node_indices[path[i]]
            node2 = self.node_indices[path[i + 1]]
            if type(self.adj_mat[node1][node2]) == type(''):
                return -1
            energy += float(self.adj_mat[node1][node2])

        print(energy)
        '''
        for h in self.house_names:
            # find the shortest length from h to a node in path and add it to energy
            continue
        '''

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

    solver.fitness(temp_path)
    '''
    for row in adj_mat:
        print(row)
    '''

if __name__  == "__main__":
    main()
