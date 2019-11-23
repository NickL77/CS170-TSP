from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import sys


filename = '50.in'
adj_mat = []

f = open(filename, 'r')

num_nodes = int(f.readline().strip('\n'))
num_houses = int(f.readline().strip('\n'))
node_names = f.readline().strip('\n').split(' ')
house_names = f.readline().strip('\n').split(' ')
adjusted_node_names = node_names[:]
start_node = f.readline()
start_num = int(start_node[4:])

for _ in range(num_nodes):
    row = f.readline().split(' ')
    adj_mat.append(row)

f.close()

for i in range(num_nodes):
    for j in range(num_nodes):
        adj_mat[i][j] = adj_mat[i][j].strip('\n')
        if adj_mat[i][j] == 'x':
            adj_mat[i][j] = sys.maxsize
        else:
            adj_mat[i][j] = int(float(adj_mat[i][j]) * 10000)
'''
house_numbers = []
for h in house_names[::-1]:
    house_numbers.append(int(h[4:]))
non_house_numbers = []

for i in range(num_nodes):
    if i+1 not in house_numbers:
        if i == start_num:
            print('not in:', i)
        else:
            non_house_numbers.append(i)
non_house_numbers.sort(reverse=True)
print(non_house_numbers)

for i in non_house_numbers:
    adjusted_node_names.pop(i) 
    adj_mat.pop(i)
    for row in adj_mat:
        row.pop(i)
'''

def print_solution(manager, routing, assignment):
    """Prints assignment on console."""
    print('Objective: {} miles'.format(assignment.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Route for vehicle 0:\n'
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = assignment.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    print(plan_output)
    plan_output += 'Route distance: {}miles\n'.format(route_distance)
    return plan_output


data = {}
data['distance_matrix'] = adj_mat
data['num_vehicles'] = 1
data['depot'] = 0

# Create the routing index manager.
manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                       data['num_vehicles'], data['depot'])

# Create Routing Model.
routing = pywrapcp.RoutingModel(manager)


def distance_callback(from_index, to_index):
    """Returns the distance between the two nodes."""
    # Convert from routing variable Index to distance matrix NodeIndex.
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]

transit_callback_index = routing.RegisterTransitCallback(distance_callback)

# Define cost of each arc.
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# Setting first solution heuristic.
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

# Solve the problem.
assignment = routing.SolveWithParameters(search_parameters)

# Print solution on console.
if assignment:
    path = print_solution(manager, routing, assignment)
    path_ls = path.split(' ')
    node_order = []
    for p in path_ls:
        if p.isdigit():
            node_order.append(int(p))
            #print(p, adjusted_node_names[int(p)])
    while node_order[0] != start_num:
        node_order.append(node_order.pop(0))
    node_order.append(node_order[0])
    print(node_order)
    outfile = str(num_nodes) + '.out'
    
    f = open(outfile, 'w')
    
    for i in range(len(node_order)):
        f.write('node' + str(node_order[i]))
        if i + 1 < node_order:
            f.write(' ')
    f.write('\n')
    f.write(str(num_houses) + '\n')
    
    for n in house_names:
        f.write(n + ' ' + n + '\n')

    f.close()




