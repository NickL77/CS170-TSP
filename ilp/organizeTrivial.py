import os
import ilp
import util
import csv

def get_score_status():

    input_dir = '../inputs/'
    output_dir = '../outputs/'
    optimal_dir = output_dir + 'optimal/'
    suboptimal_dir = output_dir + 'suboptimal/'

    d = {}
    optimal_solutions = {}
    suboptimal_solutions = {}

    scores = {}

    for f in os.listdir(optimal_dir):
        optimal_solutions[f.strip('.out')] = f

    for f in os.listdir(suboptimal_dir):
        temp = 0
        for i in range(len(f)):
            if f[i] == '_':
                if temp == 1:
                    suboptimal_solutions[f[:i]] = f
                    break
                temp += 1

    for f in os.listdir(input_dir):
        problem = f.strip('.in')
        print(problem)
        if problem not in optimal_solutions and problem not in suboptimal_solutions:
            d[problem] = ['Unsolved', -1.0, -1.0]
        else:
            nodes, houses, start, adj_mat = util.readInput(input_dir + f)
            solver = ilp.graphSolver(nodes, houses, start, adj_mat)
            if problem in optimal_solutions:
                output_file = open(optimal_dir + optimal_solutions[problem], 'r')
                path = output_file.readline().strip().split(' ')
                output_file.close()
                d[problem] = ['Optimal', solver.fitness(path), 0.0]

                curr_cost = solver.fitness(path)
                trivial_cost = solver.fitness(path[:1])
                score = (curr_cost/trivial_cost)*100
                if not scores.get(score, False):
                    scores[score] = [problem]
                else:
                    scores[score].append(problem)

            elif problem in suboptimal_solutions:
                output_file = open(suboptimal_dir + suboptimal_solutions[problem], 'r')
                path = output_file.readline().strip().split(' ')
                output_file.close()
                gap = ''
                for l in suboptimal_solutions[problem].strip('.out')[::-1]:
                    if l == '_':
                        break
                    gap += l
                gap = float(gap[::-1])
                d[problem] = ['Suboptimal', solver.fitness(path), gap]

                curr_cost = solver.fitness(path)
                trivial_cost = solver.fitness(path[:1])
                score = (curr_cost/trivial_cost)*100
                if not scores.get(score, False):
                    scores[score] = [problem]
                else:
                    scores[score].append(problem)
    return scores

d = get_score_status()
for i in sorted(d):
    print(i, d[i])
'''
f = open('progress.csv', 'w')
progress_file = csv.writer(f, delimiter = ',')
for k in d:
    progress_file.writerow([k] + d[k])
f.close()
'''
