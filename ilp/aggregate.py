import os
import shutil

input_dir = '../inputs/'
output_dir = '../outputs/'
optimal_dir = output_dir + 'optimal/'
suboptimal_dir = output_dir + 'suboptimal/'
aggregate_dir = output_dir + 'aggregate/'

for f in os.listdir(optimal_dir):
    shutil.copyfile(optimal_dir+f, aggregate_dir+f)

for f in os.listdir(suboptimal_dir):
    new_name = ""
    temp = 0
    for i in range(len(f)):
        if f[i] == '_':
            if temp == 1:
                new_name = f[:i]
                break
            temp += 1
    shutil.copyfile(suboptimal_dir+f, aggregate_dir+new_name+".out")
