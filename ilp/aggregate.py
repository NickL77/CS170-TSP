import os
import shutil

input_dir = '../inputs/'
output_dir = '../outputs/'
optimal_dir = output_dir + 'optimal/'
suboptimal_dir = output_dir + 'suboptimal/'
aggregate_dir = output_dir + 'aggregate/'
dict = {}
for f in os.listdir(optimal_dir):
    shutil.copyfile(optimal_dir+f, aggregate_dir+f)

for f in os.listdir(suboptimal_dir):
    old_name = f
    new_name = ""
    temp = 0
    for i in range(len(f)):
        if f[i] == '_':
            if temp == 1:
                new_name = f[:i]
                print(dict.get(new_name, False))
                if not dict.get(new_name, False):
                    dict[new_name] = old_name
                    print(dict.get(new_name, False))
                else:
                    prev_gap = int(dict[new_name][i+5:-4])
                    this_gap = int(old_name[i+5:-4])
                    print(prev_gap, this_gap)
                    if prev_gap >= this_gap:
                        os.remove(suboptimal_dir+dict[new_name])
                        dict[new_name] = old_name
                    else:
                        os.remove(suboptimal_dir+old_name)
                        old_name = dict[new_name]

                break
            temp += 1
    shutil.copyfile(suboptimal_dir+old_name, aggregate_dir+new_name+".out")
