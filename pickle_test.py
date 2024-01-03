import tqdm
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
from gplearn_1230.functions import _Function

with open("knn_data.pkl", "rb") as pklfile:
    pickle_trees = pickle.load(pklfile)

# print(pickle_trees)
# print(len(pickle_trees))
    
def mystr(program):
    """Overloads `print` output of the object to resemble a LISP tree."""
    terminals = [0]
    output = ''
    for i, node in enumerate(program):
        if isinstance(node, _Function):
            terminals.append(node.arity)
            output += node.name + '('
        else:
            if isinstance(node, int):
                output += 'X%s' % node
            else:
                output += '%.3f' % node
            terminals[-1] -= 1
            while terminals[-1] == 0:
                terminals.pop()
                terminals[-1] -= 1
                output += ')'
            if i != len(program) - 1:
                output += ', '
    return output

def add(x, y):
        return x + y
def sub(x, y):
    return x - y
def mul(x, y):
    return x * y
def div(x, y):
    if abs(float(y)) <= 1e-3:
        return 1.0
    return x / y

X_train = np.arange(-1, 1, 0.01).reshape(200, 1)
y_data_list = []
for i in tqdm(range(len(pickle_trees))):
    subtrees = pickle_trees[i][0].program[pickle_trees[i][1]:pickle_trees[i][2]]
    subtrees = mystr(subtrees)
    y_data = []
    for j in range(200):
        X0 = X_train[j][0]
        y = eval(subtrees)
        y_data.append(y)
    # print(f'y vector: {y_data}')
    y_data_list.append(y_data)
    # y_avg = np.average(np.array(y_data))
    # pickle_trees[i].append(y_avg)

k = 5
for i in tqdm(range(len(y_data_list))):
    mae_temp = []
    for j in range(i+1, len(y_data_list)):
        # y_diff = [abs(yi - yj) for yi, yj in zip(y_data_list[i], y_data_list[j])]
        mae = np.mean(np.abs(np.array(y_data_list[i]) - np.array(y_data_list[j])))
        mae_temp.append([j ,mae])
    sorted_mae = sorted(mae_temp, key=lambda x: x[1])
    pickle_tree_index = [index[0] for index in sorted_mae]
    # print(i, pickle_tree_index)
    pickle_trees_neighbors = [pickle_trees[k] for k in pickle_tree_index]
    # print(pickle_trees_neighbors)
    pickle_trees[i] += [item for sublist in pickle_trees_neighbors for item in sublist]
    # print(len(pickle_trees[i]))
    # print(pickle_trees[i][0:3*(k+1)])
print(len(pickle_trees))