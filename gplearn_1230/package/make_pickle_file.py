import pickle
import numpy as np
from sympy import solve
from .._program import pickle_trees
from ..functions import _Function


# turn gp expression into a readable expression
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

def make_pickle_file(index):
    # print("start")
    X_train = np.arange(-1, 1, 0.01).reshape(200, 1)
    # y_train = X_train**3 + X_train**2 + X_train  
    for i in range(len(pickle_trees)):
        subtrees = pickle_trees[i][0].program[pickle_trees[i][1]:pickle_trees[i][2]]
        # print(subtrees)
        subtrees = mystr(subtrees)
        # print(f'subtrees: {subtrees}')
        y_data = []
        # print(X_train.shape) # (200, 1)
        for j in range(200):
            X0 = X_train[j][0]
            # print(f'X0: {X0}')
            # print(type(X0))
            y = eval(subtrees)
            # print(f'y: {y}')
            # print(type(y))
            y_data.append(y)
        y_avg = np.average(np.array(y_data))
        pickle_trees[i].append(y_avg)   

    sorted_pickle_trees = sorted(pickle_trees, key= lambda p: p[3])

    with open (f"knn_data{index}.txt", "w") as f:
        for i in range(len(sorted_pickle_trees)):
            f.write(f"{mystr(sorted_pickle_trees[i][0].program)}, {sorted_pickle_trees[i][1]}, {sorted_pickle_trees[i][2]}\n")

    with open(f"knn_data{index}.pkl", "wb") as pklfile:
        print("len(sorted_pickle_trees): ", len(sorted_pickle_trees))
        pickle.dump(sorted_pickle_trees, pklfile)