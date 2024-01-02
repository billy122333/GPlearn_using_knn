import pickle
import numpy as np
from sympy import solve
from gplearn_1230.functions import _Function

with open("knn_data.pkl", "rb") as pklfile:
    read = pickle.load(pklfile)
    # print(read)
    # print(len(read))
# print(len(read))
# print(read[-1])

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
converter = {'sub' : lambda x, y : x - y,
            'div' : lambda x, y : solve(x, y),
            'mul' : lambda x, y : x * y,
            'add' : lambda x, y : x + y}

X_train = np.arange(-1, 1, 0.01).reshape(200, 1)
y_train = X_train**3 + X_train**2 + X_train  
for i in range(len(read)):
    subtrees = read[i][0].program[read[i][1]:read[i][2]]
    subtrees = mystr(subtrees)
    # print(mystr(subtrees))
    # subtrees =', '.join(subtrees)
    y_data = []
    for j in range(200):
        X0 = X_train[j][0]
        y = eval(subtrees)
        y_data.append(y)
    y_avg = np.average(np.array(y_data))
    read[i].append(y_avg)
    
# print(read[0])
# print(len(read))
# for p, s, e in enumerate(read):
# [[],[],[]]
# print(type(read))
# for i in range(len(read)):
    # print(read[i])
    # print(read[i][0])
    # print(type(read[i][0]))
    # print((read[i][0].program[read[i][1]:read[i][2]]).execute(X_train))
    # print(read[i][0].execute(X_train))

sorted_read = sorted(read, key= lambda p: p[3])
# print(sorted_read[0:5])