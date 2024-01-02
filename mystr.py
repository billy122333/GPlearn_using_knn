
from gplearn_1230.functions import _Function


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