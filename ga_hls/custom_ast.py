import random
from collections import deque
from pyparsing import ParseResults
from parsing import Parser
from collections import deque


import treenode
import json

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def isint(num):
    try:
        int(num)
        return True
    except ValueError:
        return False

class AST(object):
    """docstring for AST"""
    def __init__(self, value):
        super(AST, self).__init__()
        self.value = value if not isfloat(value) else float(value) if not isint(value) else int(value)
        self.childs = []

    def __repr__(self):
        if isinstance(self.value, float):
            return f'{self.value:.2f}'
        else:
            if len(self.childs) == 0:
                return f'{repr(self.value)}'
            else:
                return f'[{repr(self.value)}, {(self.childs)}]'

    def count_elements(self):
        nel = 0
        if self.left is not None:
            nel += self.left.count_elements()
        if self.right is not None:
            nel += self.right.count_elements()

        return 1 + nel

def build_ast(l: ParseResults):
    tree = None

    if isinstance(l, str):
        tree = AST(l)
        return tree

    if len(l) == 1:
        if isinstance(l[0], ParseResults):
            tree = build_ast(l[0])
        else:
            tree = AST(l[0])
    elif len(l) == 2:
        childs = []
        # get first element as operator
        tree = AST(l[0][0])
        # build an AST for each child
        for child in l[1]:
            childs.append(build_ast(child))
        tree.childs = childs
    else:
        # first el is child
        childs = [build_ast(l[0])]
        # second el is an operator
        tree = AST(l[1])
        # build an AST for each child
        childs.append(build_ast(l[2:]))
        tree.childs = childs

    return tree

def test():
    forms = ['And(a,b)',
             'And(2,3+2+1)',
             'Not(And(c,d))',
             '1+2+4+3',
             '1+2',
             '1+2+4+3 == 0',
             'And((Or(b,c))>=2,3+2)',
             'And((Or(b,c))>=2,(3+2) > 5.0)',   # left to right needs params to define order
             'And(2>=Or(b,c),(3+2) > 5.0)',      # does not need params
             'And(2>=(Or(b,c)),(3+2) > 5.0)',
             "And(signal_4[s]<50, signal_2[s]>=-(15.27))"
             ]
    for f in forms:
        parser = Parser()
        # print(parser.parse_formula(f))
        print(build_ast(parser.parse_formula(f)))


if __name__ == '__main__':
    test()
