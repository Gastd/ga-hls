import json
from collections import deque
import numpy as np

from . import treenode

form1 = json.loads('["ForAll",[["s"],["Implies",[["And",[[">",["s",0]],["<",["s",10]]]],["And",[["<",["signal_4(s)",50]],[">=",["signal_2(s)",-15.27]]]]]]]]')
form4 = json.loads('["Or",[["+",["x",10]],["<",["x",20]]]]')
root = treenode.parse(form1)
# ir = node_depth_first_iter(root)
nform1 = np.array(form1)
nform1 = nform1.reshape((-1,1,2))
print(nform1)
root1 = treenode.parse(form1)
# print(repr(next(ir)))
# print(repr(next(ir)))
# print(repr(next(ir)))
# print(repr(next(ir)))
# print(repr(next(ir)))
# print(repr(next(ir)))
# print(repr(next(ir)))
# print(repr(next(ir)))
# print(repr(next(ir)))
# print(repr(next(ir)))
# print(repr(next(ir)))
# print(repr(next(ir)))
# print(repr(next(ir)))
# print(next(root))
# print()
print(list(root1))
# print(list(ir))