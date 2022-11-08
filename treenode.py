import random
from collections import deque

class Node(object):
    """docstring for Node"""
    def __init__(self, value):
        super(Node, self).__init__()
        self.value = value
        self.left = None
        self.right = None

    def count_elements(self):
        nel = 0
        if self.left is not None:
            nel += self.left.count_elements()
        if self.right is not None:
            nel += self.right.count_elements()

        return 1 + nel

    def __repr__(self):
        if (self.left is None) and (self.right is None):
            return f'{repr(self.value)}'
        else:
            return f'[{repr(self.value)}, [{repr(self.left)},{repr(self.right)}]]'

    def __str__(self):
        if self.left is None:
            left = ''
        else:
            left = str(self.left)

        if self.right is None:
            right = ''
        else:
            right = str(self.right)

        return left + ' ' + str(self.value) + ' ' + right
        return f'{left} {self.value} {right}'

    def __len__(self):
        return self.count_elements()

    def __iadd__(self, other):
        self.merge(other)

    def get_subtree(self, node_idx: int):
        # print(bfs(self))
        # print(len(bfs(self)[0]))
        return bfs(self)[0][node_idx]

    def get_random_subtree(self):
        return random.choice(bfs(self)[0][1:])

    def get_parent_node(self, node):
        queue = deque([self])

        while len(queue) > 0:
            cur_node = queue.pop()
            # print(cur_node.value)
            if (cur_node.left is node) or (cur_node.right is node):
                return cur_node
            if cur_node.left is not None:
                queue.append(cur_node.left)

            if cur_node.right is not None:
                queue.append(cur_node.right)

        return None

    def cut_tree_random(self):
        # print(self)
        subtree = self.get_random_subtree()
        # print(subtree)
        childs, parents = bfs(self)
        # print('childs = ', childs)
        # print('parents = ', parents)
        child_idx = childs.index(subtree)
        parent = self.get_parent_node(subtree)
        # print('child_idx = ', child_idx)
        # print('len(childs) = ', len(childs))
        # print('childs[6] = ', childs[6])
        # print('parents[6] = ', self.get_parent_node(subtree).value)

        if parent is not None:
            if parent.left is subtree:
                parent.left = None
            elif parent.right is subtree:
                parent.right = None
            else:
                raise ValueError("Tree cut failed")

        return self, subtree, parent

    def merge(self, tree):
        if self.left is None:
            self.left = tree
        elif self.right is None:
            self.right = tree
        else:
            raise ValueError("Tree merge failed")

def bfs(root):
    if root is None:
        return
    queue = deque([root])
    vertices = []
    parents = []
    i = 0
    old_node = None

    while len(queue) > 0:
        cur_node = queue.pop()
        # print(cur_node.value)
        if cur_node.left is not None:
            queue.append(cur_node.left)

        if cur_node.right is not None:
            queue.append(cur_node.right)

        vertices.append(cur_node)
        parents.append(old_node)
        old_node = cur_node

    return vertices, parents

def dfs(root):
    if root is None:
        return

    dfs(root.left)
    print(root.value, end=' ')
    dfs(root.right)


def parse(l: list):
    tree = None

    # print(f'formula: {(l)}')
    if not isinstance(l, list):
        tree =  Node(l)
        return tree

    # print(f'formula length: {len(l)} {l}')
    if len(l) == 2:
        if isinstance(l[0], str):
            tree = Node(l[0])
        elif isinstance(l[0], list):
            tree = parse(l[0])
        else:
            raise ValueError('marlformed formula')

        if isinstance(l[1], list):
            if len(l[1]) == 1:
                tree.left = parse(l[1][0])
            elif len(l[1]) == 2:
                tree.left = parse(l[1][0])
                tree.right = parse(l[1][1])
    elif len(l) == 1:
        tree = Node(l[0])
    else:
        tree = Node(l)

    return tree

def get_terminators(node):
    if node.left is None and node.right is None:
        if isinstance(node.value, list):
            return [i for i in node.value]
        else:
            return [node.value]
    else:
        left = get_terminators(node.left)
        right = get_terminators(node.right)
        return left + right

# A function to do inorder tree traversal
def printInorder(root):
 
    if root:
 
        # First recur on left child
        printInorder(root.left)
 
        # then print the data of node
        print(root.value),
 
        # now recur on right child
        printInorder(root.right)
