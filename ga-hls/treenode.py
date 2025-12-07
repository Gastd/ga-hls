import random
from collections import deque

def node_depth_first_iter(node):
    stack = deque([node])
    while stack:
        # Pop out the first element in the stack
        node = stack.popleft()
        yield node
        # push children onto the front of the stack.
        # Note that with a deque.extendleft, the first on in is the last
        # one out, so we need to push them in reverse order.
        children = []
        if node.left:
            children.append(node.left)
        if node.right:
            children.append(node.right)
        stack.extendleft(reversed(children))

def compare_tree(root1, root2, level=0):
    MISMATCH = 5

    if root1.left is None and root1.right is None:
        return 0

    l = compare_tree(root1.left, root2.left, level+1)
    r = compare_tree(root1.right, root2.right, level+1)
    node_match = 0

    if root1.value != root2.value:
        node_match = MISMATCH*2**level

    return l + r + node_match

class Node(object):
    """docstring for Node"""
    def __init__(self, value):
        super(Node, self).__init__()
        self.value = value
        self.left = None
        self.right = None
        # self.iterator = InorderIterator(self)

    def count_elements(self):
        nel = 0
        if self.left is not None:
            nel += self.left.count_elements()
        if self.right is not None:
            nel += self.right.count_elements()

        return 1 + nel

    def __iter__(self):
        return node_depth_first_iter(self)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return self.value == other
        else:
            return self.value == other.value

    def __repr__(self):
        if isinstance(self.value, float):
            # return f'{repr(self.value)}'
            return f'{self.value:.2f}'
        else:
            return f'{repr(self.value)}'
        # if (self.left is None) and (self.right is None):
        #     return f'{repr(self.value)}'
        # else:
        #     return f'[{repr(self.value)}, [{repr(self.left)},{repr(self.right)}]]'

    def __str__(self):
        if self.left is None:
            left = ''
        else:
            left = f'{str(self.left)}\t'

        if self.right is None:
            right = ''
        else:
            right = f'\t{str(self.right)}'

        if isinstance(self.value, float):
            return f'{left}{self.value:.2f}{right}'
        else:
            return f'{left}{self.value}{right}'

    def __len__(self):
        return self.count_elements()

    def __iadd__(self, other):
        self.merge(other)

    def get_subtree(self, node_idx: int):
        if node_idx >= len(self):
            raise Exception("Node idx greater than or equal tree length node_idx >= len(self): {} > {}"\
                .format(node_idx, len(self)))
        subtrees, parents = bfs(self)
        # print(f'get_subtree: {subtrees}, {parents}')
        # print(f'get_subtree: {subtrees[node_idx]}, {parents[node_idx]}')
        return subtrees[node_idx], parents[node_idx]

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

    def cut_tree(self, idx):
        if idx >= len(self):
            raise Exception(f"Index out of range: {idx} >= {len(self)}")
        subtree = self.get_subtree(idx)
        childs, parents = bfs(self)
        parent = self.get_parent_node(subtree)

        if parent is not None:
            if parent.left is subtree:
                parent.left = None
            elif parent.right is subtree:
                parent.right = None
            else:
                raise Exception("Tree cut failed")
            return self, subtree, parent
        else:
            raise Exception(f"Tree cut failed: {self}, {subtree}, {parent}")

    def cut_tree_random(self):
        subtree = self.get_random_subtree()
        childs, parents = bfs(self)
        child_idx = childs.index(subtree)
        parent = self.get_parent_node(subtree)

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
    """
    Recursively collect terminal node values from a treenode-based formula.
    """
    if node is None:
        return []

    # Leaf node
    if getattr(node, "left", None) is None and getattr(node, "right", None) is None:
        return [node.value]

    left_terms = get_terminators(node.left) if getattr(node, "left", None) is not None else []
    right_terms = get_terminators(node.right) if getattr(node, "right", None) is not None else []

    return left_terms + right_terms

# A function to do inorder tree traversal
def printInorder(root):
    if root:
        # First recur on left child
        printInorder(root.left)
        # then print the data of node
        print(root.value),
        # now recur on right child
        printInorder(root.right)
