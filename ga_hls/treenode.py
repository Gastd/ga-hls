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

def strcmp(str1, str2):
    c1 = 0
    c2 = 0
    l = len(str1) if len(str1) < len(str2) else len(str2)
    for i in range(l):
        c1 += ord(str1[i])
        c2 += ord(str2[i])
    return c1-c2

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

    # def __sub__(self, other):
    #     try:
    #         if isinstance(self.value, str) and isinstance(other.value, str):
    #             return strcmp(self.value, other.value)
    #         elif isinstance(self.value, str):
    #             return ord(self.value[0]) - other.value
    #         elif isinstance(other.value, str):
    #             return self.value - ord(other.value[0])
    #         else:
    #             return self.value - other.value
    #     except:
    #         pass

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        if isinstance(self.value, float):
            # return f'{repr(self.value)}'
            return f'{repr(self.value):.2f}'
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
