
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


def parse(l: list):
    tree = None

    print(f'formula: {(l)}')
    if not isinstance(l, list):
        tree =  Node(l)
        return tree

    print(f'formula length: {len(l)}')
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
    else:
        tree = Node(l)

    # if not isinstance(l[1], list):
    #     print(f'formula: {(l[1].__class__)}')
    #     print(f'')
    #     tree =  Node('')
    #     tree.left = Node(l[0])
    #     tree.right = Node(l[1])
    #     return tree

    # if isinstance(l[0], str):
    #     tree = Node(l[0])
    #     if len(l[1]) == 1:
    #         tree.left = parse(l[1][0])
    #     elif len(l[1]) == 2:
    #         tree.left = parse(l[1][0])
    #         tree.right = parse(l[1][1])

    return tree
