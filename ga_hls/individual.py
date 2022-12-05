import sys
import math
import random

# from .treenode import 
import treenode

MUTATION_NODES = 1

OPS = ['ForAll', 'And', 'Or', 'Exists', 'Implies', '<', '>', '<=', '>=', '+', '-', '*', '/', '^']
QUANTIFIERS = ['ForAll', 'Exists']
RELATIONALS = ['<', '>', '<=', '>=']
ARITHMETICS = ['+', '-', '*', '/', '^']
LOGICALS = ['Not', 'And', 'Or', 'Implies']

class Individual():
    """docstring for Individual"""
    def __init__(self, formula_root: treenode.Node, terminators):
        if formula_root is None:
            raise Exception(f"{self.__class__.__name__} error: Input formula cannot be empty")

        self.root = formula_root
        self.fitness = -1
        self.term = self.check_terminators(terminators)
        self.maxint, self.minint = self.get_minmax(terminators, int)
        # print(f'maxint = {self.maxint}, minint = {self.minint}')
        self.maxfloat, self.minfloat = self.get_minmax(terminators, float)
        # print(f'maxfloat = {self.maxfloat}, minfloat = {self.minfloat}')
        self.madeit = False

    def get_minmax(self, terminators, type):
        t_list = [x for x in terminators if isinstance(x, type)]
        if len(t_list) == 0:
            return None, None
        else:
            return max(t_list), min(t_list)

    def print_genes(self):
        if self.root is not None:
            print(self.root)
        # print()

    def __repr__(self):
        return str(self.root)

    def __str__(self):
        return str(self.root)

    def mutate(self, rate: float, nmutations=MUTATION_NODES):
        for _ in range(0, nmutations):
            if (random.random() < rate):
                mut_idx = random.randrange(len(self.root))
                # print(f'from {self.root.get_subtree(mut_idx)} ->', end='')
                if self.root.get_subtree(mut_idx).left is None:
                    self.root.get_subtree(mut_idx).value = self.get_new_term(self.root.get_subtree(mut_idx).value)
                else:
                    self.root.get_subtree(mut_idx).value = self.get_new_op(self.root.get_subtree(mut_idx).value)
                # print(f' to {self.root.get_subtree(mut_idx)}')

    def get_new_term(self, t):
        if t.__class__ in self.term.keys():
            if isinstance(t, int):
                if self.minint == self.maxint:
                    return random.randint(self.minint, self.maxint)
                else:
                    minn = 0 if self.minint == 0 else 10 ** int(math.log(self.minint)+1)+1 if self.minint > 0 else -10 ** int(math.log(-self.minint)+1)+1
                    # print(f'minn = {minn}, max= {10 ** int(math.log(self.maxint)+1) -1}')
                    return random.randint(minn, 10 ** int(math.log(self.maxint)+1) -1)
            elif isinstance(t, float):
                return random.uniform(self.minfloat, self.maxfloat)
            else:
                return random.choice(self.term[t.__class__])
        else:
            raise ValueError(f'Unknown terminator of type {t.__class__} from {t}')

    def check_terminators(self, term_list):
        term_dict = {}
        for t in term_list:
            # print(t.__class__.__name__)
            if t.__class__.__name__ in term_dict.keys():
                term_dict[t.__class__] += [t]
            else:
                term_dict[t.__class__]  = [t]
        return term_dict

    def get_new_op(self, op):
        # return random.choice(OPS)
        if op in QUANTIFIERS:
            return random.choice(QUANTIFIERS)
        elif op in RELATIONALS:
            return random.choice(RELATIONALS)
        elif op in ARITHMETICS:
            return random.choice(ARITHMETICS)
        elif op in LOGICALS:
            return random.choice(LOGICALS)
        else:
            raise ValueError(f"Operator not known: {op}")

    def format(self):
        return build_str(self.root)

def build_str(root):
    s = ''
    if root is None:
        return ''
    if root.left is None:
        try:
            s = root.value.replace(')', ']')
            s = s.replace('(', '[')
            return f'{s}'
        except:
            return f'{root.value}'

    if root.value in QUANTIFIERS:
        # print(root.left.value, root.right.value)
        # print(f'{root.value}([{root.left.value}], {build_str(root.right)})')
        return f'{root.value}([{root.left.value}], {build_str(root.right)})'
    elif root.value in LOGICALS:
        return f'{root.value}({build_str(root.left)}, {build_str(root.right)})'
    elif root.value in RELATIONALS:
        return f'({build_str(root.left)} {root.value} {build_str(root.right)})'
    elif root.value in ARITHMETICS:
        return f'({build_str(root.left)} {root.value} {build_str(root.right)})'