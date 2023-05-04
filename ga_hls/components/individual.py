import sys
import math
import shlex
import random
import subprocess

from components import treenode
from components.helper import get_file_w_traces, save_check_wo_traces

MUTATION_NODES = 5

# OPS = ['ForAll', 'And', 'Or', 'Exists', 'Implies', '<', '>', '<=', '>=', '+', '-', '*', '/', '^']
QUANTIFIERS = ['ForAll', 'Exists']
RELATIONALS = ['<', '>', '<=', '>=', '==']
EQUAL = ['=']
ARITHMETICS = ['+', '-']
MULDIV = ['*', '/']
EXP = ['^']
LOGICALS = ['And', 'Or']
NEG = ['Not']   # PROBLEM
IMP = ['Implies']

def show_arg_ret_decorator(function):
    def wrapper(self, *args, **kwargs):
        ret = function(self, *args, **kwargs)
        print(f'arg = {args[0]}, ret = {ret}')
        return ret
    return wrapper

class Individual():
    """docstring for Individual"""
    def __init__(self, formula_root: treenode.Node, terminators, trace_file):
        if formula_root is None:
            raise Exception(f"{self.__class__.__name__} error: Input formula cannot be empty")

        self.trace_file = trace_file
        self.root = formula_root
        self.fitness = -1
        self.term = self.check_terminators(terminators)
        self.madeit = False
        self.sw_score = -1
        self.self_test = None
        # self.maxint, self.minint = self.get_minmax(terminators, int)
        # self.maxfloat, self.minfloat = self.get_minmax(terminators, float)

    # def get_minmax(self, terminators, type):
    #     t_list = [x for x in terminators if isinstance(x, type)]
    #     if len(t_list) == 0:
    #         return None, None
    #     else:
    #         return max(t_list), min(t_list)

    def is_viable(self):
        start, end, lines = get_file_w_traces(self.trace_file)
        save_check_wo_traces(start, end, lines, f'Not({self.format()})')
        f = open(self.trace_file, 'r')
        f.seek(0, 0)
        f.close()

        folder_name = 'ga_hls'
        run_str = f'python3 {folder_name}/z3check.py'
        run_tk = shlex.split(run_str)
        try:
            run_process = subprocess.run(run_tk,
                                         stderr=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         universal_newlines=True,
                                         timeout=10)
        except:
            return False
        # print(run_process.stdout)
        if run_process.stdout.find('SATISFIED') > 0:
            # print('Chromosome not viable')
            return False
        elif run_process.stdout.find('VIOLATED') > 0:
            # print('Chromosome viable')
            return True
        else:
            # print('Chromosome not viable')
            return False
        return

    def reset(self):
        self.fitness = -1

    def print_genes(self):
        if self.root is not None:
            print(self.root)
        # print()

    def __eq__(self, other):
        return str(self) == str(other)

    def __iter__(self):
        return iter(self.root)

    def __repr__(self):
        return repr(self.root)

    def __str__(self):
        return readable(self.root)

    def __len__(self):
        return len(self.root)

    def mutate(self, rate: float, nmutations=MUTATION_NODES):
        for _ in range(0, nmutations):
            if (random.random() < rate):
                mut_idx = random.randrange(len(self.root))
                subtree, parent = self.root.get_subtree(mut_idx)
                if subtree.left is None:
                    subtree.value = self.get_new_term(subtree.value)
                else:
                    new_operator = self.get_new_op(subtree.value)
                    if parent:
                        if subtree.value == 'Implies' and (parent.value in QUANTIFIERS):
                            # print(f'Found Implies from Quantifier: {subtree.value}, parent = {parent.value}')
                            continue
                    subtree.value = new_operator

    # @show_arg_ret_decorator
    def get_new_term(self, t):
        # print(f'For terminator t = {t}: {self.print_genes()}')
        if t.__class__ in self.term.keys():
            if isinstance(t, int):
                if t == 0:
                    if random.random() > 0.5:
                        return t + 1
                    else:
                        return t - 1
                if random.random() > 0.5:
                    ## Change the terminator inside the same maginitude order from the input
                    return t + random.randint(10 ** int(math.log(abs(t),10)), 10 ** int(math.log(abs(t),10)+1)-1)
                else:
                    return t - random.randint(10 ** int(math.log(abs(t),10)), 10 ** int(math.log(abs(t),10)+1)-1)
            elif isinstance(t, float):
                if random.random() > 0.5:
                    return t + random.uniform(10 ** int(math.log(abs(t),10)), 10 ** int(math.log(abs(t),10)+1)-1)
                else:
                    return t - random.uniform(10 ** int(math.log(abs(t),10)), 10 ** int(math.log(abs(t),10)+1)-1)
            else:
                # return random.choice(self.term[t.__class__])
                return t
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
        if op in RELATIONALS:
            return random.choice(RELATIONALS)
        elif op in ARITHMETICS:
            return random.choice(ARITHMETICS)
        elif op in LOGICALS:
            return random.choice(LOGICALS)
        elif op in QUANTIFIERS:
            return random.choice(QUANTIFIERS)
        else:
            # raise ValueError(f"Operator not known: {op}")
            return op

    def format(self):
        return build_str(self.root)

def readable(root):
    s = ''
    if root is None:
        return ''
    if root.left is None:
        if isinstance(root.value, float):
            return f'{root.value:.2f}'
        else:
            return f'{root.value}'

    if root.value in QUANTIFIERS:
        return f'{root.value} {readable(root.left)} In ({readable(root.right.left)}) Implies ({readable(root.right.right)})'
    elif root.value in LOGICALS+IMP+NEG:
        return f'{readable(root.left)} {root.value} {readable(root.right)}'
    elif root.value in RELATIONALS:
        return f'{readable(root.left)} {root.value} {readable(root.right)}'
    elif root.value in ARITHMETICS+MULDIV+EXP:
        return f'{readable(root.left)} {root.value} {readable(root.right)}'

def build_str(root):
    s = ''
    if root is None:
        return ''
    if root.left is None:
        try:
            # s = root.value#.replace(')', ']')
            s = root.value.replace(')', ']')
            s = s.replace('(', '[')
            return f'{s}'
        except:
            return f'{root.value}'

    if root.value in QUANTIFIERS:
        # print(root.left.value, root.right.value)
        # print(f'{root.value}([{root.left.value}], {build_str(root.right)})')
        return f'{root.value}([{root.left.value}], {build_str(root.right)})'
    elif root.value in LOGICALS+IMP+NEG:
        return f'{root.value}({build_str(root.left)}, {build_str(root.right)})'
    elif root.value in RELATIONALS:
        return f'({build_str(root.left)} {root.value} {build_str(root.right)})'
    elif root.value in ARITHMETICS:
        return f'({build_str(root.left)} {root.value} {build_str(root.right)})'