import sys
import math
import shlex
import random
import subprocess

# from .treenode import 
import treenode
import defs

MUTATION_NODES = 5
MUT_IDXS =[4, 7]

# OPS = ['ForAll', 'And', 'Or', 'Exists', 'Implies', '<', '>', '<=', '>=', '+', '-', '*', '/', '^']
QUANTIFIERS = ['ForAll', 'Exists']
RELATIONALS = ['<', '>', '<=', '>=']
EQUALS = ['==', '!=']
ARITHMETICS = ['+', '-', '*', '/']
MULDIV = ['*', '/']
EXP = ['^']
LOGICALS = ['And', 'Or']
NEG = ['Not']   # PROBLEM
IMP = ['Implies']
FUNC = ['ToInt']

def show_arg_ret_decorator(function):
    def wrapper(self, *args, **kwargs):
        ret = function(self, *args, **kwargs)
        print(f'arg = {args[0]}, ret = {ret}')
        return ret
    return wrapper

def check_call(function):
    def wrapper(self, *args, **kwargs):
        print(f'{function} called')
        ret = function(self, *args, **kwargs)
        return ret
    return wrapper

def choose_mutation(function):
    def wrapper(self, *args, **kwargs):
        print(f'{self.mutations}')
        ret = function(self, *args, **kwargs)
        return ret
    return wrapper

class Individual():
    """docstring for Individual"""
    def __init__(self, formula_root: treenode.Node, terminators):
        if formula_root is None:
            raise Exception(f"{self.__class__.__name__} error: Input formula cannot be empty")

        self.root = formula_root
        self.fitness = -1
        self.term = self.check_terminators(terminators)
        self.madeit = False
        self.sw_score = -1
        self.self_test = None
        self.mutations = None
        # self.maxint, self.minint = self.get_minmax(terminators, int)
        # self.maxfloat, self.minfloat = self.get_minmax(terminators, float)

    # def get_minmax(self, terminators, type):
    #     t_list = [x for x in terminators if isinstance(x, type)]
    #     if len(t_list) == 0:
    #         return None, None
    #     else:
    #         return max(t_list), min(t_list)

    def copy_temp_file(self, folder_path):
        lines = []
        with open(defs.FILEPATH2,'r') as file:
            for l in file:
                lines.append(l)

        with open(f'{folder_path}/z3check.py','w') as file:
            for l in lines:
                file.write(l)
        return f'{folder_path}/z3check.py'

    def is_viable(self, temp_path):
        def get_file_w_traces(file_path = defs.FILEPATH2):
            s = e = -1
            lines = []
            with open(file_path) as f:
                for l in f:
                    lines.append(l)
            # print('lines')
            for idx, l in enumerate(lines):
                if l.find('z3solver.add') >= 0:
                    # print(idx, l)
                    s = idx
                    break
            for idx, l in enumerate(lines):
                # print(l, l.find('z3solver.check'))
                if l.find('z3solver.check') >= 0:
                    # print(idx, l)
                    e = idx
                    break
            # print('lines')
            return s, e, lines

        def save_check_wo_traces(start, end, lines, nline, file_path = 'ga_hls/z3check.py'):
            before = lines[:start]
            after = lines[end:]
            with open(file_path,'w') as z3check_file:
                for l in before:
                    z3check_file.write(l)
                form_line = (f'\tz3solver.add({nline})\n')
                z3check_file.write('\n')
                # print(form_line)
                z3check_file.write(form_line)
                z3check_file.write('\n')
                for l in after:
                    z3check_file.write(l)

        def reset_file(path):
            f = open(path, 'r')
            f.seek(0, 0)
            f.close()

        new_file = self.copy_temp_file(temp_path)
        start, end, lines = get_file_w_traces(new_file)
        save_check_wo_traces(start, end, lines, f'Not({self.format()})', new_file)
        reset_file(defs.FILEPATH2)
        reset_file(new_file)

        # folder_name = 'ga_hls'
        folder_name = temp_path
        run_str = f'python3 {folder_name}/z3check.py'
        run_tk = shlex.split(run_str)
        try:
            run_process = subprocess.run(run_tk,
                                         stderr=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         universal_newlines=True,
                                         timeout=100)
        except:
            return False
        # print(run_process.stdout)
        if run_process.stdout.find('SATISFIED') > 0:
            # print('Chromosome not viable')
            return True
        elif run_process.stdout.find('VIOLATED') > 0:
            # print('Chromosome viable')
            return True
        else:
            # print('Chromosome not viable')
            return False
        return True

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
        return arrf(self.root)

    def __len__(self):
        return len(self.root)

    def arrf_str(self):
        return arrf(self.root)

    # @check_call
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

    # @check_call
    def force_mutate(self, mut_idxs=[], nmutations=MUTATION_NODES):
        if len(mut_idxs) == 0:
            print('MUT_IDX is empty')
            return
        # print(self.mutations)
        if self.mutations is not None:
            self.force_mutate_with_ranges(mut_idxs)
            return
        else:
            raise ValueError(f'{__class__}.mutations is {self.mutations}')
            rate = 1.0
            for mut_idx in mut_idxs:
                if (random.random() < rate):
                    # mut_idx = random.choice(mut_idxs)
                    subtree, parent = self.root.get_subtree(mut_idx)
                    # print(f"Mutating {mut_idx} from {subtree.value}")
                    if subtree.left is None:
                        subtree.value = self.get_new_term(subtree.value)
                    else:
                        new_operator = self.get_new_op(subtree.value)
                        if subtree.value in QUANTIFIERS:
                            subtree.right.value = 'Implies' if new_operator == 'ForAll' else 'And'
                        if parent:
                            if subtree.value == 'And'     and (parent.value == 'Exists'):
                                continue
                            if subtree.value == 'Implies' and (parent.value == 'ForAll'):
                                # print(f'Found Implies from Quantifier: {subtree.value}, parent = {parent.value}')
                                continue
                        subtree.value = new_operator
                    # print(f"To {subtree.value}")

    def show_idx(self):
        for idx in range(0, len(self)):
            subtree, parent = self.root.get_subtree(idx)
            print(f"{idx} = {subtree.value}")

    # @show_arg_ret_decorator
    def force_mutate_with_ranges(self, mut_idxs=[], nmutations=MUTATION_NODES):
        rate = 1.0
        for idx in mut_idxs:
            if (random.random() < rate):
                subtree, parent = self.root.get_subtree(int(idx))
                # print(f"{self}")
                # print(f"Mutating {idx} from {subtree.value} with {self.mutations[str(idx)]}")
                if subtree.left is None:
                    subtree.value = self.get_new_forced_term(subtree.value, self.mutations[str(idx)])
                else:
                    new_operator = random.choice(self.mutations[str(idx)][1])
                    if subtree.value in QUANTIFIERS:
                        subtree.right.value = 'Implies' if new_operator == 'ForAll' else 'And'
                    #     if subtree.value == 'Implies' and (parent.value in QUANTIFIERS):
                    #         # print(f'Found Implies from Quantifier: {subtree.value}, parent = {parent.value}')
                    #         continue
                    subtree.value = new_operator
                # print(f"To {subtree.value}")

    def get_new_forced_term(self, t, mutation):
        interval = mutation[1]
        if mutation[0] == 'int':
            lower = int(interval[0])
            upper = int(interval[1])
        if mutation[0] == 'float':
            lower = float(interval[0])
            upper = float(interval[1])
        # print(f'For term {t} we have interval {lower}, {upper}')
        if mutation[0] == 'int':
            ## Change the terminator inside the same maginitude order from the input
            ret = random.randint(lower, upper)
            # print(ret)
            return ret
        elif mutation[0] == 'float':
            ret = random.uniform(lower, upper)
            # print(ret)
            return ret
        else:
            ret = random.choice(interval)
            # print(ret)
            return ret

    # @show_arg_ret_decorator
    def get_new_term(self, t):
        # print(f'For terminator t = {t}: {self.print_genes()}')
        if t.__class__ in self.term.keys():
            if isinstance(t, int):
                if t == 0:
                    if random.random() > 0.5:
                        return +1
                    else:
                        return -1
                if random.random() > 0.5:
                    ## Change the terminator inside the same maginitude order from the input
                    return t + random.randint(10 ** int(math.log(abs(t),10)), 10 ** int(math.log(abs(t),10)+1)-1)
                else:
                    return t - random.randint(10 ** int(math.log(abs(t),10)), 10 ** int(math.log(abs(t),10)+1)-1)
            elif isinstance(t, float):
                if abs(t) >= 1.0:
                    diff = random.uniform(10 ** int(math.log(abs(t),10)), 10 ** int(math.log(abs(t),10)+1))
                else:
                    diff = random.uniform(0, 10 ** int(math.log(abs(t),10)))
                if random.random() > 0.5:
                    return t + diff
                else:
                    return t - diff
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
        if op in FUNC:
            return random.choice(FUNC)
        elif op in RELATIONALS:
            return random.choice(RELATIONALS)
        elif op in ARITHMETICS:
            return random.choice(ARITHMETICS)
        elif op in LOGICALS:
            return random.choice(LOGICALS)
        elif op in EQUALS:
            return random.choice(EQUALS)
        elif op in QUANTIFIERS:
            return random.choice(QUANTIFIERS)
        else:
            # raise ValueError(f"Operator not known: {op}")
            return op

    def format(self):
        return build_str(self.root)

def readable(root):
    # print('\n')
    # print(root.__class__)
    # print(f'{root}')
    # print(f'{root.value}')
    # print(f'{root.left},{root.value},{root.right}')
    s = ''
    if root is None:
        return ''
    if root.left is None:
        if isinstance(root.value, float):
            return f'{root.value:.6f}'
        else:
            return f'{root.value}'

    if root.value in QUANTIFIERS:
        return f'{root.value} {readable(root.left)} In ({readable(root.right.left)}) {root.right.value} ({readable(root.right.right)})'
    elif root.value in LOGICALS+IMP+NEG:
        return f'{readable(root.left)} {root.value} {readable(root.right)}'
    elif root.value in RELATIONALS+EQUALS:
        return f'{readable(root.left)} {root.value} {readable(root.right)}'
    elif root.value in ARITHMETICS+MULDIV+EXP:
        return f'{readable(root.left)} {root.value} {readable(root.right)}'
    elif root.value in FUNC:
        return f'{readable(root.value)}({root.left}/{readable(root.right)})'

def arrf(root):
    # print('\n')
    # print(root.__class__)
    # print(f'{root}')
    # print(f'{root.value}')
    # print(f'{root.left},{root.value},{root.right}')
    s = ''
    if root is None:
        return ''
    if root.left is None:
        if isinstance(root.value, float):
            return f'{root.value:.6f}'
        else:
            return f'{root.value}'

    if root.value in QUANTIFIERS:
        return f'{root.value},{arrf(root.right.left)},{arrf(root.right.right)}'
    elif root.value in LOGICALS+IMP+NEG:
        return f'{arrf(root.left)},{root.value},{arrf(root.right)}'
    elif root.value in RELATIONALS+EQUALS:
        return f'{arrf(root.left)},{root.value},{arrf(root.right)}'
    elif root.value in ARITHMETICS+MULDIV+EXP:
        return f'{arrf(root.left)},{root.value},{arrf(root.right)}'
    elif root.value in FUNC:
        return f'{arrf(root.left)},{root.value},{arrf(root.right)}'

def build_str(root):
    s = ''
    if root is None:
        return ''
    if root.left is None:
        try:
            s = root.value#.replace(')', ']')
            # s = root.value.replace(')', ']')
            # s = s.replace('(', '[')
            return f'{s}'
        except:
            return f'{root.value}'

    if root.value in QUANTIFIERS:
        # print(root.left.value, root.right.value)
        # print(f'{root.value}([{root.left.value}], {build_str(root.right)})')
        return f'{root.value}([{root.left.value}], {build_str(root.right)})'
    elif root.value in LOGICALS+IMP+NEG:
        return f'{root.value}({build_str(root.left)}, {build_str(root.right)})'
    elif root.value in RELATIONALS+EQUALS:
        return f'({build_str(root.left)} {root.value} {build_str(root.right)})'
    elif root.value in ARITHMETICS+MULDIV+EXP:
        return f'({build_str(root.left)} {root.value} {build_str(root.right)})'
    elif root.value in FUNC:
        if root.value == "ToInt":
            return f'({root.value}(RealVal(0)+{build_str(root.left)}/{build_str(root.right)}))'
