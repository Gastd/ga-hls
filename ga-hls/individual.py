import sys
import math
import random
import subprocess

from . import treenode
from .lang.ast import (
    ForAll,
    Exists,
    And,
    Or,
    Implies,
    RelOp,
    IntConst,
    RealConst,
    Var,
    Subscript,
    FuncCall,
    ArithOp,
    Not,
)
from .lang.python_printer import formula_to_python_expr
from .lang.ast import Formula
from .lang.internal_encoder import InternalEncodeError, formula_to_internal_obj
from .mutation import MutationConfig, mutate_formula

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
    def __init__(self, formula_root: treenode.Node, terminators, formula_ast=None):
        if formula_root is None:
            raise Exception(f"{self.__class__.__name__} error: Input formula cannot be empty")

        self.root = formula_root
        self.ast: Optional[Formula] = formula_ast
        self.fitness = -1
        self.term = self.check_terminators(terminators)
        self.madeit = 'Unknown'
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
    def _sync_root_from_ast(self):
        """
        Refresh the legacy treenode representation (`self.root`) from the AST.
        """
        if self.ast is None:
            return

        internal = formula_to_internal_obj(self.ast)
        self.root = treenode.parse(internal)

    def reset(self):
        self.fitness = -1
        self.madeit = 'Unknown'

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

    def arrf_str(self) -> str:
        """
        Return a flat, comma-separated representation of the current formula
        used for ARFF generation.
        """
        # Use AST if we have it
        formula = getattr(self, "ast", None)
        if formula is None:
            return None

        tokens: list[str] = []

        def visit(node):
            # Quantifiers
            if isinstance(node, ForAll):
                tokens.append("ForAll")
                # ForAll vars are something like ["t"]
                for v in node.vars:
                    tokens.append(str(v))
                visit(node.body)

            elif isinstance(node, Exists):
                tokens.append("Exists")
                for v in node.vars:
                    tokens.append(str(v))
                visit(node.body)

            # Logical connectives
            elif isinstance(node, And):
                tokens.append("And")
                for arg in node.args:
                    visit(arg)

            elif isinstance(node, Or):
                tokens.append("Or")
                for arg in node.args:
                    visit(arg)

            elif isinstance(node, Implies):
                tokens.append("Implies")
                visit(node.left)
                visit(node.right)

            elif isinstance(node, Not):
                tokens.append("Not")
                visit(node.arg)

            # Relational operators
            elif isinstance(node, RelOp):
                # left op right, e.g., target_y[t], >, 0.5
                visit(node.left)
                tokens.append(node.op)
                visit(node.right)

            # Arithmetic expressions (we inline them and let TERM/NUM domains handle them)
            elif isinstance(node, ArithOp):
                visit(node.left)
                visit(node.right)

            # Numeric constants
            elif isinstance(node, IntConst):
                tokens.append(str(int(node.value)))

            elif isinstance(node, RealConst):
                tokens.append(f"{float(node.value):.6f}")

            # Terms
            elif isinstance(node, Var):
                tokens.append(node.name)

            elif isinstance(node, Subscript):
                # Serialize as something like target_y[t], ego_y[t+1], etc.
                tokens.append(formula_to_python_expr(node))

            elif isinstance(node, FuncCall):
                # e.g. Abs((ego_y[t]-target_y[t]))
                tokens.append(formula_to_python_expr(node))

            else:
                # Fallback: just stringify it
                tokens.append(str(node))

        visit(formula)
        return ",".join(tokens)


        # @check_call
    def mutate(self, rate: float, mutation_config: MutationConfig | None = None):
        """
        Default mutation operator.

        Parameters
        ----------
        rate : float
            Probability in [0, 1] that this individual is mutated when this
            method is called. If rate <= 0, no mutation is applied. If
            rate >= 1, mutation is always applied.
        mutation_config : MutationConfig | None
            Configuration for AST-level mutation. If None, this is a no-op.
        """
        # No AST or no config → nothing to do
        if self.ast is None or mutation_config is None:
            return

        # Normalize the rate
        try:
            r = float(rate)
        except (TypeError, ValueError):
            # Backwards-compatible fallback: if something weird is passed,
            # behave like "always mutate"
            r = 1.0

        if r <= 0.0:
            # Mutation disabled
            return
        elif r < 1.0:
            # Probabilistic mutation
            if random.random() >= r:
                return
        # else: r >= 1.0 → always mutate

        # AST-based mutation path
        self.ast = mutate_formula(self.ast, mutation_config)

        # Keep the legacy tree representation in sync
        self._sync_root_from_ast()
   
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
