from pyparsing import (
    Literal,
    Char,
    Word,
    Group,
    Forward,
    Optional,
    Combine,
    alphas,
    SkipTo,
    alphanums,
    Regex,
    ParseException,
    CaselessKeyword,
    Suppress,
    delimitedList,
)
import math
import operator
import os
import json
# import pyparsing as pp

exprStack = []

def push_first(toks):
    exprStack.append(toks[0])


def push_unary_minus(toks):
    for t in toks:
        if t == "-":
            exprStack.append("unary -")
        else:
            break

class Parser:
    def __init__(self):
        self.grammar = None
        self.expr_stack = []
        self.build_grammar()

    def push_first(self, toks):
        self.expr_stack.append(toks[0])

    def push_unary_minus(self, toks):
        for t in toks:
            if t == "-":
                self.expr_stack.append("unary -")
            else:
                break

    def build_grammar(self):
        e = CaselessKeyword("E")
        pi = CaselessKeyword("PI")
        # Atoms
        fnumber = Regex(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
        ident = Word(alphas, alphanums + "_$") # identifiers
        var = '[' + SkipTo(']') + ']' # var names
        signal = Combine(ident + '[' + SkipTo(']') + ']') # singal names
        names = signal | ident | var

        # Arith Operators
        plus, minus, mult, div = map(Literal, "+-*/")
        lpar, rpar = map(Suppress, "()")
        lsqbr, rsqbr = map(Suppress, "[]")
        addop = plus | minus
        multop = mult | div
        expop = Literal("^")

        # Relational Operators
        lt, gt = map(Literal, "<>")
        le = Combine(lt + Optional('='))
        ge = Combine(gt + Optional('='))
        eq = Literal("==")
        df = Literal("!=")
        relationalop = le | ge | lt | gt | le | ge | eq | df

        # Build Expr
        expr = Forward()
        expr_list = delimitedList(Group(expr))
        
        # add parse action that replaces the function identifier with a (name, number of args) tuple
        def insert_fn_argcount_tuple(t):
            fn = t.pop(0)
            num_args = len(t[0])
            t.insert(0, (fn, num_args))

        fn_call = (ident + lpar - Group(expr_list) + rpar).setParseAction(
            insert_fn_argcount_tuple
        )
        atom = (
            addop[...]
            + (
                (fn_call | pi | e | fnumber | names).setParseAction(self.push_first)
                | Group(lpar + expr + rpar)
            )
        ).setParseAction(self.push_unary_minus)

        # by defining exponentiation as "atom [ ^ factor ]..." instead of "atom [ ^ atom ]...", we get right-to-left
        # exponents, instead of left-to-right that is, 2^3^2 = 2^(3^2), not (2^3)^2.
        factor = Forward()
        factor <<= atom + (expop + factor).setParseAction(self.push_first)[...]
        term = factor + (multop + factor).setParseAction(self.push_first)[...]
        arith_expr = term + (addop + term).setParseAction(self.push_first)[...]
        expr <<= arith_expr + (relationalop + arith_expr).setParseAction(self.push_first)[...]
        self.grammar = expr

    def parse_formula(self, str_form: str):
        return self.grammar.parseString(str_form)

def main():
    # interval_s=And(s>0, s<10)
    # conditions_s=And(signal_4[s]<50, signal_2[s]>=-(15.27))
    # z3solver.add(Not(ForAll([s], Implies(interval_s, conditions_s))))
    # z3solver.add(Not(ForAll([s], Implies(And(s>0, s<10), And(signal_4[s]<50, signal_2[s]>=-(15.27))))))
    test1 = 'ForAll Timestamp t In [11,50]: '
    test2 = 'And(err[ToInt(RealVal(0)+(t-0.0)/10000.0)]<=0.7, err[ToInt(RealVal(0)+(t-0.0)/10000.0)]>=-(0.7))'
    # interval_t=And(11<=t, t<=50)
    # conditions_t=And(err[ToInt(RealVal(0)+(t-0.0)/10000.0)]<=0.7, err[ToInt(RealVal(0)+(t-0.0)/10000.0)]>=-(0.7))
    # z3solver.add(Not(ForAll([t], Implies(And(11<=t, t<=50), And(err[ToInt(RealVal(0)+(t-0.0)/10000.0)]<=0.7, err[ToInt(RealVal(0)+(t-0.0)/10000.0)]>=-(0.7))))))
    test3 = 'Not(ForAll([t], Implies(And(11<=t, t<=50), And(err[ToInt(RealVal(0)+(t-0.0)/10000.0)]<=0.7, err[ToInt(RealVal(0)+(t-0.0)/10000.0)]>=-(0.7)))))'
    interval_s="And(s>0, s<10)"
    conditions_s="And(signal_4[s]<50, signal_2[s]>=-(15.27))"
    test = 'ForAll([s], Implies(interval_s, conditions_s))'
    form = 'Not(ForAll([s], Implies(And(s>0, s<10), And(signal_4[s]<50, signal_2[s]>=-(15.27)))))'
    # Define the grammar for the HLS language using pyparsing
    # LBRACE, RBRACE = map(pp.Literal, "[]")
    # LPARAM, RPARAM = map(pp.Suppress, "()")

    e = CaselessKeyword("E")
    pi = CaselessKeyword("PI")
    # fnumber = Combine(Word("+-"+nums, nums) +
    #                    Optional("." + Optional(Word(nums))) +
    #                    Optional(e + Word("+-"+nums, nums)))
    # or use provided pyparsing_common.number, but convert back to str:
    # fnumber = ppc.number().addParseAction(lambda t: str(t[0]))
    fnumber = Regex(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
    ident = Word(alphas, alphanums + "_$") # identifiers
    var = Combine('[' + SkipTo(']') + ']') # var names
    # var = '[' + Word(alphanums) + ']' # var namesSkipTo
    signal = Combine(ident + var) # singal names
    names = signal | ident | var
    print(repr(fnumber.parseString('-5.3e10')))
    print(repr(var.parseString('[s]')))
    print(repr(signal.parseString('test2[s]')))
    print(repr(signal.parseString('test2[s+1]')))
    print(repr(signal.parseString('test2[s+(1)]')))
    print(repr(signal.parseString('err[ToInt(RealVal(0)+(t-0.0)/10000.0)]')))
    # print(repr(names.parseString('test2')))

    plus, minus, mult, div = map(Literal, "+-*/")
    lpar, rpar = map(Suppress, "()")
    lsqbr, rsqbr = map(Suppress, "[]")
    addop = plus | minus
    multop = mult | div
    expop = Literal("^")
    lt, gt = map(Literal, "<>")
    le = Combine(lt + Optional('='))
    ge = Combine(gt + Optional('='))
    eq = Literal("==")
    df = Literal("!=")

    relationalop = le | ge | lt | gt | le | ge | eq | df
    ops = addop | relationalop

    expr = Forward()
    expr_list = delimitedList(Group(expr))
    # add parse action that replaces the function identifier with a (name, number of args) tuple
    def insert_fn_argcount_tuple(t):
        fn = t.pop(0)
        num_args = len(t[0])
        t.insert(0, (fn, num_args))

    fn_call = (ident + lpar - Group(expr_list) + rpar).setParseAction(
        insert_fn_argcount_tuple
    )
    atom = (
        addop[...]
        + (
            (fn_call | pi | e | fnumber | names).setParseAction(push_first)
            | Group(lpar + expr + rpar)
        )
    ).setParseAction(push_unary_minus)

    # by defining exponentiation as "atom [ ^ factor ]..." instead of "atom [ ^ atom ]...", we get right-to-left
    # exponents, instead of left-to-right that is, 2^3^2 = 2^(3^2), not (2^3)^2.
    factor = Forward()
    factor <<= atom + (expop + factor).setParseAction(push_first)[...]
    term = factor + (multop + factor).setParseAction(push_first)[...]
    arith_expr = term + (addop + term).setParseAction(push_first)[...]
    expr <<= arith_expr + (relationalop + arith_expr).setParseAction(push_first)[...]
    bnf = expr
    # function = (pp.Word(pp.alphas) + LPARAM + (args('args')) + RPARAM )
    # expression = function

    print(repr(bnf.parseString('a>=b')))
    print(repr(bnf.parseString(interval_s)))
    print(repr(bnf.parseString(conditions_s)))
    print(repr(bnf.parseString(test)))
    print(repr(bnf.parseString('err[ToInt(RealVal(0)+(t-0.0)/10000.0)]>=2')))
    print(repr(bnf.parseString(form)))
    print((bnf.parseString(test2)))
    print(repr(bnf.parseString(test3)))
    # print(repr(expression.parseString(test)))

if __name__ == '__main__':
    main()
