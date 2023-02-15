import pytest

import json
from ga_hls.treenode import parse, get_terminators, bfs, dfs

class TestProperty:
    ## interval_s=And(s>0, s<10)
    ## conditions_s=And(signal_4[s]<1000, signal_2[s]>=-(15.27))
    ## formula = (Not(ForAll([s], Implies(And(s>0, s<10), (And(signal_4[s]<1000, signal_2[s]>=-(15.27)))))))
    def test_hls_v1(self):
        form = json.loads('["ForAll",[["s"],["Implies",[["And",[[">",["s",0]],["<",["s",10]]]],["And",[["<",["signal_4(s)",1000]],[">=",["signal_2(s)",-15.27]]]]]]]]')

        t = parse(form)

        assert t is not None
        assert t.count_elements() is not None

    def test_hls_v2(self):
        form = json.loads('["ForAll",[["s"],["Implies",[["Or",[[">",["s",0]],["<",["s",10]]]],["And",[["<",["signal_4(s)",1000]],[">=",["signal_2(s)",-15.27]]]]]]]]')

        t = parse(form)

        assert t is not None
        assert t.count_elements() is not None

    def test_arith(self):
        form = json.loads('[">",[["+",["x",2]],2]]')

        t = parse(form)

        assert t is not None
        assert t.count_elements() is not None

    def test_arith_logic(self):
        form = json.loads('["Or",[["+",["x",10]],["<",["x",20]]]]')

        t = parse(form)

        assert t is not None
        assert t.count_elements() is not None
