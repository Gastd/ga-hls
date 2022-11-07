import pytest

import json
from ga_hls.treenode import parse

class TestProperty:
    ## interval_s=And(s>0, s<10)
    ## conditions_s=And(signal_4[s]<1000, signal_2[s]>=-(15.27))
    ## formula = (Not(ForAll([s], Implies(And(s>0, s<10), (And(signal_4[s]<1000, signal_2[s]>=-(15.27)))))))
    def test_sigma(self):
        form = json.loads('["Not",["ForAll",[["s"],["Implies",[["And",[[">",["s",0]],["<",["s",10]]]],["And",[["<",["signal_4(s)",1000]],[">=",["signal_2(s)",-15.27]]]]]]]]]')

        t = parse(form)

        assert t is None
        assert t.count_elements() is None