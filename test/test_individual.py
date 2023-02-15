import pytest

import json
from ga_hls.treenode import parse, get_terminators
from ga_hls.individual import Individual, QUANTIFIERS, RELATIONALS, ARITHMETICS, LOGICALS

class TestIndividual:
    def test_creation(self):
        treeroot = parse(json.loads('[">",[["+",["x",2]],2]]'))
        term = list(set(get_terminators(treeroot)))
        ind1 = Individual(treeroot, term)

        assert ind1 is not None

    def test_minmax(self):
        pass

    def test_new_terminal(self):
        pass

    def test_check_terminal(self):
        pass

    def test_mutations(self):
        treeroot = parse(json.loads('[">",[["+",["x",2]],2]]'))
        term = list(set(get_terminators(treeroot)))
        ind1 = Individual(treeroot, term)
        ind1.mutate(1)
        assert ind1 is not None
