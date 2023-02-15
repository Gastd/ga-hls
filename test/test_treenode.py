import pytest

import json
from ga_hls.treenode import parse, get_terminators, bfs, dfs

class TestProperty:

    def test_terminators(self):
        form = json.loads('["Or",[["+",["x",10]],["<",["x",20]]]]')
        t = get_terminators(parse(form))

        assert t is not None

    def test_dfs(self):
        form = json.loads('["Or",[["+",["x",10]],["<",["x",20]]]]')
        dfs(parse(form))

    def test_bfs(self):
        form = json.loads('["Or",[["+",["x",10]],["<",["x",20]]]]')
        bfs(parse(form))

    def test_get_parent(self):
        pass

    def test_subtree(self):
        pass


    def test_cut_tree(self):
        pass

    def test_merge(self):
        pass
