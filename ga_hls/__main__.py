import json
import treenode
from individual import Individual
from ga import GA
import defs
import sys

# from parsing import Parser

## interval_s=And(s>0, s<10)
## conditions_s=And(signal_4[s]<1000, signal_2[s]>=-(15.27))
## formula = (Not(ForAll([s], Implies((And(s>0, s<10)), (And(signal_4[s]<1000, signal_2[s]>=-(15.27)))))))
# treenode.parse(form1) = json.loads('["Not",["ForAll",[["s"],["Implies",[["And",[[">",["s",0]],["<",["s",10]]]],["And",[["<",["signal_4(s)",1000]],[">=",["signal_2(s)",-15.27]]]]]]]]]')
# form1 = json.loads('["ForAll",[["s"],["Implies",[["And",[[">",["s",0]],["<",["s",10]]]],["And",[["<",["signal_4(s)",1000]],[">=",["signal_2(s)",-15.27]]]]]]]]')
form2 = json.loads('["ForAll",[["s"],["Implies",[["Or",[[">",["s",0]],["<",["s",10]]]],["And",[["<",["signal_4(s)",1000]],[">=",["signal_2(s)",-15.27]]]]]]]]')
form3 = json.loads('[">",[["+",["x",2]],2]]')
form4 = json.loads('["Or",[["+",["x",10]],["<",["x",20]]]]')
form1 = json.loads('["ForAll",[["s"],["Implies",[["And",[[">",["s",0]],["<",["s",10]]]],["And",[["<",["signal_4(s)",50]],[">=",["signal_2(s)",-15.27]]]]]]]]')
form5 = json.loads('["ForAll",[["t"],["Implies",[["And",[[">",["t",0]],["<",["t",1.593E7]]]],["And",[[">", ["d2obs[ToInt(RealVal(0)+(t-0.0)/10000.0)]",0.5]], ["<", [["-", ["des_x[ToInt(RealVal(0)+(t-0.0)/10000.0)]","cur_x[ToInt(RealVal(0)+(t-0.0)/10000.0)]"]], 0.2]]]]]]]]')



example = json.loads('["ForAll", ["s", ["Implies", [["And", [[">", ["s",0]], ["<", ["s",18] ] ]], ["And", [["Implies", [ ["==", ["signal_5[s]",0]], ["==", ["signal_6[s]",0]] ]], ["Implies", [ ["==", ["signal_5[s]",1]], ["==", ["signal_6[s]",0]] ]]]]]]]]')
example1= json.loads('["ForAll", ["t", ["Implies", [["And", [["<=", [0,"t"]], ["<=", ["t",6.525E7] ] ]], [">", ["d_aramis_porthos[ToInt(RealVal(0)+(t-0.0)/25000.0)]",0.5]] ]] ]]')
example2= json.loads('["ForAll", ["t", ["Implies", [["And", [[">", ["t",11]], ["<", ["t",50]] ]], ["And", [["<=", ["err[t]",0.7]], [">=", ["err[t]",-0.7]]]]]]]]')
experiment1=json.loads('["ForAll", ["t", ["Implies", [["And", [[">", ["t",11]], ["<", ["t",50]] ]], ["And", [["<=", ["err[ToInt(RealVal(0)+(t-0.0)/10000.0)]",0.007]], [">=", ["err[ToInt(RealVal(0)+(t-0.0)/10000.0)]",-0.007]]]]]]]]')

# ForAll([t], Implies(And(0<=t, t<=(50*1000000)), (v_speed[ToInt(RealVal(0)+(t-0.0)/10000.0)])<120))
at = json.loads('["ForAll", ["t", ["Implies", [["And", [["<=",[0,"t"]], ["<=",["t",50000000]] ]], ["<",["v_speed[ToInt(RealVal(0)+(t-0.0)/10000.0)]",120]]]]]]')

### TODO: TEST NEW AST PARSER ##########################################################
# conditions_s="And(signal_4[s]<50, signal_2[s]>=-(15.27))"
# parser = Parser()
# print(parser.parse_formula(conditions_s))

# sys.exit()
### TODO: TEST NEW AST PARSER ##########################################################


# form5 = json.loads('["ForAll",[["s"],["Implies",[["And",[[">",["s",0]],["<",["s",18]]]],["And",[["==",["signal_5(s)",0]],["==",["signal_6(s)",1]]]]]]]]')
# ["Or",[["+",["x",10]],["<",["x",20]]]]
# form1 = json.loads('ex.json')

# print(len(form1[1][1][1][1][1]))
# print(    form1[1][1][1][1][1][1][0][1])

root1 = treenode.parse(at)
# root2 = treenode.parse(form2)

# print(root1)
# print(root1.count_elements())
# print(root2.count_elements())
# print(root1.get_random_subtree())
# print(root2.get_random_subtree())

# print ("\nInorder traversal of binary tree is")
# treenode.printInorder(root1)
# treenode.dfs(root1)
# print('')



vertices, parents = treenode.bfs(root1)
for i, v in enumerate(vertices):
	print(i, v.value)
print('')
for i, p in enumerate(parents):
	if p is not None:
		print(i, p.value)

# x = input("Choose what vertices to mutate, chose -1 for cancel: ([1,2,3])\n")
# x = list(x)
# l = [int(a) for a in x if a != ',']
# x.remove(',')
# print(l)

# t1 = list(set(treenode.get_terminators(treenode.parse(form5))))
# ind1 = Individual(treenode.parse(form5), t1)
# ind1.print_genes()
# ind1.force_mutate(l)
# print("parents")
# ind1.print_genes()


# ind2 = Individual(treenode.parse(form4))
# ind2.print_genes()
# print(ind2.root.cut_tree_random())

# ga = GA(form5)
# ga = GA(form1)
# ga = GA(example)



defs.FILEPATH = sys.argv[1]
defs.FILEPATH2= sys.argv[1]


# run 1
ga = GA(at)
ga.evolve()
# print("offsprings")
# of[0].print_genes()
# of[1].print_genes()
# print(treenode.parse(form1))
# print(set((treenode.get_terminators(root1))))

ranges = {
	'4': ['float',-1100, -0.007],
	'7': ['float', 0.007, 1100]
}

# run 2
# ga = GA(experiment1)
# ga.set_mutation_ranges(ranges)
# ga.set_force_mutations(True)
# ga.evolve()
