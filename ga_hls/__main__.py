import json
import treenode
from individual import Individual
from ga import GA

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



example = json.loads('["ForAll", ["s", ["Implies", [["And", [[">", ["s",0]], ["<", ["s",18] ] ]], ["And", [["Implies", [ ["==", ["signal_5[s]",0]], ["==", ["signal_6[s]",1]] ]], ["Implies", [ ["==", ["signal_5[s]",1]], ["==", ["signal_6[s]",0]] ]]]]]]]]')





# form5 = json.loads('["ForAll",[["s"],["Implies",[["And",[[">",["s",0]],["<",["s",18]]]],["And",[["==",["signal_5(s)",0]],["==",["signal_6(s)",1]]]]]]]]')
# ["Or",[["+",["x",10]],["<",["x",20]]]]
# form1 = json.loads('ex.json')

# print(len(form1[1][1][1][1][1]))
# print(    form1[1][1][1][1][1][1][0][1])

# root1 = treenode.parse(form5)
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
# v, p = treenode.bfs(root1)

# print(v)
# print(p)

# ind1 = Individual(treenode.parse(form3))
# ind2 = Individual(treenode.parse(form4))
# ind1.mutate()
# print("parents")
# ind1.print_genes()
# ind2.print_genes()
# print(ind2.root.cut_tree_random())

# ga = GA(form5)
# ga = GA(form1)
ga = GA(example)
# ga.write_population(0)
ga.evolve()
# print("offsprings")
# of[0].print_genes()
# of[1].print_genes()
# print(treenode.parse(form1))
# print(set((treenode.get_terminators(root1))))
