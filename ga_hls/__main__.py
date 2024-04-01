import json
import treenode
from individual import Individual
from ga import GA
import defs
import shlex
import sys
import subprocess

def check_if_sat(filepath):
    ret = False
    folder_name = 'ga_hls'
    run_str = f'python3 {filepath}'
    run_tk = shlex.split(run_str)
    lines = []
    try:
        run_process = subprocess.run(run_tk,
                                     stderr=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     universal_newlines=True,
                                     timeout=50)

        if run_process.stdout.find('SATISFIED') > 0:
            ret = True
        elif run_process.stdout.find('VIOLATED') > 0:
            ret = False
        else:
            ret = False
    except:
        ret = False
    return ret

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
form5 = json.loads('["ForAll",[["t"],["Implies",[["And",[[">",["t",0]],["<",["t",15930000]]]],["And",[[">", ["d2obs[ToInt(RealVal(0)+(t-0.0)/10000.0)]",0.05]], ["<", [["-", ["des_x[ToInt(RealVal(0)+(t-0.0)/10000.0)]","cur_x[ToInt(RealVal(0)+(t-0.0)/10000.0)]"]], 0.02]]]]]]]]')



example = json.loads('["ForAll", ["s", ["Implies", [["And", [[">", ["s",0]], ["<", ["s",18] ] ]], ["And", [["Implies", [ ["==", ["signal_5[s]",0]], ["==", ["signal_6[s]",0]] ]], ["Implies", [ ["==", ["signal_5[s]",1]], ["==", ["signal_6[s]",0]] ]]]]]]]]')
example1= json.loads('["ForAll", ["t", ["Implies", [["And", [["<=", [0,"t"]], ["<=", ["t",6.525E7] ] ]], [">", ["d_aramis_porthos[ToInt(RealVal(0)+(t-0.0)/25000.0)]",0.5]] ]] ]]')
example2= json.loads('["ForAll", ["t", ["Implies", [["And", [[">", ["t",11]], ["<", ["t",50]] ]], ["And", [["<=", ["err[t]",0.7]], [">=", ["err[t]",-0.7]]]]]]]]')
experiment1=json.loads('["ForAll", ["t", ["Implies", [["And", [[">", ["t",11]], ["<", ["t",50]] ]], ["And", [["<=", ["err[ToInt(RealVal(0)+(t-0.0)/10000.0)]",0.007]], [">=", ["err[ToInt(RealVal(0)+(t-0.0)/10000.0)]",-0.007]]]]]]]]')



# ForAll([t], Implies(And(0<=t, t<=(20*1000000)), (v_speed[ToInt(RealVal(0)+(t-0.0)/10000.0)])<120))
at1  = json.loads('["ForAll", ["t",  ["Implies", [["And", [["<=",[0,"t"]], ["<=",["t",20000000]] ]], ["<",["v_speed[ToInt(RealVal(0)+(t-0.0)/10000.0)]",120]]]]]]')
# ForAll([t], Implies(And(0<=t, t<=(10*1000000)), (e_speed[ToInt(RealVal(0)+(t-0.0)/10000.0)])<4750))
at2  = json.loads('["ForAll", ["t",  ["Implies", [["And", [["<=",[0,"t"]], ["<=",["t",10000000]] ]], ["<",["e_speed[ToInt(RealVal(0)+(t-0.0)/10000.0)]",4750]]]]]]')

# z3solver.add(Not(ForAll([t1], Implies(And(1<=t1, t1<=(30000000)),                                          ForAll([i], Implies(And(i>=0, i<ToInt(RealVal(0)+((30*1000000)-0.0)/10000.0)),                               Implies(And(gear[(i-1)]!=1, gear[i]==1), ForAll([t2],                                             Implies(And(t1<=t2, t2<=t1+(2500000)), gear[ToInt(RealVal(0)+(t2-0.0)/10000.0)]==1)))))))))
at51 = json.loads('["ForAll", ["t1", ["Implies", [["And", [["<=", [0,"t1"]], ["<=", ["t1", 30000000]] ] ], ["ForAll", ["i", ["Implies", [["And", [[">=", ["i",0]], ["<", ["i","ToInt(RealVal(0)+((30*1000000)-0.0)/10000.0)"]] ] ], ["Implies", [["And", [["!=", ["gear[(i-1)]",1]], ["==", ["gear[i]",1]]]], ["ForAll", ["t2", ["Implies", [["And",[["<=", ["t1","t2"]], ["<=", ["t2",["+", ["t1",2500000]]]]]], ["==", ["gear[ToInt(RealVal(0)+(t2-0.0)/10000.0)]", 1]]] ]]] ]]]]]]]] ] ]')
# z3solver.add(Not(ForAll([t1],           Implies(And(1<=t1, t1<=(30000000)),                              ForAll([i], Implies(And(i>=1, i<ToInt(RealVal(0)+((30*1000000)-0.0)/10000.0)),                                         Implies(And(gear[(i-1)]!=2, gear[i]==2),                                     ForAll([t2], Implies(And(t1<=t2, t2<=t1+((2500000))),                                                       gear[ToInt(RealVal(0)+(t2-0.0)/10000.0)]==2)))))))))
at52 = json.loads('["ForAll", ["t1", ["Implies", [["And", [["<=", [0,"t1"]], ["<=", ["t1", 30000000]] ] ], ["ForAll", ["i", ["Implies", [["And", [[">=", ["i",0]], ["<", ["i","ToInt(RealVal(0)+((30*1000000)-0.0)/10000.0)"]]] ], ["Implies", [["And",[["!=", ["gear[(i-1)]", 2]], ["==", ["gear[i]", 2]]]], ["ForAll", ["t2", ["Implies", [["And", [["<=", ["t1","t2"]], ["<=", ["t2", ["+",["t1",2500000]]]]]], ["==", ["gear[ToInt(RealVal(0)+(t2-0.0)/10000.0)]", 2]]]]]] ]]]]]]]]]]')
#  z3solver.add(Not(ForAll([t1],       Implies(And(0<=t1,                      t1<=(30000000)),                 ForAll([i],        Implies(And(i>=0, i<ToInt(RealVal(0)+((30*1000000)-0.0)/10000.0)),                           Implies(And(gear[(i-1)]!=3, gear[i]==3),                                         ForAll([t2], Implies(And(t1<=t2, t2<=t1+(2500000)),                                                       gear[ToInt(RealVal(0)+(t2-0.0)/10000.0)]==3)))))))))
at53 = json.loads('["ForAll", ["t1", ["Implies", [["And", [["<=", [0,"t1"]], ["<=", ["t1", 30000000]]]],   ["ForAll", ["i", ["Implies", [["And", [[">=", ["i",0]], ["<", ["i","ToInt(RealVal(0)+((30*1000000)-0.0)/10000.0)"]]]],   ["Implies", [["And",[["!=", ["gear[(i-1)]", 3]], ["==", ["gear[i]",3]]]], ["ForAll", ["t2", ["Implies", [["And",[["<=", ["t1","t2"]], ["<=", ["t2",["+", ["t1",2500000]]]]]], ["==", ["gear[ToInt(RealVal(0)+(t2-0.0)/10000.0)]", 3]]]]]] ]]]]]]]]]]')
#        z3solver.add(Not(ForAll([t1], Implies(And(0<=t1, t1<=(30000000)),                                 ForAll([i], Implies(And(i>=0, i<ToInt(RealVal(0)+(30-0.0)/10000.0)), Implies(And(gear[(i-1)]!=4, gear[i]==4),                                                                         ForAll([t2], Implies(And(t1<=t2, t2<=t1+(2.5*1000000)),                                                     gear[ToInt(RealVal(0)+(t2-0.0)/10000.0)]==4)))))))))
at54 = json.loads('["ForAll", ["t1", ["Implies", [["And", [["<=", [0,"t1"]], ["<=", ["t1", 30000000]]]],   ["ForAll", ["i", ["Implies", [["And", [[">=", ["i",0]], ["<", ["i","ToInt(RealVal(0)+((30*1000000)-0.0)/10000.0)"]]]], ["Implies", [["And",[["!=", ["gear[(i-1)]", 4]], ["==", ["gear[i]", 4]]]], ["ForAll", ["t2", ["Implies", [["And",[["<=", ["t1","t2"]], ["<=", ["t2",["+", ["t1",2500000]]]]]], ["==", ["gear[ToInt(RealVal(0)+(t2-0.0)/10000.0)]", 4]]]]]] ]]]]]]]]]]')

# z3solver.add(Not(ForAll([t1], Implies(And(0<=t1, t1<=(30000000)),                                         Implies(e_speed[ToInt(RealVal(0)+(t1-0.0)/10000.0)]<3000, ForAll([t2], Implies(And(0<=t2, t2<=(4000000)), v_speed[ToInt(RealVal(0)+(t2-0.0)/10000.0)]<35)))))))
at6a = json.loads('["ForAll", ["t1", ["Implies", [["And", [["<=", [0,"t1"]], ["<=", ["t1", 30000000]]]], ["Implies", [["<",["e_speed[ToInt(RealVal(0)+(t1-0.0)/10000.0)]",3000]], ["ForAll", ["t2", ["Implies", [["And", [["<=", [0,"t2"]], ["<=", ["t2",4000000]]]], ["<", ["v_speed[ToInt(RealVal(0)+(t2-0.0)/10000.0)]", 35]]]]]]]]]]]]')
# z3solver.add(Not(ForAll([t1], Implies(And(0<=t1, t1<=(30000000)),                                        Implies(e_speed[ToInt(RealVal(0)+(t1-0.0)/10000.0)]<3000,                ForAll([t2],                                            Implies(And(0<=t2, t2<=(8000000)), v_speed[ToInt(RealVal(0)+(t2-0.0)/10000.0)]<50))))))) 
at6b = json.loads('["ForAll", ["t1", ["Implies", [["And", [["<=", [0,"t1"]], ["<=", ["t1", 30000000]]]], ["Implies", [["<",["e_speed[ToInt(RealVal(0)+(t1-0.0)/10000.0)]",3000]], ["ForAll", ["t2", ["Implies", [["And", [["<=", [0,"t2"]], ["<=", ["t2",8000000]]]], ["<", ["v_speed[ToInt(RealVal(0)+(t2-0.0)/10000.0)]", 50]]]]]]]]]]]]')
# z3solver.add(Not(ForAll([t1],          Implies(And(0<=t1, t1<=(30000000)),                              Implies(e_speed[ToInt(RealVal(0)+(t1-0.0)/10000.0)]<3000,               ForAll([t2],          Implies(And(0<=t2, t2<=(20000000)), v_speed[ToInt(RealVal(0)+(t2-0.0)/10000.0)]<65)))))))
at6c = json.loads('["ForAll", ["t1", ["Implies", [["And", [["<=", [0,"t1"]], ["<=", ["t1", 30000000]]]], ["Implies", [["<",["e_speed[ToInt(RealVal(0)+(t1-0.0)/10000.0)]",3000]], ["ForAll", ["t2", ["Implies", [["And", [["<=", [0,"t2"]], ["<=", ["t2",20000000]]]], ["<", ["v_speed[ToInt(RealVal(0)+(t2-0.0)/10000.0)]", 65]]]]]]]]]]]]')

# ForAll([t], Implies(And(0<=t, t<=(100*1000000)), ((y5[ToInt(RealVal(0)+(t-0.0)/10000.0)])-(y4[ToInt(RealVal(0)+(t-0.0)/10000.0)]))<=40))
cc1  = json.loads('["ForAll", ["t", ["Implies", [["And", [["<=",[0,"t"]], ["<=",["t",100000000]] ]], ["<=", [ ["-", ["y5[ToInt(RealVal(0)+(t-0.0)/50000.0)]","y4[ToInt(RealVal(0)+(t-0.0)/50000.0)]"]] , 40]] ]] ]]')
# ForAll([t1], Implies(And(0<=t1, t1<=(70*1000000)), conditions_t1))
# Exists([t2], And(And(t1<=t2, t2<=t1+(30*1000000)), ((y5[ToInt(RealVal(0)+(t2-0.0)/10000.0)])-(y4[ToInt(RealVal(0)+(t2-0.0)/10000.0)]))>15))
#cc2  = json.loads('["Exists", ["t2", ["Implies", [["And", [["<=",["t1","t2"]], ["<=",["t2",["+", ["t1", 30000000]] ]] ]], [">", [ ["-", ["y5[ToInt(RealVal(0)+(t2-0.0)/10000.0)]","y4[ToInt(RealVal(0)+(t2-0.0)/10000.0)]"]] , 15]] ]] ]]')
cc2  = json.loads('["ForAll", ["t1", ["Implies", [["And", [["<=", [0, "t1"]], ["<=", ["t1", 70000000]] ]], ["Exists", ["t2", ["Implies", [["And", [["<=",["t1","t2"]], ["<=",["t2",["+", ["t1", 30000000]] ]] ]], [">", [ ["-", ["y5[ToInt(RealVal(0)+(t2-0.0)/50000.0)]","y4[ToInt(RealVal(0)+(t2-0.0)/50000.0)]"]] , 15]] ]] ]]]] ]]')

cc3  = json.loads('["ForAll", ["t1", ["Implies", [["And", [["<=", [0, "t1"]], ["<=", ["t1", 80000000]] ]], ["Or", [["ForAll", ["t3", ["Implies", [["And", [["<=", ["t1", "t3"]], ["<=", ["t3", ["+", ["t1", 20000000]]]] ]], ["<", [["-", ["y2[ToInt(RealVal(0)+(t3-0.0)/100000.0)]", "y1[ToInt(RealVal(0)+(t3-0.0)/100000.0)]"]], 20]] ]]]], ["Exists", ["t2", ["And", [["And", [["<=", ["t2", ["+", ["t1", 20000000]]]], ["<=", ["t1", "t2"]] ]], [">", [["-", ["y5[ToInt(RealVal(0)+(t2-0.0)/100000.0)]", "y4[ToInt(RealVal(0)+(t2-0.0)/100000.0)]"]], 40]] ]]]] ]] ]]]]')

cc4  = json.loads('["ForAll", ["t1", ["Implies", [["And", [["<=", [0, "t1"]], ["<=", ["t1", 65000000]] ]], ["Exists", ["t2", ["And", [["And", [["<=", ["t1", "t2"]], ["<=", ["t2", ["+", ["t1", 30000000]] ]] ]], ["ForAll", ["t3", ["Implies", [["And", [["<=", ["t2", "t3"]], ["<=", ["t3", ["+", ["t2", 5000000]] ]] ]], [">", [["-", ["y5[ToInt(RealVal(0)+(t3-0.0)/100000.0)]", "y4[ToInt(RealVal(0)+(t3-0.0)/100000.0)]" ]], 8 ]] ]] ]] ]] ]] ]] ]]')

cc5  = json.loads('["ForAll", ["t4", ["Implies", [["And", [["<=", [0, "t4"]], ["<=", ["t4", 72000000]] ]], ["Exists", ["t3", ["And", [["And", [["<=", ["t4", "t3"]], ["<=", ["t3", ["+", ["t4", 8000000]] ]] ]], ["Implies", [["ForAll", ["t2", ["Implies", [["And", [["<=", ["t3", "t2"]], ["<=", ["t2", ["+", ["t3", 5000000]] ]] ]], [">", [["-", ["y2[ToInt(RealVal(0)+(t2-0.0)/100000.0)]", "y1[ToInt(RealVal(0)+(t2-0.0)/100000.0)]" ]], 9 ]] ]] ]], ["ForAll", ["t2", ["Implies", [["And", [["<=", [["+", ["t2", 5000000]], "t1"]], ["<=", ["t1", ["+", ["t2", 20000000]] ]] ]], [">", [["-", ["y5[ToInt(RealVal(0)+(t1-0.0)/100000.0)]", "y4[ToInt(RealVal(0)+(t1-0.0)/100000.0)]" ]], 9 ]] ]] ]] ]] ]] ]] ]] ]]')

# z3solver.add(Not(ForAll([t], Implies(And((0000000)<=t, t<=(50000000)), And(And(And(((y5[ToInt(RealVal(0)+(t-0.0)/20000.0)])-(y4[ToInt(RealVal(0)+(t-0.0)/20000.0)]))>7.5, ((y4[ToInt(RealVal(0)+(t-0.0)/20000.0)])-(y3[ToInt(RealVal(0)+(t-0.0)/20000.0)]))>7.5),((y3[ToInt(RealVal(0)+(t-0.0)/20000.0)])-(y2[ToInt(RealVal(0)+(t-0.0)/20000.0)]))>7.5),((y2[ToInt(RealVal(0)+(t-0.0)/20000.0)])-(y1[ToInt(RealVal(0)+(t-0.0)/20000.0)]))>7.5)))))
ccx  = json.loads('["ForAll", ["t", ["Implies", [["And", [["<=", [0, "t"]], ["<=", ["t", 50000000 ]] ]], ["And", [["And", [["And", [[">", [["-", ["y5[ToInt(RealVal(0)+(t-0.0)/50000.0)]", "y4[ToInt(RealVal(0)+(t-0.0)/50000.0)]" ]], 7.5 ]], [">", [["-", ["y4[ToInt(RealVal(0)+(t-0.0)/50000.0)]", "y3[ToInt(RealVal(0)+(t-0.0)/50000.0)]" ]], 7.5 ]] ]], [">", [["-", ["y3[ToInt(RealVal(0)+(t-0.0)/50000.0)]", "y2[ToInt(RealVal(0)+(t-0.0)/50000.0)]" ]], 7.5 ]] ]], [">", [["-", ["y2[ToInt(RealVal(0)+(t-0.0)/50000.0)]", "y1[ToInt(RealVal(0)+(t-0.0)/50000.0)]" ]], 7.5 ]] ]] ]] ]]')



req2form = {
    "AS"  : at1 ,
    "AT1" : at1 ,
    "AT2" : at2 ,
    "AT51": at51,
    "AT52": at52,
    "AT53": at53,
    "AT54": at54,
    "AT6A": at6a,
    "AT6B": at6b,
    "AT6C": at6c,
    "CC1" : cc1 ,
    "CC2" : cc2 ,
    "CC3" : cc3 ,
    "CC4" : cc4 ,
    "CC5" : cc5 ,
    "CCX" : ccx ,
    "form5":form5
}

if True:
    sat = check_if_sat(sys.argv[2])

    if sat:
        print(sat)
        print('Already SATISFIED')
        sys.exit()

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

root1 = treenode.parse(req2form[sys.argv[1]])
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


form = req2form[sys.argv[1]]
print(form)
defs.FILEPATH = sys.argv[2]
defs.FILEPATH2= sys.argv[2]


# run 1
# ga = GA(form, sys.argv[3])
# ga.j48('')
# ga.evolve()
# print("offsprings")
# of[0].print_genes()
# of[1].print_genes()
# print(treenode.parse(form1))
# print(set((treenode.get_terminators(root1))))

# ranges = {
#     '3': ['int',0, 120],
#     '7': ['int', 10000000, 20000000],
#     '11': ['int', 0, 20000000]
# }
# mutations = {
#     '3': ['int', [100, 140]],
#     '7': ['int', [10000000, 30000000]],
#     '11': ['int', [0, 10000000]]
# }
mutations = {
    "3": ["op", [">", "<"]],
    "4": ["float", [0., 1.0]],
    "8": ["op", [">", "<"]],
    "9": ["float", [0., 2.0]]
}
# print(mutations)
f = open(sys.argv[4],)
mutations = json.load(f)
# print(mutations)
# run 2
ga = GA(form, mutations, sys.argv[3])
# ga = GA(form, None, sys.argv[3])
# ga.set_mutation_ranges(mutations)
# ga.set_force_mutations(True)
ga.evolve()
