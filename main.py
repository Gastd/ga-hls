import json
import treenode

## interval_s=And(s>0, s<10)
## conditions_s=And(signal_4[s]<1000, signal_2[s]>=-(15.27))
## formula = (Not(ForAll([s], Implies(And(s>0, s<10), (And(signal_4[s]<1000, signal_2[s]>=-(15.27)))))))
form = json.loads('["Not",["ForAll",[["s"],["Implies",[["And",[[">",["s",0]],["<",["s",10]]]],["And",[["<",["signal_4(s)",1000]],[">=",["signal_2(s)",-15.27]]]]]]]]]')
# form = json.loads('ex.json')

# print(len(form[1][1][1][1][1]))
# print(    form[1][1][1][1][1][1][0][1])

t = treenode.parse(form)

print(t)
print(t.count_elements())