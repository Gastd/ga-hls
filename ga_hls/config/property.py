import json

form1 = json.loads('["ForAll",[["s"],["Implies",[["And",[[">",["s",0]],["<",["s",10]]]],["And",[["<",["signal_4(s)",1000]],[">=",["signal_2(s)",-15.27]]]]]]]]')
form2 = json.loads('["ForAll",[["s"],["Implies",[["Or",[[">",["s",0]],["<",["s",10]]]],["And",[["<",["signal_4(s)",1000]],[">=",["signal_2(s)",-15.27]]]]]]]]')
form3 = json.loads('[">",[["+",["x",2]],2]]')
form4 = json.loads('["Or",[["+",["x",10]],["<",["x",20]]]]')
form1 = json.loads('["ForAll",[["s"],["Implies",[["And",[[">",["s",0]],["<",["s",10]]]],["And",[["<",["signal_4(s)",50]],[">=",["signal_2(s)",-15.27]]]]]]]]')
form5 = json.loads('["ForAll",[["t"],["Implies",[["And",[[">",["t",0]],["<",["t",1.593E7]]]],["And",[[">", ["d2obs[ToInt(RealVal(0)+(t-0.0)/10000.0)]",0.5]], ["<", [["-", ["des_x[ToInt(RealVal(0)+(t-0.0)/10000.0)]","cur_x[ToInt(RealVal(0)+(t-0.0)/10000.0)]"]], 0.2]]]]]]]]')
example = json.loads('["ForAll", ["s", ["Implies", [["And", [[">", ["s",0]], ["<", ["s",18] ] ]], ["And", [["Implies", [ ["==", ["signal_5[s]",0]], ["==", ["signal_6[s]",1]] ]], ["Implies", [ ["==", ["signal_5[s]",1]], ["==", ["signal_6[s]",0]] ]]]]]]]]')

property_str = form1