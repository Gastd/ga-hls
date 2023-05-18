# Z3Py CODE: 
from z3 import *
import time
def property_04():
	start_time=time.time()
	z3solver=Solver() 

	#signal_5 contained in the file

	#signal_6 contained in the file
	s=Int('s') 

	#Trace: property_04
	timestamps=Array('timestamps', RealSort(), IntSort())
	signal_5=Array('signal_5', RealSort(), IntSort())
	signal_6=Array('signal_6', RealSort(), IntSort())
	z3solver.add(timestamps[ 0]==0)
	z3solver.add(signal_5[0]==0.0)
	z3solver.add(signal_6[0]==1.0)
	z3solver.add(timestamps[ 1]==7211012000)
	z3solver.add(signal_5[1]==0.0)
	z3solver.add(signal_6[1]==1.0)
	z3solver.add(timestamps[ 2]==14422024000)
	z3solver.add(signal_5[2]==0.0)
	z3solver.add(signal_6[2]==1.0)
	z3solver.add(timestamps[ 3]==21633036000)
	z3solver.add(signal_5[3]==0.0)
	z3solver.add(signal_6[3]==1.0)
	z3solver.add(timestamps[ 4]==28844048000)
	z3solver.add(signal_5[4]==0.0)
	z3solver.add(signal_6[4]==1.0)
	z3solver.add(timestamps[ 5]==36055060000)
	z3solver.add(signal_5[5]==1.0)
	z3solver.add(signal_6[5]==0.0)
	z3solver.add(timestamps[ 6]==43266072000)
	z3solver.add(signal_5[6]==1.0)
	z3solver.add(signal_6[6]==0.0)
	z3solver.add(timestamps[ 7]==50477084000)
	z3solver.add(signal_5[7]==1.0)
	z3solver.add(signal_6[7]==0.0)
	z3solver.add(timestamps[ 8]==57688096000)
	z3solver.add(signal_5[8]==1.0)
	z3solver.add(signal_6[8]==0.0)
	z3solver.add(timestamps[ 9]==64899108000)
	z3solver.add(signal_5[9]==1.0)
	z3solver.add(signal_6[9]==0.0)
	z3solver.add(timestamps[ 10]==72110120000)
	z3solver.add(signal_5[10]==1.0)
	z3solver.add(signal_6[10]==0.0)
	z3solver.add(timestamps[ 11]==79321132000)
	z3solver.add(signal_5[11]==0.0)
	z3solver.add(signal_6[11]==1.0)
	z3solver.add(timestamps[ 12]==86532144000)
	z3solver.add(signal_5[12]==0.0)
	z3solver.add(signal_6[12]==1.0)
	z3solver.add(timestamps[ 13]==93743156000)
	z3solver.add(signal_5[13]==1.0)
	z3solver.add(signal_6[13]==0.0)
	z3solver.add(timestamps[ 14]==100954168000)
	z3solver.add(signal_5[14]==1.0)
	z3solver.add(signal_6[14]==0.0)
	z3solver.add(timestamps[ 15]==108165180000)
	z3solver.add(signal_5[15]==1.0)
	z3solver.add(signal_6[15]==0.0)
	z3solver.add(timestamps[ 16]==115376192000)
	z3solver.add(signal_5[16]==1.0)
	z3solver.add(signal_6[16]==0.0)
	z3solver.add(timestamps[ 17]==122587204000)
	z3solver.add(signal_5[17]==0.0)
	z3solver.add(signal_6[17]==1.0)
	z3solver.add(timestamps[ 18]==129798216000)
	z3solver.add(signal_5[18]==0.0)
	z3solver.add(signal_6[18]==1.0)




	# this is the first time stamp 0

	# this is the last time stamp 2147483647

	# this is the sample step 2147483647

	# the total number of samples is 18



	interval_s=And(s>0, s<18)
	conditions_s=And(Implies(signal_5[s]==0, signal_6[s]==0), Implies(signal_5[s]==1, signal_6[s]==0))

	# Not(ForAll([s], Implies(And(s>0, s<18), And(Implies(signal_5[s]==0, signal_6[s]==1), Implies(signal_5[s]==1, signal_6[s]==0)))))

	# ForAll([s], Implies(And(s>0, s<18), And(Implies(signal_5[s]==0, signal_6[s]==1), Implies(signal_5[s]==1, signal_6[s]==0))))

	# ForAll[s, Implies[And[>[s,0], <[s,18]], And[Implies[ ==[signal_5[s],0], ==[signal_6[s],1] ], Implies[ ==[signal_5[s],1], ==[signal_6[s],0] ]]]]


	# '["ForAll", [["s"], ["Implies", [["And", [[">", ["s",0]], ["<", ["s",18] ] ]], ["And", [["Implies", [ ["==", ["signal_5[s]",0]], ["==", ["signal_6[s]",1]] ]], ["Implies", [ ["==", ["signal_5[s]",1]], ["==", ["signal_6[s]",0]] ]]]]]]]]'

	# # ["ForAll", [["s"], ["Implies", [["And", [[">", ["s",0]], ["<", ["s",18] ] ]], ["And", [["Implies", [ ["==", ["signal_5[s]",0] ], ["==", ["signal_6[s]",1] ] ] ], ["Implies", [ ["==", ["signal_5[s]",1] ], ["==", ["signal_6[s]",0] ] ] ] ]]]]]]


	z3solver.add(Not(ForAll([s], Implies(interval_s, conditions_s))))
	status=z3solver.check()
	print(status)

	print("--- %s seconds ---" % (time.time() - start_time))
	if status == sat: 
		print("REQUIREMENT VIOLATED")
		return 0
	if status == unsat:
		print("REQUIREMENT SATISFIED")
		return 1
	else:
		print("UNDECIDED")
		return 2




if __name__ == "__main__":
	property_04()