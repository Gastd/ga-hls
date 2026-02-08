# Z3Py CODE: 
from z3 import *
import time
def AT52():
	start_time=time.time()
	z3solver=Solver() 

	#gear contained in the file
	t1=Real('t1') 
	t2=Real('t2') 
	i=Int('i') 

	#Trace: AT52
	timestamps=Array('timestamps', IntSort(), RealSort())
	gear=Array('gear', IntSort(), RealSort())

	z3solver.add(Not(ForAll([i], Implies(And((i >= (ToInt(RealVal(0)+6924527/10000.0))), (i < (ToInt(RealVal(0)+23075278/10000.0)))), Implies(And((gear[(i-1)] != 2), (gear[i] == 2)), ForAll([t2], Implies(And((timestamps[i] <= t2), (t2 <= (timestamps[i] + 1672709))), (gear[ToInt(RealVal(0)+(t2-0.0)/10000.0)] == 2))))))))

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
	AT52()