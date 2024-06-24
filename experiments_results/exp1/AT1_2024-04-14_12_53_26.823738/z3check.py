# Z3Py CODE: 
from z3 import *
import time
def AT1():
	start_time=time.time()
	z3solver=Solver() 

	#v_speed contained in the file
	t=Real('t') 

	#Trace: AT1
	timestamps=Array('timestamps', IntSort(), RealSort())
	v_speed=Array('v_speed', IntSort(), RealSort())

	z3solver.add(Not(ForAll([t], Implies(And((0 <= t), (t <= 20000000)), (v_speed[ToInt(RealVal(0)+(t-0.0)/10000.0)] < 111.88886722128798)))))

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
	AT1()