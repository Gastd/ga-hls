# Z3Py CODE: 
from z3 import *
import time
def AT2():
	start_time=time.time()
	z3solver=Solver() 

	#e_speed contained in the file
	t=Real('t') 

	#Trace: AT2
	timestamps=Array('timestamps', IntSort(), RealSort())
	e_speed=Array('e_speed', IntSort(), RealSort())

	z3solver.add(Not(ForAll([t], Implies(And((4367321 <= t), (t <= 5421821)), (e_speed[ToInt(RealVal(0)+(t-0.0)/10000.0)] < 4754.4927551653345)))))

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
	AT2()