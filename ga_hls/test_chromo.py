# Z3Py CODE: 
from z3 import *
import time
def property_01():
	start_time=time.time()
	z3solver=Solver() 

	#signal_4 contained in the file

	#signal_2 contained in the file
	s=Int('s') 

	#Trace: property_01
	timestamps=Array('timestamps', RealSort(), IntSort())
	signal_4=Array('signal_4', RealSort(), IntSort())
	signal_2=Array('signal_2', RealSort(), IntSort())


	z3solver.add(Not(Exists([s], Implies(Or((s > 0), (s < 10)), And((signal_4[s] < 11), (signal_2[s] >= 70.44525463010169))))))

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
	property_01()