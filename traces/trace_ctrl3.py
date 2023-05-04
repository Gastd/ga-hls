# Z3Py CODE: 
from z3 import *
import time
def property_03():
	start_time=time.time()
	z3solver=Solver() 

	#signal_1 contained in the file
	t=Real('t') 

	#Trace: property_03
	timestamps=Array('timestamps', RealSort(), IntSort())
	signal_1=Array('signal_1', RealSort(), IntSort())
	z3solver.add(timestamps[ 0]==0)
	z3solver.add(signal_1[0]==0.0)
	z3solver.add(timestamps[ 1]==3600000000)
	z3solver.add(signal_1[1]==0.0)
	z3solver.add(timestamps[ 2]==7200000000)
	z3solver.add(signal_1[2]==0.0)
	z3solver.add(timestamps[ 3]==10800000000)
	z3solver.add(signal_1[3]==0.0)
	z3solver.add(timestamps[ 4]==14400000000)
	z3solver.add(signal_1[4]==1.0)
	z3solver.add(timestamps[ 5]==18000000000)
	z3solver.add(signal_1[5]==1.0)
	z3solver.add(timestamps[ 6]==21600000000)
	z3solver.add(signal_1[6]==1.0)
	z3solver.add(timestamps[ 7]==25200000000)
	z3solver.add(signal_1[7]==2.0)
	z3solver.add(timestamps[ 8]==28800000000)
	z3solver.add(signal_1[8]==3.0)
	z3solver.add(timestamps[ 9]==32400000000)
	z3solver.add(signal_1[9]==3.0)
	z3solver.add(timestamps[ 10]==36000000000)
	z3solver.add(signal_1[10]==3.0)




	# this is the first time stamp 0

	# this is the last time stamp 2147483647

	# this is the sample step 2147483647

	# the total number of samples is 10



	interval_t=And(0<=t, t<=3.6E10)
	conditions_t=Or(signal_1[ToInt(RealVal(0)+(t-0.0)/3.6E9)]==1, signal_1[ToInt(RealVal(0)+(t-0.0)/3.6E9)]==2)
	z3solver.add(Not(ForAll([t], Implies(interval_t, conditions_t))))
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
	property_03()