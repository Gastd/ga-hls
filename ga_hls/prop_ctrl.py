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
	z3solver.add(timestamps[ 0]==0)
	z3solver.add(signal_4[0]==100.0)
	z3solver.add(signal_2[0]==15.0)
	z3solver.add(timestamps[ 1]==3600000000)
	z3solver.add(signal_4[1]==100.0)
	z3solver.add(signal_2[1]==19.998289474268805)
	z3solver.add(timestamps[ 2]==7200000000)
	z3solver.add(signal_4[2]==80.0)
	z3solver.add(signal_2[2]==39.99315555555556)
	z3solver.add(timestamps[ 3]==10800000000)
	z3solver.add(signal_4[3]==60.0)
	z3solver.add(signal_2[3]==20.006844444444443)
	z3solver.add(timestamps[ 4]==14400000000)
	z3solver.add(signal_4[4]==40.0)
	z3solver.add(signal_2[4]==34.99486250142708)
	z3solver.add(timestamps[ 5]==18000000000)
	z3solver.add(signal_4[5]==30.0)
	z3solver.add(signal_2[5]==25.00537117633483)
	z3solver.add(timestamps[ 6]==21600000000)
	z3solver.add(signal_4[6]==20.0)
	z3solver.add(signal_2[6]==29.99731388888889)
	z3solver.add(timestamps[ 7]==25200000000)
	z3solver.add(signal_4[7]==5.0)
	z3solver.add(signal_2[7]==39.99462222520987)
	z3solver.add(timestamps[ 8]==28800000000)
	z3solver.add(signal_4[8]==5.0)
	z3solver.add(signal_2[8]==2.913524752694663)
	z3solver.add(timestamps[ 9]==32400000000)
	z3solver.add(signal_4[9]==5.0)
	z3solver.add(signal_2[9]==0.0)
	z3solver.add(timestamps[ 10]==36000000000)
	z3solver.add(signal_4[10]==5.0)
	z3solver.add(signal_2[10]==0.0)




	# this is the first time stamp 0

	# this is the last time stamp 2147483647

	# this is the sample step 2147483647

	# the total number of samples is 10



	interval_s=And(s>0, s<10)
	conditions_s=And(signal_4[s]<1000, signal_2[s]>=-(15.27))
	# x = ["Not",["ForAll",[["s"],["Implies",["And",[[">",["s",0]],["<",["s",10]]],["And",[["<",["signal_4(s)",1000]],[[">=",["signal_2(s)",-15.27]]]]]]]]]]
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
	property_01()