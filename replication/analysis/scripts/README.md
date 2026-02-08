# Results analysis
This folder contains the Python scripts used to compute the main results presented in the Evaluation and Discussion sections.  
This folder contains:

* One accuracy script for each trace-requirement combination. More details below.
* Three csv files containing the traces, divided by model: `tracesAT.csv`, `tracesCC.csv`, and `traceRR.csv`.
* The Python script `countSamples.py`: this script loads all the traces and performs a simple statistical analysis on the number of samples.
* The Python script `RQ2.py`: this script performs a simple statistical analysis on the execution time of the 34 experiments. The results are reported in Section 5.3 of the paper.
* The Python script `printBoxplot.py`: this script performs a simple statistical analysis on the precision and recall of the 34 experiments and generates the Boxplot used in Figure 11 of the paper.

## Accuracy scripts
This folder contains a script for each of the 17 trace-requirement combinations used in the Evaluation section of the paper.
Each script computes the precision and recall as described in Section 5.2 of the paper.

The scripts can be run in two different ways: *exp_1* and *exp_2*.
These refer to the two different sets of mutation operators used for each trace-requirement combination.  
For example, in `AccuracyAT52.py`, if `exp = 1`, then the script will return the results for Experiment 7; otherwise, if `exp = 2`, then it will return the results for Experiment 8.  
The scripts `AccuracyAT1.py` and `AccuracyAT2.py` also have an option for *exp_3*: they are the experiments performed with MG = 100 and mentioned in Section 6.

Each experiment has to define:

* The total range for the numerical or discrete parameters.
* The hypercubes that define the satisfiability regions according to the tool diagnosis.
* The hypercubes that define the satisfiability regions according to the authors' prediction.

Each hypercube is defined by the minimum and maximum values for each mutation operator.  
For example, the first diagnosis hypercube in Experiment 4 is `[0.0, 5.0, 5.0, 7.171464, 4700.0, 4754.193158]`.
This means that if the mutation operators satisfy the condition below, then the trace will satisfy the requirement (according to the tool):

```
0.0 <= Num0 <= 5.0 AND
5.0 <= Num1 <= 7.171464 AND
4700.0 <= Num2 <= 4754.193158
```

**Note:** Running each script will take approximately 10s.
Experiments 14, 16, 18, and 20 will take significantly more time, since they will check 101^4 ~ 100 million combinations.
On average, they take 30 minutes to run.