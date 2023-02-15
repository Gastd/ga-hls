## Experiments with Decision Trees

15/02/2023

**Objective:** Identify are the elements of that turn the [original formula](original_formula) false.

```
Origina Formula: 
ForAll s In (s > 0 And s < 10) Implies (signal_4(s) < 50 And signal_2(s) >= -15.27)
```

**Method:** Apply a Decision Tree Learner (J48) on a [set of formulae](dataset) classified between FALSE or TRUE. The set of formulae is divided in two, ```all``` and ```pareto```. The latter means that we use pareto and take 20% of the formulae that are closest to the original formula, assuming that they should contain the most relevant information. The former contains all the formulae resulting from the mutations.

**Results:** We ran J48 in Weka (3.9.6) resulting in [eight decision trees](results) to be compared. Each decision tree used a set of formulae, distinguished by generations. Apparently, more data is better for the J48 algorithm. The trees generated with the pareto do not predict well (very low accuracy) and are not informative.

**Conclusion:** We should probably try to generate more formulae in our mutations while still using the pareto approach. 

