#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 00:11:33 2024

@author: fedformi_new
"""

# Load modules
import numpy as np
from timeit import default_timer as timer

# Start timer
tic = timer()

#%% Define range, tool result and expert's prediction
exp = 2
if exp == 1:
    # Parameters are in the order: Quant1, Log2, Quant0
    totalRange = np.array([0.0, 1.0, 1.0, 2.0, 0.0, 2.0])
    toolSatis = np.array([[0.0, 1.0, 1.0, 2.0, 0.0, 0.0]])
    expertSatis = np.array([[0.0, 1.0, 1.0, 2.0, 1.0, 2.0]])
    
elif exp == 2:
    # Parameters are in the order: Quant0, Quant1, Log2
    totalRange = np.array([0.0, 2.0, 0.0, 2.0, 0.0, 2.0])
    toolSatis = np.array([[0.0, 2.0, 1.0, 2.0, 1.0, 2.0]])
    expertSatis = np.array([[0.0, 1.0, 1.0, 2.0, 1.0, 2.0],
                            [1.0, 2.0, 0.0, 2.0, 1.0, 2.0]])
    
else:
    raise ValueError("Invalid experiment number. Choose either 1 or 2.")
    
#%% Generate samples and evaluate them according to the tool and the expert's prediction

N = 100 # Number of equally spaced intervals to split each range.

# Initialize precision variables
truePos = 0     # Number of True Positive instances
trueNeg = 0     # Number of True Negative instances
falsePos = 0    # Number of False Positive instances
falseNeg = 0    # Number of False Negative instances

# Initialize empty parameter array
param = np.zeros(3, dtype=float)

# Loop over values
for i in np.arange(totalRange[0],totalRange[1],step=1.0):
    param[0] = i+0.5
    
    for j in np.arange(totalRange[2],totalRange[3],step=1.0):
        param[1] = j+0.5
        
        for k in np.arange(totalRange[4],totalRange[5],step=1.0):
            param[2] = k+0.5
            
            # Check satisfiability according to tool prediction
            toolBool = False
            for m in range(0,toolSatis.shape[0]):
                toolBool = toolBool or ((toolSatis[m][0] < param[0] <= toolSatis[m][1] or param[0] == totalRange[0] == toolSatis[m][0]) and
                                        (toolSatis[m][2] < param[1] <= toolSatis[m][3] or param[1] == totalRange[2] == toolSatis[m][2]) and
                                        (toolSatis[m][4] < param[2] <= toolSatis[m][5] or param[2] == totalRange[4] == toolSatis[m][4]))
                
            # Check satisfiability according to expert's prediction
            expertBool = False
            for m in range(0,expertSatis.shape[0]):
                expertBool = expertBool or ((expertSatis[m][0] < param[0] <= expertSatis[m][1] or param[0] == totalRange[0] == expertSatis[m][0]) and
                                        (expertSatis[m][2] < param[1] <= expertSatis[m][3] or param[1] == totalRange[2] == expertSatis[m][2]) and
                                        (expertSatis[m][4] < param[2] <= expertSatis[m][5] or param[2] == totalRange[4] == expertSatis[m][4]))
                
            # Compare tool and expert's prediction on current sample
            truePos += toolBool and expertBool
            trueNeg += not toolBool and not expertBool
            falsePos += toolBool and not expertBool
            falseNeg += not toolBool and expertBool
            
        # Break second loop for first experiment (only param[2] changes)
        if exp == 1:
            break
        
    # Break first loop for first experiment (only param[2] changes)
    if exp == 1:
        break
    
#%% Compute Precision and Recall
precision = truePos/(truePos+falsePos)
recall = truePos/(truePos+falseNeg)

# Stop timer
toc = timer()

# Print results
if exp == 1:
    print("The results of the effectiveness analysis on CC3 - experiment 25 are as follows:")
else:
    print("The results of the effectiveness analysis on CC3 - experiment 26 are as follows:")
    
print("\t- True Positive:  %i" % truePos)
print("\t- True Negative:  %i" % trueNeg)
print("\t- False Positive: %i" % falsePos)
print("\t- False Negative: %i" % falseNeg)

print("\nPrecision metric: %.1f %%" % (precision*100))
print("Recall metric:    %.1f %%" % (recall*100))
print("Execution time for Effectiveness analysis: %f s" % (toc-tic))



