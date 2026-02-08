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
    totalRange = np.array([1.0, 2.0, 7.0, 11.0, 1.0, 2.0, 7.0, 11.0])
    toolSatis = np.array([[1.0, 2.0, 10.716652, 11.0, 1.0, 2.0, 7.0, 7.759978],
                          [1.0, 2.0, 9.983620, 11.0, 1.0, 2.0, 7.759978, 10.681500]])
    expertSatis = np.array([[1.0, 2.0, 7.0, 10.001060, 1.0, 2.0, 7.0, 7.587900],
                            [1.0, 2.0, 10.001060, 11.0, 1.0, 2.0, 7.0, 11.0]])
    
elif exp == 2:
    totalRange = np.array([0.0, 2.0, 7.0, 11.0, 0.0, 2.0, 7.0, 11.0])
    toolSatis = np.array([[0.0, 1.0, 7.0, 11.0, 0.0, 2.0, 7.0, 11.0],
                          [1.0, 2.0, 9.973520, 11.0, 0.0, 2.0, 7.0, 11.0]])
    expertSatis = np.array([[0.0, 1.0, 7.0, 10.0, 0.0, 2.0, 7.0, 7.58],
                            [1.0, 2.0, 7.0, 10.0, 1.0, 2.0, 7.0, 7.58],
                            [0.0, 1.0, 7.0, 10.0, 0.0, 2.0, 7.58, 11.0],
                            [0.0, 2.0, 10.0, 11.0, 0.0, 2.0, 7.0, 11.0]])
    
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
param = np.zeros(4, dtype=float)

# Loop over values
for i in np.arange(totalRange[0],totalRange[1],step=1.0):
    param[0] = i+0.5
    
    for j in range(0,N+1):
        param[1] = totalRange[2]+(totalRange[3]-totalRange[2])/N*j
        
        for k in np.arange(totalRange[4],totalRange[5],step=1.0):
            param[2] = k+0.5
            
            for m in range(0,N+1):
                param[3] = totalRange[6]+(totalRange[7]-totalRange[6])/N*m
            
                # Check satisfiability according to tool prediction
                toolBool = False
                for n in range(0,toolSatis.shape[0]):
                    toolBool = toolBool or ((toolSatis[n][0] < param[0] <= toolSatis[n][1] or param[0] == totalRange[0] == toolSatis[n][0]) and
                                            (toolSatis[n][2] < param[1] <= toolSatis[n][3] or param[1] == totalRange[2] == toolSatis[n][2]) and
                                            (toolSatis[n][4] < param[2] <= toolSatis[n][5] or param[2] == totalRange[4] == toolSatis[n][4]) and
                                            (toolSatis[n][6] < param[3] <= toolSatis[n][7] or param[3] == totalRange[6] == toolSatis[n][6]))
                    
                # Check satisfiability according to expert's prediction
                expertBool = False
                for n in range(0,expertSatis.shape[0]):
                    expertBool = expertBool or ((expertSatis[n][0] < param[0] <= expertSatis[n][1] or param[0] == totalRange[0] == expertSatis[n][0]) and
                                            (expertSatis[n][2] < param[1] <= expertSatis[n][3] or param[1] == totalRange[2] == expertSatis[n][2]) and
                                            (expertSatis[n][4] < param[2] <= expertSatis[n][5] or param[2] == totalRange[4] == expertSatis[n][4]) and
                                            (expertSatis[n][6] < param[3] <= expertSatis[n][7] or param[3] == totalRange[6] == expertSatis[n][6]))
                    
                # Compare tool and expert's prediction on current sample
                truePos += toolBool and expertBool
                trueNeg += not toolBool and not expertBool
                falsePos += toolBool and not expertBool
                falseNeg += not toolBool and expertBool
            
            # Break third loop for first experiment (only param[1] and param[3] change)
            if exp == 1:
                break
        
    # Break first loop for first experiment (only param[1] and param[3] change)
    if exp == 1:
        break
    
#%% Compute Precision and Recall
precision = truePos/(truePos+falsePos)
recall = truePos/(truePos+falseNeg)

# Stop timer
toc = timer()

# Print results
if exp == 1:
    print("The results of the effectiveness analysis on CC5 - experiment 29 are as follows:")
else:
    print("The results of the effectiveness analysis on CC5 - experiment 30 are as follows:")
    
print("\t- True Positive:  %i" % truePos)
print("\t- True Negative:  %i" % trueNeg)
print("\t- False Positive: %i" % falsePos)
print("\t- False Negative: %i" % falseNeg)

print("\nPrecision metric: %.1f %%" % (precision*100))
print("Recall metric:    %.1f %%" % (recall*100))
print("Execution time for Effectiveness analysis: %f s" % (toc-tic))



