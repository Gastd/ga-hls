#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 00:11:33 2024

@author: fedforni_new
"""

# Load nodules
import numpy as np
from timeit import default_timer as timer
import pandas as pd
from scipy.interpolate import interp1d

# Start tiner
tic = timer()

#%% Define range, tool result and expert's prediction
exp = 2
if exp == 1:
    totalRange = np.array([30.0, 30.0, 2800.0, 3200.0, 4.0, 4.0, 30.0, 40.0])
    toolSatis = np.array([[30.0, 30.0, 2800.0, 2912.652906, 4.0, 4.0, 30.0, 36.003122],
                          [30.0, 30.0, 2800.0, 3200.0, 4.0, 4.0, 36.003122, 40.0]])
    expertSatis = np.array([[30.0, 30.0, 2800.0, 2913.192700, 4.0, 4.0, 30.0, 40.0],
                            [30.0, 30.0, 2913.192700, 3200.0, 4.0, 4.0, 35.960581, 40.0]])
    
elif exp == 2:
    totalRange = np.array([20.0, 40.0, 2800.0, 3200.0, 2.0, 6.0, 30.0, 40.0])
    toolSatis = np.array([[20.0, 40.0, 2800.000000, 2913.702908, 2.000000, 3.816740, 30.000000, 40.000000],
                          [20.0, 40.0, 2800.000000, 2913.702908, 4.018456, 6.000000, 30.000000, 40.000000],
                          [20.0, 40.0, 2913.702908, 3200.000000, 2.000000, 2.472328, 30.000000, 32.159694],
                          [20.0, 40.0, 2913.702908, 3200.000000, 2.000000, 3.266351, 32.159694, 37.783678],
                          [20.0, 40.0, 2913.702908, 3200.000000, 3.266351, 4.165356, 34.716290, 37.783678],
                          [20.0, 40.0, 2913.702908, 3200.000000, 2.000000, 6.000000, 37.783678, 40.000000]])
    
    # Load trace from csv file
    traceDataframe = pd.read_csv("tracesAT.csv")        # Read csv
    traceDataframe = traceDataframe[traceDataframe['Requirement'] == "AT6A"]    # Filter by requirement
    traceStr = traceDataframe.at[traceDataframe.index[0], ' Output Trace']      # Get first trace
    traceStr = "[" + traceStr + "]"                     # Add square brackets at beginning and end
    traceStr = traceStr.replace("; ","],[")             # Replace semicolon with comma and square brackets
    traceStr = traceStr.replace(" ",",")                # Replace whitespace with comma
    traceData = eval(traceStr)                          # Convert string to list of lists
    del traceData[-1]                                   # Remove last element (i.e., an empty list)
    traceData = np.array(traceData)                     # Convert to np.array
    traceData = np.delete(traceData,[2, 3],1)           # Drop engine speed and gear columns
    traceFunc = interp1d(traceData[:,0],traceData[:,1]) # Convert trace into a function
    
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
for i in range(0,N+1):
    param[0] = totalRange[0]+(totalRange[1]-totalRange[0])/N*i
    
    for j in range(0,N+1):
        param[1] = totalRange[2]+(totalRange[3]-totalRange[2])/N*j
        
        for k in range(0,N+1):
            param[2] = totalRange[4]+(totalRange[5]-totalRange[4])/N*k
            
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
                if exp == 1:
                    expertBool = False
                    for n in range(0,expertSatis.shape[0]):
                        expertBool = expertBool or ((expertSatis[n][0] < param[0] <= expertSatis[n][1] or param[0] == totalRange[0] == expertSatis[n][0]) and
                                                (expertSatis[n][2] < param[1] <= expertSatis[n][3] or param[1] == totalRange[2] == expertSatis[n][2]) and
                                                (expertSatis[n][4] < param[2] <= expertSatis[n][5] or param[2] == totalRange[4] == expertSatis[n][4]) and
                                                (expertSatis[n][6] < param[3] <= expertSatis[n][7] or param[3] == totalRange[6] == expertSatis[n][6]))
                        
                else:
                    expertBool = (param[1] <= 2913.0) or (param[3] > traceFunc(param[2])) 
                            
                    
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
    print("The results of the effectiveness analysis on AT6a - experiment 13 are as follows:")
else:
    print("The results of the effectiveness analysis on AT6a - experiment 14 are as follows:")
    
print("\t- True Positive:  %i" % truePos)
print("\t- True Negative:  %i" % trueNeg)
print("\t- False Positive: %i" % falsePos)
print("\t- False Negative: %i" % falseNeg)

print("\nPrecision netric: %.1f %%" % (precision*100))
print("Recall netric:    %.1f %%" % (recall*100))
print("Execution time for Effectiveness analysis: %f s" % (toc-tic))



