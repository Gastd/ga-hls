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
    totalRange = np.array([30.0, 30.0, 2800.0, 3200.0, 20.0, 20.0, 50.0, 80.0])
    toolSatis = np.array([[30.0, 30.0, 2800.0, 2972.147193, 20.0, 20.0, 50.0, 80.0],
                          [30.0, 30.0, 2972.147193, 3062.351416, 20.0, 20.0, 71.218868, 80.0],
                          [30.0, 30.0, 3077.353375, 3082.559240, 20.0, 20.0, 71.218868, 80.0],
                          [30.0, 30.0, 3087.446002, 3105.133856, 20.0, 20.0, 71.218868, 80.0],
                          [30.0, 30.0, 3105.133856, 3200.0, 20.0, 20.0, 71.218868, 80.0]])
    expertSatis = np.array([[30.0, 30.0, 2800.0, 2973.074700, 20.0, 20.0, 50.0, 80.0],
                            [30.0, 30.0, 2973.074700, 3200.0, 20.0, 20.0, 70.97682, 80.0]])
    
elif exp == 2:
    totalRange = np.array([20.0, 40.0, 2800.0, 3200.0, 15.0, 25.0, 50.0, 80.0])
    toolSatis = np.array([[20.0, 40.0, 2800.0, 2983.258156, 15.0, 25.0, 50.0, 80.0],
                          [20.0, 34.255560, 2983.258156, 3200.0, 15.0, 16.092594, 62.102839, 71.895253],
                          [20.0, 34.255560, 2983.258156, 3200.0, 16.891505, 17.618984, 62.102839, 71.895253],
                          [32.599609, 32.922184, 2983.258156, 3061.340642, 17.618984, 25.0, 62.102839, 71.895253],
                          [32.922184, 34.255560, 2983.258156, 3061.340642, 21.065876, 25.0, 62.102839, 71.895253],
                          [34.255560, 40.0, 2983.258156, 3085.865410, 15.0, 25.0, 50.0, 71.895253],
                          [36.091753, 40.0, 3085.865410, 3200.0, 15.0, 16.521013, 50.0, 71.895253],
                          [36.091753, 40.0, 3085.865410, 3188.947558, 19.051827, 25.0, 50.0, 71.895253],
                          [20.0, 40.0, 2983.258156, 3200.0, 15.0, 24.190488, 71.895253, 80.0],
                          [32.922184, 40.0, 2983.258156, 3200.0, 24.190488, 25.0, 71.895253, 80.0]])
    
    # Load trace from csv file
    traceDataframe = pd.read_csv("tracesAT.csv")        # Read csv
    traceDataframe = traceDataframe[traceDataframe['Requirement'] == "AT6B"]    # Filter by requirement
    traceStr = traceDataframe.at[traceDataframe.index[0], ' Output Trace']      # Get first trace
    traceStr = "[" + traceStr + "]"                     # Add square brackets at beginning and end
    traceStr = traceStr.replace("; ","],[")             # Replace semicolon with comma and square brackets
    traceStr = traceStr.replace(" ",",")                # Replace whitespace with comma
    traceData = eval(traceStr)                          # Convert string to list of lists
    del traceData[-1]                                   # Remove last element (i.e., an empty list)
    traceData = np.array(traceData)                     # Convert to np.array
    traceData = np.delete(traceData,3,1)                # Drop gear column
    traceFunc1 = interp1d(traceData[:,0],traceData[:,1]) # Convert trace into a function
    traceFunc2 = interp1d(traceData[:,0],traceData[:,2]) # Convert trace into a function
    
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
                    expertBool = (param[1] <= 2973.0) or (param[1] > traceFunc2(param[0])) or (param[3] > traceFunc1(param[2]))
                            
                # if toolBool:
                #     print('a')
                # Compare tool and expert's prediction on current sample
                truePos += toolBool and expertBool
                trueNeg += not toolBool and not expertBool
                falsePos += toolBool and not expertBool
                falseNeg += not toolBool and expertBool
            
            # Break third loop for first experiment (only param[1] and param[3] change)
            if exp == 1:
                break
        
    # Break first loop for first experiment (only param[2] and param[3] change)
    if exp == 1:
        break
    
#%% Compute Precision and Recall
precision = truePos/(truePos+falsePos)
recall = truePos/(truePos+falseNeg)

# Stop timer
toc = timer()

# Print results
if exp == 1:
    print("The results of the effectiveness analysis on AT6c - experiment 17 are as follows:")
else:
    print("The results of the effectiveness analysis on AT6c - experiment 18 are as follows:")
    
print("\t- True Positive:  %i" % truePos)
print("\t- True Negative:  %i" % trueNeg)
print("\t- False Positive: %i" % falsePos)
print("\t- False Negative: %i" % falseNeg)

print("\nPrecision netric: %.1f %%" % (precision*100))
print("Recall netric:    %.1f %%" % (recall*100))
print("Execution time for Effectiveness analysis: %f s" % (toc-tic))



