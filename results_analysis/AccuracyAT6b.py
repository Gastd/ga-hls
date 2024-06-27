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
    totalRange = np.array([30.0, 30.0, 2800.0, 3200.0, 8.0, 8.0, 40.0, 60.0])
    toolSatis = np.array([[30.0, 30.0, 2800.0, 2827.941205, 8.0, 8.0, 40.0, 51.696311],
                          [30.0, 30.0, 2800.0, 3200.0, 8.0, 8.0, 51.696311, 60.0]])
    expertSatis = np.array([[30.0, 30.0, 2800.0, 2830.707000, 8.0, 8.0, 40.0, 60.0],
                            [30.0, 30.0, 2830.707000, 3200.0, 8.0, 8.0, 51.918594, 60.0]])
    
elif exp == 2:
    totalRange = np.array([20.0, 40.0, 2800.0, 3200.0, 4.0, 12.0, 40.0, 60.0])
    toolSatis = np.array([[20.000000, 30.668511, 2800.000000, 3200.000000, 04.000000, 04.825345, 40.000000, 44.022237],
                          [20.000000, 30.668511, 2800.000000, 3200.000000, 04.000000, 05.448012, 44.022237, 48.840268],
                          [20.000000, 30.668511, 2800.000000, 3200.000000, 05.448012, 06.211314, 46.679013, 48.840268],
                          [20.000000, 30.668511, 2800.000000, 2814.785606, 06.211314, 12.000000, 40.000000, 48.840268],
                          [20.000000, 30.668511, 2800.000000, 3200.000000, 04.000000, 08.189435, 48.840268, 57.085968],
                          [20.000000, 25.038919, 2800.000000, 3200.000000, 08.189435, 08.946279, 52.845510, 57.085968],
                          [25.038919, 30.668511, 2800.000000, 3200.000000, 08.189435, 10.227629, 52.845510, 57.085968],
                          [20.000000, 30.668511, 2800.000000, 3200.000000, 04.000000, 10.227629, 57.085968, 60.000000],
                          [20.000000, 30.668511, 2800.000000, 3097.953708, 11.268312, 12.000000, 57.199797, 60.000000],
                          [20.000000, 30.668511, 3097.953708, 3200.000000, 10.227629, 12.000000, 57.199797, 60.000000],
                          [31.673668, 40.000000, 2800.000000, 3045.987565, 04.000000, 12.000000, 40.000000, 41.949363],
                          [30.668511, 40.000000, 2800.000000, 3147.640064, 04.000000, 12.000000, 42.188072, 60.000000],
                          [30.668511, 40.000000, 3147.640064, 3200.000000, 04.000000, 06.681406, 42.188072, 60.000000],
                          [30.668511, 40.000000, 3147.640064, 3200.000000, 06.681406, 07.820884, 48.920876, 60.000000],
                          [30.668511, 31.335813, 3147.640064, 3200.000000, 07.820884, 12.000000, 48.920876, 60.000000],
                          [31.335813, 40.000000, 3147.640064, 3200.000000, 11.761297, 12.000000, 48.920876, 60.000000]])
    
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
                    expertBool = (param[1] <= traceFunc2(param[0])) or (param[3] > traceFunc1(param[2])) 
                            
                    
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
    print("The results of the effectiveness analysis on AT6b - experiment 15 are as follows:")
else:
    print("The results of the effectiveness analysis on AT6b - experiment 16 are as follows:")
    
print("\t- True Positive:  %i" % truePos)
print("\t- True Negative:  %i" % trueNeg)
print("\t- False Positive: %i" % falsePos)
print("\t- False Negative: %i" % falseNeg)

print("\nPrecision netric: %.1f %%" % (precision*100))
print("Recall netric:    %.1f %%" % (recall*100))
print("Execution time for Effectiveness analysis: %f s" % (toc-tic))



