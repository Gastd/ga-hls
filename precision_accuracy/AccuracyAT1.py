#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 00:11:33 2024

@author: fedformi_new
"""

# Load modules
import numpy as np
from timeit import default_timer as timer
import pandas as pd
from scipy.interpolate import interp1d

# Start timer
tic = timer()

#%% Define range, tool result and expert's prediction
exp = 2
if exp == 1:
    totalRange = np.array([0.0, 0.0, 20.0, 20.0, 100.0, 140.0])
    toolSatis = np.array([[0.0, 0.0, 20.0, 20.0, 120.006093, 140.0]])
    expertSatis = np.array([[0.0, 0.0, 20.0, 20.0, 120.022620, 140.0]])
    
elif exp == 2:
    totalRange = np.array([0.0, 10.0, 10.000000, 30.000000, 100.000000, 140.000000])
    toolSatis = np.array([[0.0, 10.0, 10.000000, 12.696411, 100.000000, 105.291267],
                          [0.0, 10.0, 12.696411, 13.309259, 102.890852, 105.291267],
                          [0.0, 10.0, 10.000000, 13.986527, 105.291267, 124.873017],
                          [0.0, 10.0, 13.986527, 15.896834, 108.980719, 115.435108],
                          [0.0, 10.0, 15.896834, 17.003177, 111.497926, 115.435108],
                          [0.0, 10.0, 13.986527, 18.917859, 115.435108, 124.873017],
                          [0.0, 10.0, 18.917859, 19.969545, 119.726918, 120.487499],
                          [0.0, 10.0, 18.917859, 21.228596, 120.487499, 124.873017],
                          [0.0, 10.0, 10.000000, 23.563323, 124.873017, 140.000000],
                          [0.0, 10.0, 23.563323, 25.066859, 127.344796, 131.000874],
                          [0.0, 10.0, 25.066859, 25.842157, 129.003908, 131.000874],
                          [0.0, 10.0, 23.563323, 28.286904, 131.000874, 140.000000],
                          [0.0, 10.0, 28.286904, 30.000000, 134.017274, 140.000000]])
    
    # toolSatis = np.array([[0.000000,10.000000,10.000000,12.541552,100.000000,112.606509],
    #                       [0.000000,10.000000,12.541552,13.821897,103.043637,112.606509],
    #                       [0.000000,10.000000,13.821897,15.685739,107.814131,112.606509],
    #                       [0.000000,10.000000,15.685739,16.286497,110.519468,112.606509],
    #                       [0.000000,10.000000,10.000000,17.468380,112.606509,125.339753],
    #                       [0.000000,10.000000,17.468380,18.622671,117.752412,118.515026],
    #                       [0.000000,10.000000,17.468380,19.511257,118.515026,120.787095],
    #                       [0.000000,10.000000,17.468380,21.693210,120.787095,125.339753],
    #                       [0.000000,10.000000,21.693210,22.588905,124.006303,125.339753],
    #                       [0.000000,10.000000,10.000000,23.021822,125.339753,126.472967],
    #                       [0.000000,10.000000,10.000000,25.451783,126.472967,140.000000],
    #                       [0.000000,10.000000,25.451783,26.983504,129.777801,133.115385],
    #                       [0.000000,10.000000,26.983504,28.200015,131.835491,133.115385],
    #                       [0.000000,10.000000,25.451783,30.000000,133.115385,140.000000]])
    
    # Load trace from csv file
    traceDataframe = pd.read_csv("tracesAT.csv")        # Read csv
    traceDataframe = traceDataframe[traceDataframe['Requirement'] == "AT1"]     # Filter by requirement
    traceStr = traceDataframe.at[traceDataframe.index[0], ' Output Trace']      # Get first trace
    traceStr = "[" + traceStr + "]"                     # Add square brackets at beginning and end
    traceStr = traceStr.replace("; ","],[")             # Replace semicolon with comma and square brackets
    traceStr = traceStr.replace(" ",",")                # Replace whitespace with comma
    traceData = eval(traceStr)                          # Convert string to list of lists
    del traceData[-1]                                   # Remove last element (i.e., an empty list)
    traceData = np.array(traceData)                     # Convert to np.array
    traceData = np.delete(traceData,[2, 3],1)           # Drop engine speed and gear columns
    traceFunc = interp1d(traceData[:,0],traceData[:,1]) # Convert trace into a function
    
elif exp == 3:
    totalRange = np.array([0.0, 10.0, 10.0, 30.0, 100.0, 140.0])
    toolSatis = np.array([[0.0, 10.0, 10.0, 16.794447, 107.258769, 125.366352],
                          [0.0, 10.0, 16.794447, 21.355433, 118.690749, 125.366352],
                          [0.0, 10.0, 10.0, 25.647437, 125.366352, 140.0],
                          [0.0, 10.0, 25.647437, 30.0, 133.701053, 140.0]])
    
    # Load trace from csv file
    traceDataframe = pd.read_csv("tracesAT.csv")        # Read csv
    traceDataframe = traceDataframe[traceDataframe['Requirement'] == "AT1"]     # Filter by requirement
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
    raise ValueError("Invalid experiment number. Choose either 1, 2 or 3.")
    
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
for i in range(0,N+1):
    param[0] = totalRange[0]+(totalRange[1]-totalRange[0])/N*i
    
    for j in range(0,N+1):
        param[1] = totalRange[2]+(totalRange[3]-totalRange[2])/N*j
        
        for k in range(0,N+1):
            param[2] = totalRange[4]+(totalRange[5]-totalRange[4])/N*k
            
            # Check satisfiability according to tool prediction
            toolBool = False
            for m in range(0,toolSatis.shape[0]):
                toolBool = toolBool or ((toolSatis[m][0] < param[0] <= toolSatis[m][1] or param[0] == totalRange[0] == toolSatis[m][0]) and
                                        (toolSatis[m][2] < param[1] <= toolSatis[m][3] or param[1] == totalRange[2] == toolSatis[m][2]) and
                                        (toolSatis[m][4] < param[2] <= toolSatis[m][5] or param[2] == totalRange[4] == toolSatis[m][4]))
                
            # Check satisfiability according to expert's prediction
            if exp == 1:
                expertBool = False
                for m in range(0,expertSatis.shape[0]):
                    expertBool = expertBool or ((expertSatis[m][0] < param[0] <= expertSatis[m][1] or param[0] == totalRange[0] == expertSatis[m][0]) and
                                            (expertSatis[m][2] < param[1] <= expertSatis[m][3] or param[1] == totalRange[2] == expertSatis[m][2]) and
                                            (expertSatis[m][4] < param[2] <= expertSatis[m][5] or param[2] == totalRange[4] == expertSatis[m][4]))
                    
            else:
                expertBool = param[2] > traceFunc(param[1])
                
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
    print("The results of the effectiveness analysis on AT1 - experiment 1 are as follows:")
elif exp == 2:
    print("The results of the effectiveness analysis on AT1 - experiment 2 are as follows:")
else:
    print("The results of the effectiveness analysis on AT1 - experiment 2 for 100 formulas are as follows:")
    
print("\t- True Positive:  %i" % truePos)
print("\t- True Negative:  %i" % trueNeg)
print("\t- False Positive: %i" % falsePos)
print("\t- False Negative: %i" % falseNeg)

print("\nPrecision metric: %.1f %%" % (precision*100))
print("Recall metric:    %.1f %%" % (recall*100))
print("Execution time for Effectiveness analysis: %f s" % (toc-tic))



