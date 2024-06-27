#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:40:48 2024

@author: fedformi_new
"""
import numpy as np
import pandas as pd

# List of requirements
reqList = ["AT1", "AT2", "AT51", "AT52", "AT53", "AT54", "AT6A", "AT6B", "AT6C", "AT6ABC",
           "CC1", "CC2", "CC3", "CC4", "CC5", "CCX", "RR"]
sample = np.zeros(len(reqList))

#%% Loop over all the requirements

for ii in range(len(reqList)):
    
    # Load sample depending on requirement
    if reqList[ii].startswith("AT"):
        traceDataframe = pd.read_csv("tracesAT.csv")        # Read csv
        traceDataframe = traceDataframe[traceDataframe['Requirement'] == reqList[ii]]   # Filter by requirement
        traceStr = traceDataframe.at[traceDataframe.index[0], ' Output Trace']          # Get first trace
        traceStr = "[" + traceStr + "]"                     # Add square brackets at beginning and end
        traceStr = traceStr.replace("; ","],[")             # Replace semicolon with comma and square brackets
        traceStr = traceStr.replace(" ",",")                # Replace whitespace with comma
        traceData = eval(traceStr)                          # Convert string to list of lists
        del traceData[-1]                                   # Remove last element (i.e., an empty list)
        traceData = np.array(traceData)                     # Convert to np.array
    elif reqList[ii].startswith("CC"):
        traceDataframe = pd.read_csv("tracesCC.csv")        # Read csv
        traceDataframe = traceDataframe[traceDataframe['Requirement'] == reqList[ii]]   # Filter by requirement
        traceStr = traceDataframe.at[traceDataframe.index[0], ' Output Trace']           # Get first trace
        traceStr = "[" + traceStr + "]"                     # Add square brackets at beginning and end
        traceStr = traceStr.replace("; ","],[")             # Replace semicolon with comma and square brackets
        while '  ' in traceStr:                             # Remove multiple whitespaces
            traceStr = traceStr.replace("  "," ")
        traceStr = traceStr.replace(" ",",")                # Replace whitespace with comma
        traceData = eval(traceStr)                          # Convert string to list of lists
        del traceData[-1]                                   # Remove last element (i.e., an empty list)
        traceData = np.array(traceData)                     # Convert to np.array
    elif reqList[ii].startswith("RR"):
        traceDataframe = pd.read_csv("traceRR.csv", header = None)     # Read csv
        traceData = traceDataframe[[0, 2, 4, 6]]
        traceData = traceData.to_numpy()
        traceData[:,0] = traceData[:,0]/1e6
    else:
        raise ValueError("Invalid requirement name.")
        
    # Save number of samples
    sample[ii] = traceData.shape[0]
    
#%% Get average, min, max and std dev of the samples number
print("The average number of samples in the trace is: %.1f" % np.mean(sample))
print("The minimum number of samples in the trace is: %i" % np.min(sample))
print("The maximum number of samples in the trace is: %i" % np.max(sample))
print("The standard deviation of the number of samples in the trace is: %.1f" % np.std(sample))