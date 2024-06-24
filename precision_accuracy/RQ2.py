#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 01:33:03 2024

@author: fedformi_new
"""

import numpy as np

#%% Define time required for each diagnosis
time = np.array([[11.6,         # AT1
                  8.8,
                  7.9,          # AT2
                  6.1,
                  12.5,         # AT51
                  16.1,
                  12.5,         # AT52
                  46.7,
                  22.1,         # AT53
                  13.0,
                  16.5,         # AT54
                  11.5,
                  120.0,        # AT6a
                  120.0,
                  120.0,        # AT6b
                  120.0,
                  120.0,        # AT6c
                  120.0,
                  120.0,        # AT6abc
                  120.0,
                  120.0,        # CC1
                  120.0,
                  120.0,        # CC2
                  120.0,
                  120.0,        # CC3
                  120.0,
                  120.0,        # CC4
                  120.0,
                  120.0,        # CC5
                  120.0,
                  120.0,        # CCx
                  120.0,
                  10.8,         # RR
                  10.8]])

#%% Print statistical analysis of computational time

# Print total computation time
print("To replicate all the experiments and run them in series, it would take approximately %.0f days.\n" % (np.sum(time)/24))

# Remove experiments that took 120 hours
time = time[time < 120]
print("The average computational time is: %.1f" % np.mean(time))
print("The minimum computational time is: %.1f" % np.min(time))
print("The maximum computational time is: %.1f" % np.max(time))
print("The Standanrd Deviation of computational time is: %.1f\n" % np.std(time))