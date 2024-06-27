#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 20:24:35 2024

@author: fedformi_new
"""

# Load modules
import numpy as np
import matplotlib.pyplot as plt
import tabulate as tab

#%% Define True positive, False Positive, and False Negative array (only AT2 to AT54)
truePos = np.array([[50,            # AT1
                     549844,
                     46,            # AT2
                     591456,
                     33,            # AT51
                     693769,
                     24,            # AT52
                     936977,
                     41,            # AT53
                     490961,
                     36,            # AT54
                     452581,
                     5809,          # AT6a
                     63007032,
                     4562,          # AT6b
                     62096524,
                     6004,          # AT6c
                     70376869,
                     6004,          # AT6abc
                     63865078,
                     45,            # CC1
                     765479,
                     84,            # CC2
                     415716,
                     np.NaN,        # CC3
                     2,
                     45,            # CC4
                     250,
                     2002,          # CC5
                     25467,
                     48,            # CCx
                     577705,
                     77,            # RR
                     16526]])

falsePos = np.array([[0,            # AT1
                      7575,
                      0,            # AT2
                      6464,
                      0,            # AT51
                      0,
                      0,            # AT52
                      0,
                      0,            # AT53
                      0,
                      0,            # AT54
                      0,
                      0,            # AT6a
                      2254320,
                      93,           # AT6b
                      6762042,
                      0,            # AT6c
                      3021499,
                      0,            # AT6abc
                      1847172,
                      0,            # CC1
                      4646,
                      0,            # CC2
                      24745,
                      0,            # CC3
                      0,
                      0,            # CC4
                      1,
                      74,           # CC5
                      187,
                      0,            # CCx
                      440,
                      0,            # RR
                      100]])

falseNeg = np.array([[0,            # AT1
                      13332,
                      0,            # AT2
                      0,
                      0,            # AT51
                      0,
                      0,            # AT52
                      0,
                      0,            # AT53
                      0,
                      1,            # AT54
                      8888,
                      72,           # AT6a
                      3362593,
                      59,           # AT6b
                      4809225,
                      207,          # AT6c
                      30015278,
                      207,          # AT6abc
                      30189808,
                      0,            # CC1
                      0,
                      0,            # CC2
                      25452,
                      0,            # CC3
                      1,
                      3,            # CC4
                      101,
                      1663,         # CC5
                      1125,
                      0,            # CCx
                      13704,
                      24,           # RR
                      0]])

# Compute Precision and Recall as percentage
precision = truePos/(truePos+falsePos)*100
recall = truePos/(truePos+falseNeg)*100
effectiveness = np.transpose(np.concatenate((precision,recall),axis=0))

# Print out Precision and Recall
expStr = ["exp" + str(i+1) for i in range(0,truePos.shape[1])]
tableRQ1 = [[expStr[i], "%.1f %%" % precision[0][i], "%.1f %%" % recall[0][i]] for i in range(0,truePos.shape[1])]
print(tab.tabulate(tableRQ1, headers=["Experiment","Precision","Recall"]))
print("\n")

#%% Print statistical analysis of precision and recall

# Remove NaN
precision = np.expand_dims(precision[~np.isnan(precision)],axis=1)
recall = np.expand_dims(recall[~np.isnan(recall)],axis=1)
effectiveness = np.concatenate((precision,recall),axis=1)

# Statistical analysis of the precision
print("The average precision is: %.1f %%" % np.mean(precision))
print("The minimum precision is: %.1f %%" % np.min(precision))
print("The maximum precision is: %.1f %%" % np.max(precision))
print("The Standanrd Deviation of precision is: %.1f %%\n" % np.std(precision))

# Statistical analysis of the precision
print("The average recall is: %.1f %%" % np.mean(recall))
print("The minimum recall is: %.1f %%" % np.min(recall))
print("The maximum recall is: %.1f %%" % np.max(recall))
print("The Standanrd Deviation of recall is: %.1f %%" % np.std(recall))

#%% Draw boxplot of Precision and Recall
fig, ax = plt.subplots()

BP = ax.boxplot(effectiveness, positions=[2,3.5], widths=1.0, patch_artist=True,
                showmeans=True, showfliers=True,
                meanprops={"marker": "D","markerfacecolor": "white", "markeredgecolor": "green", "markeredgewidth": 1.0, "markersize":15},
                medianprops={"color": "red", "linewidth": 1.0},
                boxprops={"facecolor": "white", "edgecolor": "blue", "linewidth": 1.5},
                whiskerprops={"color": "black", "linewidth": 1.5, "linestyle": "dashed"},
                capprops={"color": "black", "linewidth": 1.5},
                flierprops={"marker": "+","markerfacecolor": "white", "markeredgecolor": "red", "markeredgewidth": 1.0, "markersize":10})

ax.set(xlim=(1, 4.5), ylim=(50, 105), yticks=np.arange(50, 110, 10))
ax.grid(visible=True)
ax.set_xticks([2,3.5],labels=["Precision","Recall"],fontsize=16)
ax.set_ylabel("Percentage [%]",fontsize=16)

# Save figure
fig.savefig("Boxplot.pdf")

#%% Draw violinplot of Precision and Recall
fig2, ax2 = plt.subplots()

VP = ax2.violinplot(effectiveness, positions=[2,3.5], widths=1.0, vert=True,
                    showmeans=True, showextrema=True, showmedians=True, points=100,)

vp = VP['cmeans']
vp.set_edgecolor("green")

vp = VP['cmedians']
vp.set_edgecolor("red")

for line in ['cmins', 'cmaxes']:
    vp = VP[line]
    vp.set_edgecolor("black")

ax2.set(xlim=(1, 4.5), ylim=(50, 105), yticks=np.arange(50, 110, 10))
ax2.grid(visible=True)
ax2.set_xticks([2,3.5],labels=["Precision","Recall"],fontsize=16)
ax2.set_ylabel("Percentage [%]",fontsize=16)

# Save figure
fig2.savefig("Violinplot.pdf")


