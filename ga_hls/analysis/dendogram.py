#!/usr/bin/python3
# -*- coding: latin-1 -*-
# analyze.py
# Author: Caio Batista de Melo
# Date created: 2018-06-04
# Last modified: 2023-03-05
# Description: This scripts takes a list of ISs traces as input and exports results
#              that helps the user to analyze what similarities those ISs have.
#
# New addition (2018-08-07): added the Levenshtein distance to compare CBs.
# New addition (2023-03-05): extract function to place in module within ga_hls.
#
# Note: to time this script, simply remove all '##', the lines to calculate and output
#       times are commented that way and they are correctly idented already, so simply
#       removing those double hashes should do the trick.

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import dendrogram, linkage
from timeit import default_timer as timer

import matplotlib
import numpy as np
import sys

# this method is used to create the dendrogram that shows which CBs are closer to each other
def create_dendrogram (filename, distance_matrix, labels, inverse_score):
    condensed_matrix = []
    
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 

    # condenses the distance matrix into one dimension, according if the score needs to be inversed or not
    # that is, if the higher the score the more similar the elements, the score needs to be inversed (i.e., 1/score)
    # so that the most similar elements have a lower score among them
    if inverse_score == True:
        for i in range(len(distance_matrix)):
            for j in range(i+1, len(distance_matrix)):
                if distance_matrix[i][j] == 0:
                    condensed_matrix.append(1.0)
                else:
                    condensed_matrix.append(1.0 / distance_matrix[i][j])

    else:
        for i in range(len(distance_matrix)):
            for j in range(i+1, len(distance_matrix)):
                condensed_matrix.append(distance_matrix[i][j])

    figs = []
    # linkage methods considered
    # methods = ['ward', 'centroid', 'single', 'complete', 'average']
    methods = ['complete']

    # draws one dendrogram for each linkage method
    # and the distance used is:
    #                     |-> 0,                   if c1 = c2
    #  dist (c1, c2) =    |-> 1,                   if SW(c1, c2) = 0
    #                     |-> 1 / SW(c1, c2),      otherwise.
    #
    # if the score inverse_score == True. If inverse_score == False, the distance used is the Levenshtein distance.

    for method in methods:
        Z = linkage(condensed_matrix, method)
        # print(method)
        fig = plt.figure(figsize=(25, 10))
        fig.suptitle(method, fontsize=20, fontweight='bold')
        dn = dendrogram(Z, leaf_font_size=20, labels=labels)
        # if method == 'complete':
        #     # i = 0
        #     for i, l in enumerate(Z):
        #         print(f'{i+1}: {l}')

        #     print(dn)
        figs.append(fig)

    try:
        # exports each dendrogram drawn in a separate page in the pdf
        # pdf = PdfPages(str("dendro_for_" + filename + ".pdf"))
        pdf = PdfPages(str(filename + ".pdf"))
        for fig in figs:
            pdf.savefig(fig)
        pdf.close()
        return Z
    except Exception as e:
        print(e)
        print("ERROR: unable to create output file with dendrogram.")
        # quit()
        return None
