# #!/usr/bin/python3
# # -*- coding: latin-1 -*-
# # analyze.py
# # Author: Caio Batista de Melo
# # Date created: 2018-06-04
# # Last modified: 2018-08-07
# # Description: This scripts takes a list of ISs traces as input and exports results
# #              that helps the user to analyze what similarities those ISs have.
# #
# # New addition (2018-08-07): added the Levenshtein distance to compare CBs.
# #
# # Note: to time this script, simply remove all '##', the lines to calculate and output
# #       times are commented that way and they are correctly idented already, so simply
# #       removing those double hashes should do the trick.

# from anytree import Node
# from matplotlib import pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# from scipy.cluster.hierarchy import dendrogram, linkage
# from timeit import default_timer as timer

# import matplotlib
# import numpy
# import sys

# # this class contains information of a single IS trace
# class Trace(object):
#     _id = None
#     messages = None
#     messages_loopless = None

#     def __init__ (self, string):
#         index = string.index(':')
#         self._id = int(float(string[:index]))
#         self.messages = []
#         data = string[index+2:]
#         data = data[1:-1]
#         data = data.split()
#         for d in data:
#             if d[-1] == ',':
#                 self.messages.append(d[:-1])
#             else:
#                 self.messages.append(d)
#         self.removeLoops()

#     def __str__ (self):
#         return str(self._id) + ": " + str(self.messages)

#     # this method is used to remove repeated messages from the trace
#     def removeLoops(self):
#         nodes = []
#         nodes_names = []
#         length = len(self.messages)

#         for i, message in zip(range(length), self.messages):          
#             if (message in nodes_names) and (i != length-1):
#                 while message != nodes_names[-1]:
#                     current_node = nodes[-1]
#                     current_node.parent = None
#                     nodes.pop(-1)
#                     nodes_names.pop(-1)
#             else:
#                 current_node = Node(message)
#                 if i == 0:
#                     nodes.append(current_node)
#                     nodes_names.append(message)
#                 else:
#                     current_node.parent = nodes[-1]
#                     nodes.append(current_node)
#                     nodes_names.append(message)

#         self.messages_loopless = nodes_names

#     def popFirst(self):
#         self.messages_loopless.pop(0)

# # this class is used to calculate the Smith-Waterman scores
# class Smith_Waterman(object):
#     match = None
#     mismatch = None
#     gap = None

#    	# default values for the algorithm can be changed when constructing the object
#     def __init__ (self, match_score = 3, mismatch_penalty = -3, gap_penalty = -2):
#         self.match = match_score
#         self.mismatch = mismatch_penalty
#         self.gap = gap_penalty

#     # main method to compare two sequences with the algorithm
#     def compare (self, sequence_1, sequence_2, index1, index2):
#         rows = len(sequence_1) + 1
#         cols = len(sequence_2) + 1
#         # first we calculate the scoring matrix
#         scoring_matrix = numpy.zeros((rows, cols))
#         for i, element_1 in zip(range(1, rows), sequence_1):
#             for j, element_2 in zip(range(1, cols), sequence_2):
#                 similarity = self.match if element_1 == element_2 else self.mismatch
#                 scoring_matrix[i][j] = self._calculate_score(scoring_matrix, similarity, i, j)
        
#         # now we find the max value in the matrix
#         score = numpy.amax(scoring_matrix)
#         index = numpy.argmax(scoring_matrix)
#         # and decompose its index into x, y coordinates
#         x, y = int(index / cols), (index % cols)

#         # now we traceback to find the aligned sequences
#         # and accumulate the scores of each selected move
#         alignment_1, alignment_2 = [], []
#         DIAGONAL, LEFT= range(2)
#         gap_string = "#GAP#"
#         while scoring_matrix[x][y] != 0:
#             move = self._select_move(scoring_matrix, x, y)
            
#             if move == DIAGONAL:
#                 x -= 1
#                 y -= 1
#                 alignment_1.append(sequence_1[x])
#                 alignment_2.append(sequence_2[y])
#             elif move == LEFT:
#                 y -= 1
#                 alignment_1.append(gap_string)
#                 alignment_2.append(sequence_2[y])
#             else: # move == UP
#                 x -= 1
#                 alignment_1.append(sequence_1[x])
#                 alignment_2.append(gap_string)

#         # now we reverse the alignments list so they are in regular order
#         alignment_1 = list(reversed(alignment_1))
#         alignment_2 = list(reversed(alignment_2))

#         return SW_Result([alignment_1, alignment_2], score, [sequence_1, sequence_2], [index1, index2])

#     # inner method to assist the calculation
#     def _calculate_score (self, scoring_matrix, similarity, x, y):
#         max_score = 0
#         try:
#             score = similarity + scoring_matrix[x - 1][y - 1]
#             if score > max_score:
#                 max_score = score                    
#         except:
#             pass
#         try:
#             score = self.gap + scoring_matrix[x][y - 1]
#             if score > max_score:
#                 max_score = score
#         except:
#             pass
#         try:
#             score = self.gap + scoring_matrix[x - 1][y]
#             if score > max_score:
#                 max_score = score
#         except:
#             pass
#         return max_score

#     # inner method to assist the calculation
#     def _select_move (self, scoring_matrix, x, y):
        
#         scores = []
#         try:
#             scores.append(scoring_matrix[x-1][y-1])
#         except:
#             scores.append(-1)
#         try:
#             scores.append(scoring_matrix[x][y-1])
#         except:
#             scores.append(-1)
#         try:
#             scores.append(scoring_matrix[x-1][y])
#         except:
#             scores.append(-1)

#         max_score = max(scores)
#         return scores.index(max_score)

# # this class contains the results of the SW applied to a pair of CBs
# # it's only used to export the results
# class SW_Result(object):
#     aligned_sequences = None
#     traceback_score = None
#     elements = None
#     indices = None

#     def __init__ (self, sequence, score, compared_sequences, indices):
#         self.aligned_sequences = sequence
#         self.traceback_score = score
#         self.elements = compared_sequences
#         self.indices = indices

#     def __str__ (self):
#         out = "Alignment of\n\t" + str(self.indices[0]) +  ": " + str(self.elements[0]) + "\nand\n\t"
#         out += str(self.indices[1]) +  ": " +str(self.elements[1]) + ":\n\n\n"

#         for sequence in self.aligned_sequences:
#             out += "\t" + str(sequence) + "\n"
#         out += "\n\tScore: " + str(self.traceback_score)
        
#         return out

#     def __gt__(self, result_2):
#         return self.traceback_score > result_2.traceback_score

#     def __eq__(self, result_2):
#         return self.traceback_score == result_2.traceback_score

# # this class calculates and exports the Levenshtein distance between two elements
# class Levenshtein (object):
#     cb1 = None
#     cb2 = None
#     index1 = None
#     index2 = None
#     _distance = None
#     _calculated = None

#     # gets the elements that will be compared and sets that the distance has not been calculated yet
#     def __init__ (self, cb1, cb2, index1 = 0, index2 = 0):
#         self.cb1 = cb1
#         self.cb2 = cb2
#         self.index1 = index1
#         self.index2 = index2
#         self._calculated = False

#     # method to return the result as a string
#     def __str__ (self):
#         out = "Levenshtein distance between\n\t" + str(self.index1) +  ": " + str(self.cb1) + "\nand\n\t"
#         out += str(self.index2) +  ": " + str(self.cb2) + "\nis " + str(self.get_distance())
#         return out

#     # method used to allow the comparison between various Levenshtein distances
#     def __gt__(self, Lev2):
#         return self.get_distance() > Lev2.get_distance()

#     # method used to allow the comparison between various Levenshtein distances
#     def __eq__(self, Lev2):
#         return self.get_distance() == Lev2.get_distance()

#     # returns the Levenhstein distance between the elements cb1 and cb2
#     def get_distance (self):
#         if not self._calculated:
#             self._calculate()
#             self._calculated = True

#         return self._distance

#     # calculates the distance, that is, the number of required changes to go from cb1 to cb2
#     def _calculate (self):
#         # there will be len(cb1)+1 columns and len(cb2)+1 rows
#         cols, rows = len(self.cb1)+1, len(self.cb2)+1
#         mx = numpy.zeros((rows, cols))

#         # loop that populate the entire matrix
#         # j will go over the rows (i.e., cb2 elements) and i over the columns (i.e., cb1 elements)
#         # NOTE: numpy uses [colum][row] as the order of indices!
#         for j in range(rows):
#             for i in range(cols):

#                 # intializes the first row
#                 if i == 0:
#                     mx[j][0] = j

#                 # intializes the first column
#                 elif j == 0:
#                     mx[0][i] = i

#                 # calculates the other values in the matrix (i.e., values that are neither in the first column or in the first row)
#                 else:
#                     # checks if the current elements (cb1[i-1] and cb2[j-1]) are equal or not
#                     # if they are, 0 will be added to the distance, as there is no need for another change,
#                     # if they are not, however, 1 will be added, as a new change is required
#                     new_change = 0 if self.cb1[i-1] == self.cb2[j-1] else 1

#                     # finds the minimum changes among the 3 previous adjacent positions (i.e., [j-1][i-1], [j-1][i], and [j][i-1]) + the cost to get this position,
#                     # the cost is 1 if it is not the diagonal neighbor (i.e., [j-1][i-1]), because it doesn't represents an insertion in one of the sequences; the cost is
#                     # new_change if it is diagonal, because it will represent either an insertion (cost 0) or substitution (cost 1).
#                     number_of_changes = [mx[j][i-1]+1, mx[j-1][i-1]+new_change, mx[j-1][i]+1]
                    
#                     # then sets the current number of changes ([j][i]) as the minimum number among the ones calculated above
#                     mx[j][i] = min(number_of_changes)


#         # after populating the whole matrix, sets the Levenshtein distance,
#         # which is the last value calculated in the matrix (i.e., bottom right value)
#         self._distance = mx[rows-1][cols-1]


# # this method is used to create the dendrogram that shows which CBs are closer to each other
# def create_dendrogram (filename, distance_matrix, inverse_score):
#     condensed_matrix = []
    
#     matplotlib.rc('xtick', labelsize=20) 
#     matplotlib.rc('ytick', labelsize=20) 

#     # condenses the distance matrix into one dimension, according if the score needs to be inversed or not
#     # that is, if the higher the score the more similar the elements, the score needs to be inversed (i.e., 1/score)
#     # so that the most similar elements have a lower score among them
#     if inverse_score == True:
#         for i in range(len(distance_matrix)):
#             for j in range(i+1, len(distance_matrix)):
#                 if distance_matrix[i][j] == 0:
#                     condensed_matrix.append(1.0)
#                 else:
#                     condensed_matrix.append(1.0 / distance_matrix[i][j])

#     else:
#         for i in range(len(distance_matrix)):
#             for j in range(i+1, len(distance_matrix)):
#                 condensed_matrix.append(distance_matrix[i][j])

#     figs = []
#     # linkage methods considered
#     methods = ['ward', 'single', 'complete', 'average']

#     # draws one dendrogram for each linkage method
#     # and the distance used is:
#     #                     ┌ 0,                   if cb1 = cb2
#     #  dist (cb1, cb2) =  ├ 1,                   if SW(cb1, cb2) = 0
#     #                     └ 1 / SW(cb1, cb2),    otherwise.
#     #
#     # if the score inverse_score == True. If inverse_score == False, the distance used is the Levenshtein distance.

#     for method in methods:
#         Z = linkage(condensed_matrix, method)
#         fig = plt.figure(figsize=(25, 10))
#         fig.suptitle(method, fontsize=20, fontweight='bold')
#         dn = dendrogram(Z, leaf_font_size=20)
#         figs.append(fig)

#     try:
#         # exports each dendrogram drawn in a separate page in the pdf
#         pdf = PdfPages(str("dendro_for_" + filename + ".pdf"))
#         for fig in figs:
#             pdf.savefig(fig)
#         pdf.close()
#     except:
#         print("ERROR: unable to create output file with dendrogram.")
#         quit()

# # this method parses the traces from the input file
# def parse_traces (filename):
#     file_in = None
#     try:
#         file_in = open(filename, "r")
#     except:
#         print("ERROR: unable to open '" + filename + "'!") 
#         quit()

#     text_traces = file_in.read().split("\n")
#     file_in.close()

#     traces = []

#     # includes only lines formatted as ISs export from our modified LTSA-MSC
#     for text_trace in text_traces:
#         if len(text_trace) > 1 and text_trace[0].isdigit():
#             t = Trace(text_trace)
#             traces.append(t)

#     return traces

# # this method removes the extension of a filename
# def remove_extension (filename):
#     last_period = filename.rfind(".")
#     return filename[:last_period]

# # this method applies the SW algorithm to all detected common behaviors (CBs)
# # it returns the matrix where each element A(i,j) contains the SW score between
# # CB[i] and CB[j], with i!=j
# def compare_sequences (CBs):
#     results = []
#     matrix = [[0 for x in range(len(CBs))] for y in range(len(CBs))] 
#     SW = Smith_Waterman()
#     for i, cb1 in zip(range(len(CBs)), CBs):
#         for j, cb2 in zip(range(i+1, len(CBs)), CBs[i+1:]):
#             new_result = SW.compare(cb1, cb2, i, j)
#             matrix[j][i] = matrix[i][j] = new_result.traceback_score
#             results.append(new_result)

#     return list(sorted(results, reverse = True)), matrix

# # this method calculates the Levenshtein distance between all pairs of detected common behaviors (CBs)
# # it returns the matrix where each element A(i,j) contains the distance score between CBi and CBj
# def compare_sequences_lev (CBs):
#     results = []
#     matrix = [[0 for x in range(len(CBs))] for y in range(len(CBs))] 
#     for i, cb1 in zip(range(len(CBs)), CBs):
#         for j, cb2 in zip(range(i+1, len(CBs)), CBs[i+1:]):
#             lev = Levenshtein(cb1, cb2, i, j)
#             matrix[j][i] = matrix[i][j] = lev.get_distance()
#             results.append(lev)

#     return list(sorted(results, reverse = False)), matrix

# # here we export the test cases to be used with the check_scenarios.py file
# # it groups all analyzed ISs by common behavior, this way it is possible to see
# # that ISs in the same CB are being treated together and if all ISs are being resolved
# def export_test_cases (common_behaviors, filename):
#     out = "! Test file for " + filename + "\n\n"

#     out += "!\n! Here you can manually add the expected behaviors to check that they're still reachable.\n!\n! You can generate them with the model_parser.py script.\n!\n"
    
#     for i, common_behavior in zip(range(len(common_behaviors)), common_behaviors):
#         out += "\n### Common Behavior " + str(i) + "\n\n"
#         out += "!\n! " + str(common_behavior[0][0].messages_loopless) + "\n!\n\n"

#         for scenario in common_behavior:
#             messages = str(scenario[0].messages).replace('[', '').replace(']', '').replace("'", "")
#             out += str(scenario[1]) + ": " + messages + "\n"
#     try:
#         filename = str("tests_" + filename + ".txt")
#         file_out = open(filename, "w")
#         file_out.write(out)
#         file_out.close()
#     except:
#         pass

# # this is the main method to analyze the input data
# def analyze_main (input_file, traces):
#     CBs = []

#     # we find the unique common behaviors among the given traces
#     for i, trace in zip(range(len(traces)), traces):
#         new_CB = True
#         for CB in CBs:
#             if CB[0][0].messages_loopless == trace.messages_loopless:
#                 CB.append([trace, i])
#                 new_CB = False
#         if new_CB:
#             CBs.append([[trace, i]])
        
#     # and export them
#     filename = str("CBs_for_" + input_file + ".txt")

#     output = "Common Behaviors for " + input_file + ":\n"
#     for i, CB in zip(range(len(CBs)), CBs):
#         output += "\n"
#         output += "  " + str(i) + ": " + str(CB[0][0].messages_loopless)

#     try:
#         file_out = open(filename, "w")
#         file_out.write(output)
#         file_out.close()
#     except:
#         print("ERROR: unable to create output file!") 
#         quit()

#     # now we generate the test file for the check_scenarios.py
#     export_test_cases(CBs, input_file)

#     # now we analyze all possible pairs of CBs with the SW algorithm
#     cbs_only = []
#     for CB in CBs:
#         cbs_only.append(CB[0][0].messages_loopless)

#  	# all comments with double # are lines to measure time of the script
#  	# if it's desired to time a run, simply remove all '##'
#     ##start_sw = timer()
#     results, score_matrix = compare_sequences(cbs_only)
    
#     output = "CB"
#     for i in range(len(cbs_only)):
#         output += ", " + str(i)

#     for i in range(len(cbs_only)):
#         output += "\n" + str(i)
#         for j in range(len(cbs_only)):
#             output += ", " + str(score_matrix[i][j])

#     try:
#         filename = str("sw_for_" + input_file + ".csv")
#         file_out = open(filename, "w")
#         file_out.write(output)
#         file_out.close()
#     except:
#         pass

#     output = ""

#     for i, result in zip(range(len(results)), results):
#         if i > 0:
#             output += "\n\n--- --- --- --- ---\n\n"
#         output += str(result)

#     try:
#         filename = str("sw_for_" + input_file + ".txt")
#         file_out = open(filename, "w")
#         file_out.write(output)
#         file_out.close()
#     except:
#         pass
    
#     ##end_sw = timer()
#     ##print("Time for SW: %fs" % (end_sw-start_sw))

#     try:
#         ##start_dendro = timer()
#         create_dendrogram(str('sw_' + input_file), score_matrix, inverse_score=True)
#         ##print("Time for Dendrogram: %fs" % (timer()-start_dendro))
#     except:
#         print("Unable to export a dendrogram.")

#     ##return (timer() - start_sw)

#     # now we do the same thing but using the Levenhstein distance intead of the Smith-Waterman algorithm
#     results, score_matrix = compare_sequences_lev(cbs_only)
    
#     output = "CB"
#     for i in range(len(cbs_only)):
#         output += ", " + str(i)

#     for i in range(len(cbs_only)):
#         output += "\n" + str(i)
#         for j in range(len(cbs_only)):
#             output += ", " + str(score_matrix[i][j])

#     try:
#         filename = str("lev_for_" + input_file + ".csv")
#         file_out = open(filename, "w")
#         file_out.write(output)
#         file_out.close()
#     except:
#         pass

#     output = ""

#     for i, result in zip(range(len(results)), results):
#         if i > 0:
#             output += "\n\n--- --- --- --- ---\n\n"
#         output += str(result)

#     try:
#         filename = str("lev_for_" + input_file + ".txt")
#         file_out = open(filename, "w")
#         file_out.write(output)
#         file_out.close()
#     except:
#         pass

#     #try:
#     create_dendrogram(str('lev_' + input_file), score_matrix, inverse_score=False)
#     #except:
#     #    print("Unable to export a dendrogram.")

# # this main method simply checks if the files have been passed.
# # it can also time the run of the analysis.
# def main():
#     ##start = timer()
#     if len(sys.argv) < 2:
#         print("ERROR: a file should be passed as argument!")
#         print("Example of usage: python3 analyze.py traces.txt")
#         quit()
#     traces = parse_traces(sys.argv[1])
#     sw_time = analyze_main(remove_extension(sys.argv[1]), traces)
#     ##end = timer()
#     ##total = end-start
#     ##find_CBs = total - sw_time
#     ##print("Time for CBs: %fs" % find_CBs)
#     ##print("Total Time: %fs" % total)

# if __name__ == "__main__":
#     main()