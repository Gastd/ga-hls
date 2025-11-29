# tool/ga_hls/fitness_smithwaterman.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Any, Protocol
import numpy as np


class Fitness(Protocol):
    """
    Abstract fitness interface: given token sequences for the seed and a
    chromosome, return a numeric score. Higher is better.
    """

    def compute_max_score(self, tokens: Sequence[Any]) -> float:
        """Score of the seed against itself (used for normalization)."""
        ...

    def score(self, seed_tokens: Sequence[Any], chrom_tokens: Sequence[Any]) -> float:
        """Score of a chromosome relative to the seed."""
        ...

# this class contains the results of the SW applied to a pair of CBs
# it's only used to export the results
class SW_Result(object):
    aligned_sequences = None
    traceback_score = None
    elements = None
    indices = None

    def __init__ (self, sequence, score, compared_sequences, indices):
        self.aligned_sequences = sequence
        self.traceback_score = score
        self.elements = compared_sequences
        self.indices = indices

    def __str__ (self):
        out = "Alignment of\n\t" + str(self.indices[0]) +  ": " + str(self.elements[0]) + "\nand\n\t"
        out += str(self.indices[1]) +  ": " +str(self.elements[1]) + ":\n\n\n"

        for sequence in self.aligned_sequences:
            out += "\t" + str(sequence) + "\n"
        out += "\n\tScore: " + str(self.traceback_score)
        
        return out

    def __add__(self, result_2):
        self.traceback_score += result_2.traceback_score
        return self

    def __iadd__(self, result_2):
        return self + result_2

    def __ge__(self, result_2):
        return self.traceback_score >= result_2.traceback_score
    
    def __gt__(self, result_2):
        return self.traceback_score > result_2.traceback_score

    def __le__(self, result_2):
        return self.traceback_score <= result_2.traceback_score

    def __le__(self, result_2):
        return self.traceback_score < result_2.traceback_score

    def __eq__(self, result_2):
        return self.traceback_score == result_2.traceback_score


# this class is used to calculate the Smith-Waterman scores
class Smith_Waterman(object):
    match = None
    mismatch = None
    gap = None

    # default values for the algorithm can be changed when constructing the object
    def __init__ (self, match_score = 3, mismatch_penalty = -3, gap_penalty = -2):
        self.match = match_score
        self.mismatch = mismatch_penalty
        self.gap = gap_penalty

    # main method to compare two sequences with the algorithm
    def compare (self, sequence_1, sequence_2, index1, index2):
        rows = len(sequence_1) + 1
        cols = len(sequence_2) + 1
        # first we calculate the scoring matrix
        scoring_matrix = np.zeros((rows, cols))
        for i, element_1 in zip(range(1, rows), sequence_1):
            for j, element_2 in zip(range(1, cols), sequence_2):
                similarity = self.match if element_1 == element_2 else self.mismatch
                scoring_matrix[i][j] = self._calculate_score(scoring_matrix, similarity, i, j)
        # print(f'\n{scoring_matrix}\n')
        
        # now we find the max value in the matrix
        score = np.amax(scoring_matrix)
        index = np.argmax(scoring_matrix)
        # and decompose its index into x, y coordinates
        x, y = int(index / cols), (index % cols)

        # now we traceback to find the aligned sequences
        # and accumulate the scores of each selected move
        alignment_1, alignment_2 = [], []
        DIAGONAL, LEFT= range(2)
        gap_string = "#GAP#"
        while scoring_matrix[x][y] != 0:
            move = self._select_move(scoring_matrix, x, y)
            
            if move == DIAGONAL:
                x -= 1
                y -= 1
                alignment_1.append(sequence_1[x])
                alignment_2.append(sequence_2[y])
            elif move == LEFT:
                y -= 1
                alignment_1.append(gap_string)
                alignment_2.append(sequence_2[y])
            else: # move == UP
                x -= 1
                alignment_1.append(sequence_1[x])
                alignment_2.append(gap_string)

        # now we reverse the alignments list so they are in regular order
        alignment_1 = list(reversed(alignment_1))
        alignment_2 = list(reversed(alignment_2))

        return SW_Result([alignment_1, alignment_2], score, [sequence_1, sequence_2], [index1, index2])

    # inner method to assist the calculation
    def _calculate_score (self, scoring_matrix, similarity, x, y):
        max_score = 0
        try:
            score = similarity + scoring_matrix[x - 1][y - 1]
            if score > max_score:
                max_score = score                    
        except:
            pass
        try:
            score = self.gap + scoring_matrix[x][y - 1]
            if score > max_score:
                max_score = score
        except:
            pass
        try:
            score = self.gap + scoring_matrix[x - 1][y]
            if score > max_score:
                max_score = score
        except:
            pass
        return max_score

    # inner method to assist the calculation
    def _select_move (self, scoring_matrix, x, y):
        
        scores = []
        try:
            scores.append(scoring_matrix[x-1][y-1])
        except:
            scores.append(-1)
        try:
            scores.append(scoring_matrix[x][y-1])
        except:
            scores.append(-1)
        try:
            scores.append(scoring_matrix[x-1][y])
        except:
            scores.append(-1)

        max_score = max(scores)
        return scores.index(max_score)

class SmithWatermanFitness(Fitness):
    """
    Default fitness implementation: Smith–Waterman over token sequences.
    """

    def __init__(self):
        self._sw = Smith_Waterman()

    def compute_max_score(self, tokens: Sequence[Any]) -> float:
        res = self._sw.compare(tokens, tokens, 0, 0)
        return res.traceback_score

    def score(self, seed_tokens: Sequence[Any], chrom_tokens: Sequence[Any]) -> float:
        res = self._sw.compare(seed_tokens, chrom_tokens, 0, 0)
        return res.traceback_score
