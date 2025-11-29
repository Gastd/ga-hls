from __future__ import annotations

from pathlib import Path

# Root of the installed ga_hls package
_PACKAGE_ROOT = Path(__file__).resolve().parent

# Default ThEodorE Python property script used by legacy GA/Z3 integration.
# This is still a legacy path; the HLS file is now the canonical requirement
# source, but GA viability checks rely on this Python script.
FILEPATH = str(_PACKAGE_ROOT / "benchmark" / "property_distance_obs_r2.py")
FILEPATH2 = FILEPATH

CROSSOVER_RATE = 0.95 ## Rate defined by Núnez-Letamendia
MUTATION_RATE = 0.9  ## Rate defined by Núnez-Letamendia
POPULATION_SIZE = 50 #30  ## Must be an EVEN number
# GENE_LENGTH = 32
# MAX_ALLOWABLE_GENERATIONS = 1000 #10 #616 ##Calculated using ALANDER , J. 1992. On optimal population size of genetic algorithms.
# MAX_ALLOWABLE_GENERATIONS = 3 #10 #616 ##Calculated using ALANDER , J. 1992. On optimal population size of genetic algorithms.
# NUMBER_OF_PARAMETERS = 17 ## Number of parameters to be evolved
# CHROMOSOME_LENGTH = GENE_LENGTH * NUMBER_OF_PARAMETERS
CHROMOSOME_TO_PRESERVE = 0 #4            ## Must be an EVEN number
PARENTS_TO_BE_CHOSEN = 10

SW_THRESHOLD = 35
FOLDS = 10

SCALE = 0.5