import json
import components.treenode
from components.individual import Individual
from ga import GA

from config.property import property_str

if __name__ == '__main__':
    ga = GA(property_str,'trace.py')
    ga.evolve()

