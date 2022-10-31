from carpath import CarPath
import random
from pgraph import UGraph

class Individual():
    """docstring for Individual"""
    def __init__(self, graph: UGraph, ncars: int = 0, positions: list = [], nchecks: int = 0):
        if len(positions) != ncars:
            raise Exception(f"{self.__class__.__name__} error: position list has to have the same length as # of cars: {len(positions)} != {ncars}")

        self.score = 0
        self.ncars = ncars
        self.nchecks = nchecks
        self.graph = graph
        self.positions = positions

        self.init_cars()
        self.create_checkpoints()
        self.make_paths()

    def init_cars(self):
        self.paths = []
        for i in range(self.ncars):
            self.paths.append(CarPath(self.positions[i]))

    def create_checkpoints(self):
        for i in range(self.ncars):
            self.paths[i].create_checkpoints(self.nchecks, self.graph.n)

    def make_paths(self):
        for i in range(self.ncars):
            self.paths[i].make_path(self.graph)

    def get_path(self):
        return self.paths

    def print_genes(self):
        for i in range(self.ncars):
            print(self.paths[i])
