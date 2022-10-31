from carpath import CarPath
from individual import Individual

import json
from pgraph import UGraph, DGraph

from ga import GA

from time import sleep

g = UGraph()
# load the JSON file
with open('routes.json', 'r') as f:
    data = json.loads(f.read())

for name, info in data['places'].items():
    g.add_vertex(name=name, coord=info["utm"])

# create an edge for every route, and the cost is the driving distance
for route in data['routes']:
    g.add_edge(route['start'], route['end'], cost=route['distance'])

# print(len(g._vertexlist))
# print(len(g.edges()))

ga = GA(g, 30, 3, [8, 15, 19])
ga.evolve()
ga.show()
solution = ga.get_best()
for i in range(solution.ncars):
    g.plot(block=False)
    g.highlight_path(solution.paths[i].path)

