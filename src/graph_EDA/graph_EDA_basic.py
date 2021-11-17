# this script performs EDA on the full Dublin graph
# assumed wd is project root

import networkx as nx
import matplotlib.pyplot as plt

dub_graph = nx.read_gpickle(".\\data\\dublin_graph.gpickle")

# ~~~~~~~~~~~~~~~~~
# degree
# ~~~~~~~~~~~~~~~~~
degrees = [dub_graph.degree(i) for i in dub_graph.nodes()]

graph_degree = max(degrees)

# plot distribution of node degrees
plt.figure()
plt.hist(degrees)
plt.yscale("log")
plt.title("Full Dublin Graph Node Degree Distribution")
plt.xlabel("Node Degree")
plt.ylabel("Number of Nodes")

# ~~~~~~~~~~~~~~~~~
# eccentricity
# ~~~~~~~~~~~~~~~~~

eccentricities = nx.algorithms.distance_measures.eccentricity(dub_graph)

radius = nx.algorithms.distance_measures.radius(dub_graph, e=eccentricities)

diam = nx.algorithms.distance_measures.diameter(dub_graph, e=eccentricities)

centre = nx.algorithms.distance_measures.center(dub_graph, e=eccentricities)