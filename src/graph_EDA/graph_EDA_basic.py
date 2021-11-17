# this script performs EDA on the full Dublin graph
# assumed wd is project root

import networkx as nx
import matplotlib.pyplot as plt

dub_graph = nx.read_gpickle(".\\data\\dublin_graph.gpickle")

degrees = [dub_graph.degree(i) for i in dub_graph.nodes()]

graph_degree = max(degrees)

# plot distribution of node degrees
plt.figure()
plt.hist(degrees)
plt.yscale("log")
plt.title("Full Dublin Graph Node Degree Distribution")
plt.xlabel("Node Degree")
plt.ylabel("NUmber of Nodes")