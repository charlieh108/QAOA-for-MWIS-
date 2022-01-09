# Module Created To Hold all of the Plotting Tools and Graph Generation so I don't have to redo the functions each time


# Graph Attribute 'node_weight' is used to denote node weight
import numpy as np
from numpy import random

import networkx as nx
#matplotlib.use("pgf")

import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.patches as mpatches


############### Graph Generation ########################
def weighted_erdos_graph(nodes, prob, seed =None):
    """Generates an erdos graph with weighted nodes
    https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model
    Node weights randomly assigned with the same seed as the erdos graph
    """
    graph = nx.erdos_renyi_graph(n=nodes, p =prob, seed=seed, directed=False)
    np.random.seed(seed)
    graph_weights = np.random.randint(1,high=11,size =nodes)
    name = str("Erdos Graph "+str(nodes)+" nodes weighted "+str(list(graph_weights)))

    graph.nodes[0]["graph_name"] = name
    for i in range(0,nodes):
        graph.nodes[i]["node_weight"] = graph_weights[i]
    #print(list(graph.nodes(data=True)))
    return graph

def unweighted_erdos_graph(nodes, prob, seed =None):
    """Generates an erdos graph with weighted nodes
    https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model
    Node weights randomly assigned with the same seed as the erdos graph
    """
    graph = nx.erdos_renyi_graph(n=nodes, p =prob, seed=seed, directed=False)
    name = str(" Unweighted Erdos Graph "+str(nodes)+" nodes")
    graph.nodes[0]["graph_name"] = name

    for i in range(0,nodes):
        graph.nodes[i]["node_weight"] = 1
    return graph

def weighted_path_graph(number_of_nodes,graph_weights):
    """
    Creates a weighted path graph of default length three with different weights
    Graph_weights is a list with the desired node weights

    """


    path = nx.path_graph(number_of_nodes)
    name = str("Path Graph: "+str(number_of_nodes)+" nodes weighted "+str(list(graph_weights)))
    path.nodes[0]["graph_name"] = name

    for i in range(0,number_of_nodes):
        path.nodes[i]["node_weight"] = graph_weights[i]

    return path



def unweighted_path_graph(number_of_nodes):

    """
    Creates an evenly weighted path graph of default length three with equal node weights

    """

    path = nx.path_graph(number_of_nodes)
    name = str("Unweighted Path Graph with "+str(number_of_nodes))+ " nodes"
    path.nodes[0]["graph_name"] = name

    for i in range(0,number_of_nodes):
        path.nodes[i]["node_weight"] = 1
    return path


def test_graph(graph_weights=[1,1,100,1,1,1]):


    adj = np.array([[0,1,0,0,0,0],[1,0,1,1,0,0],[0,1,0,0,0,0],[0,1,0,0,1,0],[0,0,0,1,0,1],[0,0,0,0,1,0]])

    G  = nx.from_numpy_matrix(adj)

    for i in range(0,6):
        print(i)
        G.nodes[i]["node_weight"] = graph_weights[i]

    return G


def unweighted_melbourne():

    melbourne = nx.path_graph(15)

    name = "Unweighted Melbourne Chip Graph"
    melbourne.nodes[0]["graph_name"] = name

    list_of_edges_to_add = [ (0,14),  (1,13), (2,12), (3,11), (4,10),(5,9), (6,8)]
    melbourne.remove_edge(6,7)
    melbourne.add_edges_from(list_of_edges_to_add)
    for i in range(0,15):
        melbourne.nodes[i]["node_weight"] = 1
    return melbourne

def unweighted_ibm_5():


        G = nx.Graph()
        G.add_nodes_from([0,1,2,3,4])
        G.add_edges_from([(0, 1), (1, 2),(1,3),(3,4)])

        name = "Unweighted IBM Chip"
        G.nodes[0]["graph_name"] = name

        for i in range(0,5):
            G.nodes[i]["node_weight"] = 1

        return G



def weighted_ibm_5(graph_weights):


        G = nx.Graph()
        G.add_nodes_from([0,1,2,3,4])
        G.add_edges_from([(0, 1), (1, 2),(1,3),(3,4)])

        name = "Weighted IBM Chip with weights" +str(graph_weights)
        G.nodes[0]["graph_name"] = name

        for i in range(0,5):
            G.nodes[i]["node_weight"] = graph_weights[i]

        return G

def melbourne_chip_graph():

    melbourne = nx.path_graph(15)

    list_of_edges_to_add = [ (0,14),  (1,13), (2,12), (3,11), (4,10),(5,9), (6,8)]
    melbourne.remove_edge(6,7)
    melbourne.add_edges_from(list_of_edges_to_add)
    for i in range(0,15):
        melbourne.nodes[i]["node_weight"] = 1
    return melbourne


####################  Graph Drawing ########################


def draw_sorted_graph(Graph,colours):

    '''

    Plots a graph using the networkx library.

    and uses positions based off of Graph object, and positions  using

    nx.spring_layout
    https://stackoverflow.com/questions/56294715/networkx-graph-plot-node-weights node weights

​

    '''

    pos = nx.spring_layout(Graph)
    print(Graph.nodes(data=True))
    default_axes = plt.axes(frameon=True)
    #labels_dict = {n: '#' + str(n) + ';   ' + str(Graph.nodes[n]['node_weight']) for n in Graph.nodes}
    labels_dict = {n: "         " + str(Graph.nodes[n]['node_weight']) for n in Graph.nodes} # adds weight to label , space so not on top of node number which still appears
    labels = nx.draw_networkx_labels(Graph, pos = pos, labels = labels_dict)
    blue_patch = mpatches.Patch(color='blue', label='Members of MWIS')
    brown_patch = mpatches.Patch(color='brown', label='Not Invited')
    plt.legend(handles=[blue_patch,brown_patch])
    nx.draw_networkx(Graph, node_color=colours, node_size=600, alpha=.8, ax=default_axes, pos=pos)

    plt.show()

def draw_unsorted_graph(Graph,colours):

    '''

    Plots a graph using the networkx library.

    and uses positions based off of Graph object, and positions  using

    nx.spring_layout
    https://stackoverflow.com/questions/56294715/networkx-graph-plot-node-weights node weights

​

    '''

    pos = nx.spring_layout(Graph)
    print(Graph.nodes(data=True))
    default_axes = plt.axes(frameon=True)
    #labels_dict = {n: '#' + str(n) + ';   ' + str(Graph.nodes[n]['node_weight']) for n in Graph.nodes}
    labels_dict = {n: "         " + str(Graph.nodes[n]['node_weight']) for n in Graph.nodes} # adds weight to label , space so not on top of node number which still appears
    labels = nx.draw_networkx_labels(Graph, pos = pos, labels = labels_dict)
    nx.draw_networkx(Graph, node_color=colours, node_size=600, alpha=.8, ax=default_axes, pos=pos)

    plt.show()



def sorted_colours(G):

    colours = []
    for node in G:
         # gets the node weight
        in_set = G.nodes[node]['in_maximum_set']

        if in_set == 1:
            colours.append('blue')

        elif in_set ==0:
            colours.append('brown')
    return colours

def standard_colours(Graph, colour='brown'):

    colours = [colour for node in Graph.nodes()]

    return colours
