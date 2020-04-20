import numpy as np
import networkx as nx


def edmonds(scores):
    """
    Find the most likely arc-in for every word.
        :param scores: n by n numpy array score matrix
        :param sent: sentence (list of strings) including 'root' at the beginning
        :param labels: list of labels (strings)
    """
    graph = nx.from_numpy_matrix(scores, create_using=nx.DiGraph())
    tree = maximum_spanning_tree(graph, 0)
    arc_in = np.zeros(len(scores))
    for edge in tree.edges():
        for node in tree.nodes():
            if edge[1] == node:
                arc_in[node] = edge[0]
    return list(arc_in.astype(int))

    
def contract(graph, cycle, cycle_node):
    """
    Replace cycle in directed graph by a node.
        :param graph: networkx DiGraph()
        :param cycle: list of nodes which form a cycle in the graph
        :param cycle_node: node which will replace the cycle
    """
    # create new directed graph with nodes
    contracted_graph = nx.DiGraph()
    contracted_graph.add_nodes_from(graph)
    contracted_graph.remove_nodes_from(cycle)
    contracted_graph.add_node(cycle_node)
    
    # add the edges in right form
    # remember the corresponding edge from graph
    # calculate the weight of this edge
    for (u,v) in graph.edges():
        if u not in cycle and v in cycle:
            contracted_graph.add_edge(u, cycle_node)
            best_node = arg_max(graph, v)
            contracted_graph[u][cycle_node]['edge'] = (u, v)
            contracted_graph[u][cycle_node]['weight'] = graph[u][v]['weight'] - graph[best_node][v]['weight']
        elif u in cycle and v not in cycle:
            contracted_graph.add_edge(cycle_node, v)
            contracted_graph[cycle_node][v]['edge'] = (u, v)
            contracted_graph[cycle_node][v]['weight'] = graph[u][v]['weight']
        elif u not in cycle and v not in cycle:
            contracted_graph.add_edge(u, v)
            contracted_graph[u][v]['edge'] = (u, v)
            contracted_graph[u][v]['weight'] = graph[u][v]['weight']
    return contracted_graph


def expand(graph, contracted_tree, cycle, cycle_node):
    """
    Find tree by replacing nodes of a contracted tree by cycle.
        :param graph: networkx DiGraph()
        :param cycle: list of nodes which form a cycle in the graph
        :param cycle_node: node which will replace the cycle
    """
    # create new directed graph with nodes
    tree = nx.DiGraph()
    tree.add_nodes_from(graph)
    
    # add the edges in right form
    # calculate the weight of this edge
    for (u,v) in contracted_tree.edges():
        
        # add corresponding edge from the complete graph with right weight
        (m, n) = contracted_tree[u][v]['edge']
        tree.add_edge(m, n)
        tree[m][n]['weight'] = graph[m][n]['weight']
        tree[m][n]['edge'] = graph[m][n]['edge']
        
        # add all cycle egdes expect for (pi(v),v) with right weight
        if v == cycle_node:
            cycle_prime = [i for i in cycle]
            cycle_prime.remove(n)
            best_node = arg_max(graph, n, nodes = cycle_prime)
            if not(cycle[-1] == best_node and cycle[0]==n):
                tree.add_edge(cycle[-1], cycle[0])
                tree[cycle[-1]][cycle[0]]['weight'] = graph[cycle[-1]][cycle[0]]['weight']
                tree[cycle[-1]][cycle[0]]['edge'] = graph[cycle[-1]][cycle[0]]['edge']
            for i in range(len(cycle)-1):
                if not(cycle[i] == best_node and cycle[i+1]==n):
                    tree.add_edge(cycle[i], cycle[i+1])
                    tree[cycle[i]][cycle[i+1]]['weight'] = graph[cycle[i]][cycle[i+1]]['weight']
                    tree[cycle[i]][cycle[i+1]]['edge'] = graph[cycle[i]][cycle[i+1]]['edge']
    return tree


def arg_max(graph, receiver, nodes=None):
    """
    Find tuple which represents edge with maximal weight and this maximal weight.
         :param graph: networkx DiGraph()
         :param receiver: node of graph
         :param nodes: list with nodes of graph
    """
    if nodes == None: nodes = graph.nodes()
    max_score = None
    for node in nodes:
        if (node, receiver) in graph.edges():
            if max_score == None or graph[node][receiver]['weight'] > max_score:
                max_score = graph[node][receiver]['weight']
                best_node = node
    return best_node


def find_cycle(graph):
    """
    Find cycle in graph.
        :param graph: networkx DiGraph()
    """
    try:
        cycle = nx.find_cycle(graph)
    except:
        cycle = []
    return cycle


def maximum_spanning_tree(graph, root):
    """
    Find maximum spanning tree of a directed graph with weights.
        :param graph: networkx DiGraph() with key: weight and nodes are represented by integers
        :param root: node representing the root
    """
    # checks if root is in graph
    if root not in graph.nodes(): raise ValueError("The root is not a node of the graph.")
    
    # remove edges with as destination root
    destination_root = []
    for edge in graph.edges():
        if root == edge[1]: destination_root.append(edge)
    graph.remove_edges_from(destination_root)
        
    # remove reflexive edges
    for node in graph.nodes():
        if (node,node) in graph.edges:
            graph.remove_edge(node,node)
    
    # add edge information
    for (u,v) in graph.edges():
        if 'edge' not in graph[u][v]:
            graph[u][v]['edge'] = (u,v)
    
    tree = nx.DiGraph()
    tree.add_nodes_from(graph)
    for node in graph.nodes():
        if node != root:
            best_node = arg_max(graph, node)
            tree.add_edge(best_node, node)
            tree[best_node][node]['weight'] = graph[best_node][node]['weight']
            tree[best_node][node]['edge'] = graph[best_node][node]['edge']
    
    cycle = find_cycle(tree)
    path = []
    for (n,m) in cycle:
        path.append(n)

    if len(cycle) == 0:
        return tree
    else:
        cycle_node = max(graph.nodes()) + 1
        # eleminate cycle in graph
        contracted_graph = contract(graph, path, cycle_node)
        # find max spanning tree if this smaller graph
        contracted_tree = maximum_spanning_tree(contracted_graph, root)
        # use this to find max spanning tree
        return expand(graph, contracted_tree, path, cycle_node)