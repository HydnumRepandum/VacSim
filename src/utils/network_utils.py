def build_edge_list(similar_agents_idx):
    '''
    :param similar_agents_idx: a list of agents that are similar to each other
    :return: a list of edges between agents in the network
    '''
    edge_list = []
    # all n choose 2
    for i in range(len(similar_agents_idx)):
        for j in range(i+1, len(similar_agents_idx)):
            edge_list.append((similar_agents_idx[i], similar_agents_idx[j]))
    return edge_list

def calculate_homophily(graph, attribute):
    same_attribute_edges = 0
    total_edges = 0
    same_one_edges = 0; same_two_edges = 0; same_three_edges = 0; same_four_edges = 0
    set_one_nodes, set_two_nodes, set_three_nodes, set_four_nodes = set(), set(), set(), set()
    for u, v in graph.edges():
        total_edges += 1
        if graph.nodes[u][attribute] == graph.nodes[v][attribute]:
            same_attribute_edges += 1 # Undirected graph, so count both directions
        if graph.nodes[u][attribute] == 1 and graph.nodes[v][attribute] == 1:
            same_one_edges += 1
            if u not in set_one_nodes:
                set_one_nodes.add(u)
            if v not in set_one_nodes:
                set_one_nodes.add(v)
        if graph.nodes[u][attribute] == 2 and graph.nodes[v][attribute] == 2:
            same_two_edges += 1
            if u not in set_two_nodes:
                set_two_nodes.add(u)
            if v not in set_two_nodes:
                set_two_nodes.add(v)
            
        if graph.nodes[u][attribute] == 3 and graph.nodes[v][attribute] == 3:
            same_three_edges += 1
            if u not in set_three_nodes:
                set_three_nodes.add(u)
            if v not in set_three_nodes:
                set_three_nodes.add(v)

        if graph.nodes[u][attribute] == 4 and graph.nodes[v][attribute] == 4:
            same_four_edges += 1
            if u not in set_four_nodes:
                set_four_nodes.add(u)
            if v not in set_four_nodes:
                set_four_nodes.add(v)

    if total_edges == 0:
        return 0  # Avoid division by zero
    
    
    return same_attribute_edges / total_edges, same_one_edges / total_edges, same_two_edges / total_edges, same_three_edges / total_edges, same_four_edges / total_edges

def homophily_corr(G):
    return calculate_homophily(G, 'attitude')
    
    
