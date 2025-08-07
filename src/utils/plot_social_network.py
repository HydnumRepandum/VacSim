import matplotlib.pyplot as plt
import networkx as nx
import pickle
from matplotlib.colors import ListedColormap
import numpy as np
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--social_network", type=str, default="nx_social_network.pkl")
    args=parser.parse_args()
    # Load the graph
    with open(args.social_network, "rb") as f:
        G = pickle.load(f)

    # Compute the communities
    communities = nx.community.louvain_communities(G)

    # Assign a color to each community
    colors = plt.cm.tab20(np.linspace(0, 1, len(communities)))  # Use a colormap
    node_color_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_color_map[node] = colors[idx]

    # Get the node colors in the correct order
    node_colors = [node_color_map[node] for node in G.nodes()]

    # Create a community-based position layout
    pos = nx.spring_layout(G, seed=42)  # Initial spring layout
    for idx, community in enumerate(communities):
        # Adjust positions to bring nodes in the same community closer
        shift = np.random.rand(2) * 0.1 - 0.05  # Slight random offset to separate communities
        for node in community:
            pos[node] += shift  # Move nodes in the same community closer to each other

    # Plot the graph
    fig = plt.figure(figsize=(40, 40))
    nx.draw(
        G,
        pos=pos,
        with_labels=True,
        node_size=200,
        font_size=8,
        linewidths=0.08,
        width=0.08,
        node_color=node_colors
    )
    plt.show()
    plt.savefig("test-img-sc.png")
