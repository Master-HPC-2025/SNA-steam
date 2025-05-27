import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import networkx.algorithms.community as nx_comm

def read_matrix_market_graph(file_path):
    """ Reads an MTX file and converts it into a NetworkX graph with relabeled nodes """
    matrix = scipy.io.mmread(file_path)
    G = nx.from_scipy_sparse_array(matrix)

    # Relabel nodes to ensure continuous indexing
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    return G

def visualize_network(G, file, layout_type="spring"):
    """Clusters the network and visualizes it with distinct colors"""
    
    # Clustering (grouping the nodes into comunities)
    communities = list(nx_comm.greedy_modularity_communities(G))
    color_map = {}
    for i, community in enumerate(communities):
        for node in community:
            color_map[node] = i

    # custom_palette = ["red", "#4A90E2", "orange", "green", "cyan", "yellow", "brown", "lime"]
    # colors = [custom_palette[color_map[node] % len(custom_palette)] for node in G.nodes()]

    colors = [color_map[node] for node in G.nodes()]

    layout_funcs = {
        "spring": nx.spring_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "spectral": nx.spectral_layout,
        "shell": nx.shell_layout,
        "random": nx.random_layout,
        "planar": nx.planar_layout,
        "fruchterman_reingold": nx.fruchterman_reingold_layout
    }
    
    pos = layout_funcs.get(layout_type, nx.spring_layout)(G)

    fig = plt.figure(figsize=(20, 20))
    plt.axis('off')
    fig.canvas.manager.set_window_title(file.name)

    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.get_cmap("tab10"), node_size=50, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color="black", width=0.3, alpha=0.9)
    plt.suptitle(f"{file.name} - {layout_type.capitalize()} Layout", fontsize=40, fontweight="bold")
    
    # plt.show()

    # Save the plots in Plots/ file.name
    # if this folder does not exists then it creates it
    directory = f"Plots/{file.name}"
    if not os.path.exists(directory):
        os.makedirs(directory)
 
    plt.savefig(f"{directory}/{file.name}-{layout_type.capitalize()} Layout.png")


for file in os.scandir("network"): # .venv in root dir
    if '.mtx' not in file.name:
        continue
    
    # load the mtx file
    matrix = scipy.io.mmread(file.path)

    # convert the matrix to a NetworkX graph
    G = nx.from_scipy_sparse_array(matrix)
    
    print(file.name)

    visualize_network(G, file, layout_type="spring")
    visualize_network(G, file, layout_type="spectral")
    visualize_network(G, file, layout_type="kamada_kawai")
    visualize_network(G, file, layout_type="fruchterman_reingold")
    visualize_network(G, file, layout_type="shell")
    visualize_network(G, file, layout_type="random")
    print(end="\n\n")