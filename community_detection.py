import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import matplotlib.cm as cm 
import seaborn as sns
import pandas as pd

def load_data_to_graph(file):
    f = open(file, 'r')

    G = nx.read_edgelist(file, create_using=nx.DiGraph())
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    return G


def louvain_community_detection(G):
    # Assuming G is your graph
    G = G.to_undirected()

    # Compute the best partition using Louvain method
    partition = community_louvain.best_partition(G)

    # Visualize the communities
    pos = nx.spring_layout(G)
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    for node, color in partition.items():
        nx.draw_networkx_nodes(G, pos, [node], node_size=50,
                            node_color=[cmap.colors[color]])
    plt.savefig('louvain_community_visualization.png')

    # Compute the modularity
    louvain_modularity_score = community_louvain.modularity(partition, G)
    print("Louvain modularity score: ", louvain_modularity_score)

    # Compute the degree distribution
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    plt.figure()
    plt.loglog(degree_sequence, 'b-', marker='o')
    plt.title("Degree distribution")
    plt.ylabel("Degree")
    plt.xlabel("Rank")
    plt.savefig('louvain_degree_distribution.png')

    return partition, louvain_modularity_score

def girvan_community_detection(G):
    # Assuming G is your graph
    communities_generator = nx.community.girvan_newman(G)
    communities = list(communities_generator)

    # Modularity -> measures the strength of division of a network into modules
    modularity_df = pd.DataFrame(
        [
            [k + 1, nx.community.modularity(G, communities[k])]
            for k in range(len(communities))
        ],
        columns=["k", "modularity"],
    )
    # Plot change in modularity as the important edges are removed
    modularity_df.plot.bar(
        x="k",
        color="#F2D140",
        title="Modularity Trend for Girvan-Newman Community Detection",
    )
    plt.savefig('girvan_communities.png')
    # Run Girvan-Newman algorithm

    # Find the best partition with the highest modularity score
    best_partition = None
    best_modularity = float('-inf')

    for partition in communities_generator:
        modularity = nx.community.modularity(G, partition)
        if modularity > best_modularity:
            best_modularity = modularity
            best_partition = partition

    # Print the best partition and its modularity score
    print("Best partition:", best_partition)
    print("Modularity:", best_modularity)




def clustering_community_detection(G):
    # Create adjacency matrix
    adj_mat = nx.adjacency_matrix(G).toarray()

    # Perform Spectral Clustering
    sc = SpectralClustering(affinity='precomputed', n_clusters=10, assign_labels='discretize')
    sc.fit(adj_mat)

    # Get labels and create a dictionary of node-community pairs
    labels = sc.labels_
    community_dict = {node: label for node, label in enumerate(labels)}

    # Visualize the communities
    pos = nx.spring_layout(G)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_color=labels)

    # Save the community visualization
    plt.savefig('clustering_community_visualization.png')

    # Compute the modularity
    clustering_modularity_score = metrics.normalized_mutual_info_score(labels, labels)
    print("Clustering modularity score: ", clustering_modularity_score)

    # Compute the degree distribution
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    plt.figure()
    plt.loglog(degree_sequence, 'b-', marker='o')
    plt.title("Degree distribution")
    plt.ylabel("Degree")
    plt.xlabel("Rank")
    plt.savefig('clustering_degree_distribution.png')

    return community_dict, clustering_modularity_score

# Loading the dataset
data_file = 'Influence-Maximization-Analysis/data/wiki-Vote.txt'
G = load_data_to_graph(data_file)

# Checking the basic properties
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Is the graph directed: {G.is_directed()}")

res = {}

# Louvain algorithm
louvain_communities, louvain_modularity_score = louvain_community_detection(G)
print('communities:' + str(set(list(louvain_communities.values()))))
print('modularity ' + str(louvain_modularity_score))
res['louvain'] = [louvain_communities, louvain_modularity_score]

# Spectral clustering algorithm
clustering_communities, clustering_modularity_score = clustering_community_detection(G)
print('communities:' + str(set(list(clustering_communities.values()))))
print('modularity '+ str(clustering_modularity_score))
res['spectral'] = [clustering_communities, clustering_modularity_score]

# girvan_community_detection(G)

modularity_scores = {
    'Louvain': res['louvain'][1],
    'Spectral': res['spectral'][1],
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame([modularity_scores], index=['Modularity'])
print(df)

# Create the heatmap
# Create bar chart
plt.figure()
plt.bar(modularity_scores.keys(), modularity_scores.values(), color=['blue', 'orange'])
# Add labels and title
plt.xlabel('Algorithm')
plt.ylabel('Modularity Score')
plt.title('Modularity Comparison')
# Save the heatmap
plt.savefig('modularity_comparison_heatmap.png')
from collections import Counter
# Get the size of each community
community_sizes = Counter(louvain_communities.values())

# Convert the data into a DataFrame
df = pd.DataFrame({'Community': list(community_sizes.keys()), 'Size': list(community_sizes.values())})

# Save the DataFrame to a CSV file
df.to_csv('louvain_community_sizes.csv', index=False)

community_sizes = Counter(clustering_communities.values())

# Convert the data into a DataFrame
df = pd.DataFrame({'Community': list(community_sizes.keys()), 'Size': list(community_sizes.values())})

# Save the DataFrame to a CSV file
df.to_csv('spectral_community_sizes.csv', index=False)



