#note: script was modified according to our requirements. 
#Reference: https://github.com/AdnanRasad/Influence-Maximization-Analysis/blob/master/Kempe-Independent-Cascades-Model/Sample-Kempe-IM Adnan Rasad
##submit comparasion between greedy and clef ## ref: https://hautahi.com/im_greedycelf

import random
import networkx as nx
import numpy as np
from tqdm import tqdm
import datetime
import argparse
import matplotlib.pyplot as plt
import igraph as ig
import time

def IC(g,S,p=0.5,mc=1000):
    """
    Input:  graph object, set of seed nodes, propagation probability
            and the number of Monte-Carlo simulations
    Output: average number of nodes influenced by the seed nodes
    """
    
    # Loop over the Monte-Carlo Simulations
    spread = []
    for i in range(mc):
        
        # Simulate propagation process      
        new_active, A = S[:], S[:]
        while new_active:

            # For each newly active node, find its neighbors that become activated
            new_ones = []
            for node in new_active:
                
                # Determine neighbors that become infected
                np.random.seed(i)
                success = np.random.uniform(0,1,len(g.neighbors(node,mode="out"))) < p
                new_ones += list(np.extract(success, g.neighbors(node,mode="out")))

            new_active = list(set(new_ones) - set(A))
            
            # Add newly activated nodes to the set of activated nodes
            A += new_active
            
        spread.append(len(A))
        
    return(np.mean(spread))

def greedy(g,k,p=0.1,mc=1000):
    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """
    print('--'*10+'starting greedy '+'--'*10)
    S, spread, timelapse, start_time = [], [], [], time.time()
    
    # Find k nodes with largest marginal gain
    for _ in range(k):

        # Loop over nodes that are not yet in seed set to find biggest marginal gain
        best_spread = 0
        for j in tqdm(set(range(g.vcount()))-set(S)):

            # Get the spread
            s = IC(g,S + [j],p,mc)

            # Update the winning node and spread so far
            if s > best_spread:
                best_spread, node = s, j

        # Add the selected node to the seed set
        S.append(node)
        
        # Add estimated spread and elapsed time
        spread.append(best_spread)
        timelapse.append(time.time() - start_time)
    print('--'*10+'ending greedy '+'--'*10)

    return(S,spread,timelapse)

def celf(g,k,p=0.1,mc=1000):  
    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """
      
    # --------------------
    # Find the first node with greedy algorithm
    # --------------------
    
    # Calculate the first iteration sorted list
    start_time = time.time() 
    marg_gain = [IC(g,[node],p,mc) for node in range(g.vcount())]

    # Create the sorted list of nodes and their marginal gain 
    Q = sorted(zip(range(g.vcount()),marg_gain), key=lambda x: x[1],reverse=True)

    # Select the first node and remove from candidate list
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS, timelapse = Q[1:], [g.vcount()], [time.time()-start_time]
    
    # --------------------
    # Find the next k-1 nodes using the list-sorting procedure
    # --------------------
    print('--'*10+'starting celf '+'--'*10)
    for _ in tqdm(range(k-1)):    

        check, node_lookup = False, 0
        
        while not check:
            
            # Count the number of times the spread is computed
            node_lookup += 1
            
            # Recalculate spread of top node
            current = Q[0][0]
            
            # Evaluate the spread function and store the marginal gain in the list
            Q[0] = (current,IC(g,S+[current],p,mc) - spread)

            # Re-sort the list
            Q = sorted(Q, key = lambda x: x[1], reverse = True)

            # Check if previous top node stayed on top after the sort
            check = (Q[0][0] == current)

        # Select the next node
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)
        LOOKUPS.append(node_lookup)
        timelapse.append(time.time() - start_time)

        # Remove the selected node from the list
        Q = Q[1:]
    print('--'*10+'end celf '+'--'*10)

    return(S,SPREAD,timelapse,LOOKUPS)

# def seedsGeneration(G,k,a):
#     d=nx.get_node_attributes(G,a)
#     dict_sorted=dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
#     return list(dict_sorted.keys())[:k]

# def IC_with_selected_seeds(graph,num_seed_nodes,prob=0.2, n_iters=100):
#     seed_seclection = ['betweenness', 'out_degree_centrality', 'pagerank', 'closeness_centrality', 'eigenvector_centrality']
    
#     spread_dict = {}
#     for s in seed_seclection:
#         #get seeds 
#         seeds=seedsGeneration(graph,num_seed_nodes,s)
#         print('Number of Seed Nodes: ', num_seed_nodes)
#         print('seeds',seeds)
#         print('start iterations'+'*'*10)
#         max_spreads = []
#         for i in tqdm(seeds):
#             # for node in nods:
#             each_infl  = IC(g, i, prob, n_iters)
#             max_spreads.append(each_infl)
#         spread_dict[s]=sum(max_spreads)
#     # ultimate_seed_set, max_spreads = KempeGreedy(graph, num_seed_nodes, prob, n_iters)
#     # spread_dict['greedy']= sum(max_spreads)
#     return spread_dict


# def centrality(G):
#   print('Run Centrality'+ '*'*10)
#   print('betweenness'+ '*'*10)
#   nx.set_node_attributes(G, nx.betweenness_centrality(G), "betweenness")
#   print('out degree'+ '*'*10)
#   nx.set_node_attributes(G, nx.out_degree_centrality(G), "out_degree_centrality")
#   print('page rank'+ '*'*10)
#   nx.set_node_attributes(G, nx.pagerank(G), "pagerank")
# #   print('katz centrality'+ '*'*10)
# #   nx.set_edge_attributes(G, nx.katz_centrality(G), "katz_centrality")
#   print('closeness centrality'+ '*'*10)
#   nx.set_node_attributes(G, nx.closeness_centrality(G), "closeness_centrality")
#   print('eigenvector centrality'+ '*'*10)
#   nx.set_node_attributes(G, nx.eigenvector_centrality(G), "eigenvector_centrality")
#   print('Done Centrality'+ '*'*10)

# def run_greedy(g,k,prob,n_iters,dataset):

#     print("Starting Greedy Independent Cascade Model"+"*"*10)

#     greedy_solution, greedy_spreads = KempeGreedy(g, k, prob, n_iters)

#     print('Seed Set: ', greedy_solution)
#     print('Maximum_Influences: ', greedy_spreads)

#     # Get the current date and time
#     current_datetime = datetime.datetime.now()

#     # Format the date and time as a string
#     formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

#     # Define the filename with the date and time included
#     output_file = 'results/greedy_results_{}.txt'.format(formatted_datetime)

#     # Open the file in write mode
#     with open(output_file, 'w') as f:
#         # Write the seed set to the file
#         f.write('Seed Set: {}\n'.format(greedy_solution))
        
#         # Write the maximum influences to the file
#         f.write('Maximum Influences: {}\n'.format(greedy_spreads))
#         f.write('configuration:\n')
#         f.write('Number of Seed Nodes: {}\n'.format(k))
#         f.write('Probability: {}\n'.format(prob))
#         f.write('Number of Iterations: {}\n'.format(n_iters))
#         f.write('dataset: {}\n'.format(dataset))

#     # Print a message to confirm that the results have been saved
#     print('Results saved to', output_file)

# def run_IC_with_selected_seeds(g,k,a,prob,n_iters,dataset):
    
#         print("Starting Independent Cascade Model with selected seeds"+"*"*10)
    
#         solution, spreads = IC_with_selected_seeds(g, k, prob, n_iters)
    
#         print('Seed Set: ', solution)
#         print('Maximum_Influences: ', spreads)
    
#         # Get the current date and time
#         current_datetime = datetime.datetime.now()
    
#         # Format the date and time as a string
#         formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    
#         # Define the filename with the date and time included
#         output_file = 'results/IC_with_selected_seeds_results_{}.txt'.format(formatted_datetime)
    
#         # Open the file in write mode
#         with open(output_file, 'w') as f:
#             # Write the seed set to the file
#             f.write('Seed Set: {}\n'.format(solution))
            
#             # Write the maximum influences to the file
#             f.write('Maximum Influences: {}\n'.format(spreads))
#             f.write('configuration:\n')
#             f.write('Number of Seed Nodes: {}\n'.format(k))
#             f.write('Probability: {}\n'.format(prob))
#             f.write('Number of Iterations: {}\n'.format(n_iters))
#             f.write('seed selection by: {}\n'.format(a))
#             f.write('dataset: {}\n'.format(dataset))
    
#         # Print a message to confirm that the results have been saved
#         print('Results saved to', output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--k', type=int, default=10, help='Number of Seed Nodes')
    parser.add_argument('--prob', type=float, default=0.2, help='Probability')
    parser.add_argument('--n_iters', type=int, default=10, help='Number of Iterations')
    parser.add_argument('--dataset', type=str, default='../data/email-Eu-core.txt', help='Dataset')
    args = parser.parse_args()

    k = args.k
    prob = args.prob
    n_iters = args.n_iters
    dataset = args.dataset

    g = nx.read_edgelist(dataset, create_using=nx.DiGraph())
    print("number nodes: ", g.number_of_nodes())
    h = ig.Graph.from_networkx(g)
    # Run algorithms
    celf_output   = celf(h,k,p = prob,mc = n_iters)
    greedy_output = greedy(h,k,p = prob,mc = n_iters)
    # Print resulting seed sets
    print("celf output:   " + str(celf_output[0]))
    print("greedy output: " + str(greedy_output[0]))
        # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Format the date and time as a string
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    # Define the filename with the date and time included
    output_file = 'greedy_vs_celf{}.txt'.format(formatted_datetime)

    # Open the file in write mode
    with open(output_file, 'w') as f:
        # Write the seed set to the file
        f.write('Seed Set (greedy): {}\n'.format(greedy_output[0]))
        
        # Write the maximum influences to the file
        f.write('Maximum Influences (greedy): {}\n'.format(greedy_output[1]))
        f.write('Seed Set (celf): {}\n'.format(celf_output[0]))
        # Write the maximum influences to the file
        f.write('Maximum Influences (celf): {}\n'.format(celf_output[1]))
        f.write('configuration:\n')
        f.write('Number of Seed Nodes: {}\n'.format(k))
        f.write('Probability: {}\n'.format(prob))
        f.write('Number of Iterations: {}\n'.format(n_iters))
        f.write('dataset: {}\n'.format(dataset))

    # Print a message to confirm that the results have been saved
    print('Results saved to', output_file)

    # Plot settings
    plt.rcParams['figure.figsize'] = (9,6)
    plt.rcParams['lines.linewidth'] = 4
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['ytick.left'] = False

    # Plot Computation Time
    plt.plot(range(1,len(greedy_output[2])+1),greedy_output[2],label="Greedy",color="#FBB4AE")
    plt.plot(range(1,len(celf_output[2])+1),celf_output[2],label="CELF",color="#B3CDE3")
    plt.ylabel('Computation Time (Seconds)'); plt.xlabel('Size of Seed Set')
    plt.title('Computation Time'); plt.legend(loc=2)
    plt.savefig('greedy_vs_celf_computation_time.png')
    plt.show()

    # Plot Expected Spread by Seed Set Size
    plt.plot(range(1,len(greedy_output[1])+1),greedy_output[1],label="Greedy",color="#FBB4AE")
    plt.plot(range(1,len(celf_output[1])+1),celf_output[1],label="CELF",color="#B3CDE3")
    plt.xlabel('Size of Seed Set'); plt.ylabel('Expected Spread')
    plt.title('Expected Spread'); plt.legend(loc=2)
    plt.savefig('greedy_vs_celf_expected_spread.png')
    plt.show()






