#note: script was modified according to our requirements. 
#Reference: https://github.com/AdnanRasad/Influence-Maximization-Analysis/blob/master/Kempe-Independent-Cascades-Model/Sample-Kempe-IM Adnan Rasad

import networkx as nx
import numpy as np
from tqdm import tqdm
import datetime
import argparse


def IC(Networkx_Graph,Seed_Set,Probability,Num_of_Simulations):
    spread = []
    # print('start IC'+'*'*10)
    # for i in tqdm(range(Num_of_Simulations)):
    for i in range(Num_of_Simulations):
        
        new_active, Ans = Seed_Set[:], Seed_Set[:]
        while new_active:
            #Getting neighbour nodes of newly activate node
            targets = Neighbour_finder(Networkx_Graph,Probability,new_active)
    
            #Calculating if any nodes of those neighbours can be activated, if yes add them to new_ones.
            np.random.seed(i)
            success = np.random.uniform(0,1,len(targets)) < Probability
            new_ones = list(np.extract(success, sorted(targets)))
            
            #Checking which ones in new_ones are not in our Ans...only adding them to our Ans so that no duplicate in Ans.
            new_active = list(set(new_ones) - set(Ans))
            Ans = list(Ans)
            Ans += new_active
            
        spread.append(len(Ans))
        
    return(np.mean(spread))
    
    
    
def Neighbour_finder(g,p,new_active):
    
    targets = []
    for node in new_active:
        if g.has_node(node):
            targets += g.neighbors(node)

    return(targets)


def KempeGreedy(graph, num_seed_nodes, prob=0.2, n_iters=100):
   #Solution gives 2 parameters: the selected seed set for which we found the maximum influence & their influences
   #Here we used this method for Networkx Directed Graph
    max_spreads = []
    ultimate_seed_set = []
    print('Number of Seed Nodes: ', num_seed_nodes)
    print('start iterations'+'*'*10)
    for _ in range(num_seed_nodes):
        best_node = -1
        best_spread = -np.inf

       
        nods = graph.nodes - ultimate_seed_set;

        # for node in nods:
        for node in tqdm(nods):
            each_infl  = IC(g, ultimate_seed_set + [node], prob, n_iters)
            if each_infl  > best_spread:
                best_spread = each_infl 
                best_node = node

        ultimate_seed_set.append(best_node)
        max_spreads.append(best_spread)
    return ultimate_seed_set, max_spreads

def seedsGeneration(G,k,a):
    d=nx.get_node_attributes(G,a)
    dict_sorted=dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
    return list(dict_sorted.keys())[:k]

def IC_with_selected_seeds(graph,num_seed_nodes,a,prob=0.2, n_iters=100):
    #get seeds 
    seeds=seedsGeneration(graph,num_seed_nodes,a)
    max_spreads = []
    print('Number of Seed Nodes: ', num_seed_nodes)
    print('seeds',seeds)
    print('start iterations'+'*'*10)
    for i in tqdm(seeds):
        # for node in nods:
        each_infl  = IC(g, i, prob, n_iters)
        max_spreads.append(each_infl)
    return seeds, max_spreads


def centrality(G,a):
  print('Run Centrality'+ '*'*10)
  if a == 'betweenness':
    print('betweenness'+ '*'*10)
    nx.set_node_attributes(G, nx.betweenness_centrality(G), "betweenness")
  elif a == 'out_degree_centrality':
    print('out degree'+ '*'*10)
    nx.set_node_attributes(G, nx.out_degree_centrality(G), "out_degree_centrality")
  elif a == 'pagerank':
    print('page rank'+ '*'*10)
    nx.set_node_attributes(G, nx.pagerank(G), "pagerank")
  print('Done Centrality'+ '*'*10)

def run_greedy(g,k,prob,n_iters,dataset):

    print("Starting Greedy Independent Cascade Model"+"*"*10)

    greedy_solution, greedy_spreads = KempeGreedy(g, k, prob, n_iters)

    print('Seed Set: ', greedy_solution)
    print('Maximum_Influences: ', greedy_spreads)

    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Format the date and time as a string
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    # Define the filename with the date and time included
    output_file = 'results/greedy_results_{}.txt'.format(formatted_datetime)

    # Open the file in write mode
    with open(output_file, 'w') as f:
        # Write the seed set to the file
        f.write('Seed Set: {}\n'.format(greedy_solution))
        
        # Write the maximum influences to the file
        f.write('Maximum Influences: {}\n'.format(greedy_spreads))
        f.write('configuration:\n')
        f.write('Number of Seed Nodes: {}\n'.format(k))
        f.write('Probability: {}\n'.format(prob))
        f.write('Number of Iterations: {}\n'.format(n_iters))
        f.write('dataset: {}\n'.format(dataset))

    # Print a message to confirm that the results have been saved
    print('Results saved to', output_file)

def run_IC_with_selected_seeds(g,k,a,prob,n_iters,dataset):
    
        print("Starting Independent Cascade Model with selected seeds"+"*"*10)
    
        solution, spreads = IC_with_selected_seeds(g, k, a, prob, n_iters)
    
        print('Seed Set: ', solution)
        print('Maximum_Influences: ', spreads)
    
        # Get the current date and time
        current_datetime = datetime.datetime.now()
    
        # Format the date and time as a string
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    
        # Define the filename with the date and time included
        output_file = 'results/IC_with_selected_seeds_results_{}.txt'.format(formatted_datetime)
    
        # Open the file in write mode
        with open(output_file, 'w') as f:
            # Write the seed set to the file
            f.write('Seed Set: {}\n'.format(solution))
            
            # Write the maximum influences to the file
            f.write('Maximum Influences: {}\n'.format(spreads))
            f.write('configuration:\n')
            f.write('Number of Seed Nodes: {}\n'.format(k))
            f.write('Probability: {}\n'.format(prob))
            f.write('Number of Iterations: {}\n'.format(n_iters))
            f.write('seed selection by: {}\n'.format(a))
            f.write('dataset: {}\n'.format(dataset))
    
        # Print a message to confirm that the results have been saved
        print('Results saved to', output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--k', type=int, default=2, help='Number of Seed Nodes')
    parser.add_argument('--prob', type=float, default=0.2, help='Probability')
    parser.add_argument('--n_iters', type=int, default=10, help='Number of Iterations')
    parser.add_argument('--dataset', type=str, default='data/web-Google.txt', help='Dataset')
    args = parser.parse_args()

    k = args.k
    prob = args.prob
    n_iters = args.n_iters
    dataset = args.dataset

    # g = nx.read_edgelist('data/sampleGraph2.txt', create_using=nx.DiGraph());
    g = nx.read_edgelist(dataset, create_using=nx.DiGraph())

    greedy = input('Run greedy IC? (y/n): ')
    print("number nodes: ", g.number_of_nodes())
    print("number edges: ", g.number_of_edges())
    if greedy == 'y':
        run_greedy(g,k,prob,n_iters,dataset)
    else:
        a = input('Choose centrality measure for seed selection (betweenness, out_degree_centrality,pagerank): ')
        centrality(g,a)
        run_IC_with_selected_seeds(g,k,a,prob,n_iters,dataset)


