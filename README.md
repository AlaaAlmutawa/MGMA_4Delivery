# Community Detection and Influence Maximazation 

## Description

This repository focuses on processing real-world graphs, specifically the web-Google dataset from Stanford. It includes implementations of various algorithms for community detection and influence maximization.

## Datasets

We use different datasets (which can be found in the data folder)

The datasets are taken from SNAP.

| Dataset | Description | Number of Nodes | Number of Edges |
| --- | --- | --- | --- |
| `web-Google.txt` | Web graph from Google | 875,713 | 5,105,039 |
| `email-Eu-core.txt` | Email data from a large European research institution. Outgoing and incoming emails between the departments. | 1,005 | 25,571 |

You can replace the dataset names and descriptions with the actual details of your datasets. 

## Algorithms


### Community Detection

1. **Heuristic - Louvain Algorithm:** The Louvain algorithm is implemented for heuristic-based community detection, offering an efficient approach to uncovering community structures in the graph.

2. **ML-based - Spectral Clustering:** Spectral clustering is utilized for community detection using machine learning techniques, providing an alternative perspective to uncovering graph communities.


### Influence Maximization

1. **Heuristic - Independent Cascades Model**
2. **ML-based - Seed selection using PageRank**


## Usage

### Utilities 

Dataset reduction 

```bash
python utils/reduce_graph.py --dataset data/web-Google.txt --reduction 0.5 --output data/web-Google-reduced.txt
```
### Independent Cascades Model

To run indepedent cascade model for inlfuence maximzation, you can run the below command. 

| Argument | Description |
| --- | --- |
| `-h, --help` | Show this help message and exit |
| `--k ` | Number of Seed Nodes |
| `--prob ` | Probability |
| `--n_iters` | Number of Iterations |
| `--dataset` | Dataset |

```bash
python kempe-indepedent-cascades-model/Sample-Kempe-IM.py --dataset data/web-Google-reduced.txt
```