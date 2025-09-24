import argparse
import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from common import utils
from subgraph_matching.config import parse_encoder
from subgraph_matching.train import build_model
from subgraph_matching.alignment import query_match


def load_graph(path, default_size=8, p=0.25):
    """Load graph from pickle or generate a random one."""
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return nx.gnp_random_graph(default_size, p)


def visualize(target, mat, out="query_match.png"):
    """Highlight matched nodes in the target graph."""
    pos = nx.spring_layout(target, seed=42)
    nx.draw(target, pos, with_labels=True,
            node_color="lightblue", edge_color="gray")
    # highlight nodes with highest scores per query node
    mapping = np.argmax(mat, axis=1)
    nx.draw_networkx_nodes(target, pos,
                           nodelist=mapping.tolist(),
                           node_color="orange")
    plt.title("Target graph: matched nodes in orange")
    plt.savefig(out)
    print(f"Saved visualization to {out}")
from networkx.algorithms import isomorphism

def exact_subgraph_check(query, target):
    """Exact check using NetworkX VF2 isomorphism."""
    GM = isomorphism.GraphMatcher(target, query)
    return GM.subgraph_is_isomorphic()


