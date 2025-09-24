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

