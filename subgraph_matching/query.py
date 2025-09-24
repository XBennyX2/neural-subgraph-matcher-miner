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


def main():
    # Build parser with defaults from repo
    parser = argparse.ArgumentParser(description="Query subgraph in target graph")
    utils.parse_optimizer(parser)
    parse_encoder(parser)

    parser.add_argument("--query_path", type=str, default=None,
                        help="Pickle file of query graph")
    parser.add_argument("--target_path", type=str, default=None,
                        help="Pickle file of target graph")
    # parser.add_argument("--model_path", type=str, default="ckpt/model.pt",
    #                     help="Path to pretrained model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for subgraph match")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the match result")
    parser.add_argument("--out", type=str, default="query_match.png",
                        help="Output path for visualization PNG")
    # parser.add_argument("--method_type", type=str, default="order",
    #                     help="Model type (order or mlp)")
    args = parser.parse_args()

    args.test = True  # important for build_model()

    # Load query and target graphs
    query = load_graph(args.query_path)
    target = load_graph(args.target_path, default_size=16)

    # Build model and load checkpoint
    model = build_model(args)

    # Run query match
    exists, best_score, mat = query_match(
        model, query, target,
        threshold=args.threshold,
        method_type=args.method_type
    )

    print("Neural model prediction:", exists)
    print("Best score:", best_score)
    exact = exact_subgraph_check(query, target)
    print("Exact isomorphism check:", exact)


    print("Match exists?:", exists)
    print("Best score:", best_score)

    if args.visualize:
        visualize(target, mat, out=args.out)


if __name__ == "__main__":
    main()
