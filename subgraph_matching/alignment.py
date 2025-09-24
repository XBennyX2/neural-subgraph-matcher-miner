"""
Build an alignment matrix for matching a query subgraph in a target graph.
Subgraph matching model needs to have been trained with the node-anchored option
(default).
"""

import argparse
import os
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch

from common import utils
from subgraph_matching.config import parse_encoder
from subgraph_matching.train import build_model


def gen_alignment_matrix(model, query, target, method_type="order"):
    """
    Generate subgraph matching alignment matrix for a given query and
    target graph. Each entry (u, v) of the matrix contains the confidence score
    the model gives for the query graph, anchored at u, being a subgraph of the
    target graph, anchored at v.

    Args:
        model: trained subgraph matching model (must be node-anchored)
        query: networkx query graph
        target: networkx target graph
        method_type: "order" or "mlp"
    """
    query_nodes = sorted(query.nodes())
    target_nodes = sorted(target.nodes())

    mat = np.zeros((len(query_nodes), len(target_nodes)))

    for i, u in enumerate(query_nodes):
        for j, v in enumerate(target_nodes):
            batch = utils.batch_nx_graphs([query, target], anchors=[u, v])
            embs = model.emb_model(batch)
            pred = model(embs[1].unsqueeze(0), embs[0].unsqueeze(0))
            raw_pred = model.predict(pred)

            # Handle different model types properly
            if method_type == "order":
                score = raw_pred.item()  # assume already log-likelihood/margin
            elif method_type == "mlp":
                # assume softmax output, take probability of "positive"
                score = raw_pred[0][1].item()
            else:
                raise ValueError(f"Unknown method_type {method_type}")

            mat[i][j] = score
    return mat


def query_match(model, query, target, threshold=0.5, method_type="order"):
    """
    Returns True if the query graph exists as a subgraph of the target graph,
    according to the trained model. Otherwise False.

    Args:
        model: trained subgraph matching model
        query: networkx query graph
        target: networkx target graph
        threshold: decision threshold
        method_type: "order" or "mlp"
    """
    mat = gen_alignment_matrix(model, query, target, method_type=method_type)
    best_score = float(mat.max())

    # Thresholds depend on scoring scale
    if method_type == "order":
        exists = best_score >= 0.0   # order embeddings: 0 ≈ neutral boundary
    elif method_type == "mlp":
        exists = best_score >= threshold
    else:
        exists = False

    return exists, best_score, mat


def main():
    if not os.path.exists("plots/"):
        os.makedirs("plots/")
    if not os.path.exists("results/"):
        os.makedirs("results/")

    parser = argparse.ArgumentParser(description="Alignment arguments")
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    parser.add_argument("--query_path", type=str, help="path of query graph", default="")
    parser.add_argument("--target_path", type=str, help="path of target graph", default="")
    parser.add_argument("--model_path", type=str, help="path to model checkpoint", default="ckpt/model.pt")
    parser.add_argument("--method_type", type=str, choices=["order", "mlp"], default="order")
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()
    args.test = True

    # Load graphs
    if args.query_path and os.path.exists(args.query_path):
        with open(args.query_path, "rb") as f:
            query = pickle.load(f)
    else:
        query = nx.gnp_random_graph(8, 0.25)

    if args.target_path and os.path.exists(args.target_path):
        with open(args.target_path, "rb") as f:
            target = pickle.load(f)
    else:
        target = nx.gnp_random_graph(16, 0.25)

    # Build model + load checkpoint
    model = build_model(args)
    if args.model_path and os.path.exists(args.model_path):
        state = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"Loaded model weights from {args.model_path}")
    else:
        print("⚠️ Warning: No model checkpoint found, using randomly initialized model!")

    # Run alignment
    exists, best_score, mat = query_match(model, query, target,
                                          threshold=args.threshold,
                                          method_type=args.method_type)

    print(f"Neural model prediction: {exists}")
    print(f"Best score: {best_score:.4f}")

    # Save results
    np.save("results/alignment.npy", mat)
    print("Saved alignment matrix in results/alignment.npy")

    plt.clf()
    plt.imshow(mat, interpolation="nearest")
    plt.colorbar(label="Match score")
    plt.title("Alignment Matrix")
    plt.savefig("plots/alignment.png")
    plt.close()
    print("Saved alignment matrix plot in plots/alignment.png")


if __name__ == "__main__":
    main()
