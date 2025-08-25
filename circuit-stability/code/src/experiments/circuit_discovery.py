import os
import sys
import pickle
import random
import argparse
import torch
import json
from functools import partial

from cdatasets import DatasetBuilder, PromptFormatter
from eap import Graph, attribute, evaluate_baseline, evaluate_graph
from utils import (
    seed_everything,
    parse_key_value_pairs,
    make_dataset,
    get_metric,
    get_extraction,
    extraction_schema,
)

import torch.nn.functional as F
from transformer_lens import HookedTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="model")
    parser.add_argument("ofname", type=str, help="output filename")
    parser.add_argument("--batch_size", type=int, help="batch size", default=32)
    parser.add_argument("--ndevices", type=int, help="number of devices", default=1)
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DatasetBuilder.ids.keys()),
        help="dataset name",
        required=True,
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=list(PromptFormatter.ids.keys()),
        help="format name",
        required=True,
    )
    parser.add_argument("--data_params", nargs="*", default=[], help="dataset params")
    parser.add_argument("--format_params", nargs="*", default=[], help="format params")
    parser.add_argument(
        "--patching_metric", type=str, default="kl", help="patching metric"
    )
    parser.add_argument(
        "--extraction",
        type=str,
        default="last_token",
        help="method for extracting comparison tokens",
    )
    parser.add_argument("--ig_steps", type=int, default=5, help="number of IG steps")
    parser.add_argument("--device", type=str, default="cuda", help="device to use")
    args = parser.parse_args()
    args.data_params = parse_key_value_pairs(args.data_params)
    args.format_params = parse_key_value_pairs(args.format_params)
    return args


def main():
    opts = parse_args()
    seed_everything(opts.seed)
    n_thoughts = 5
    
    # Print device info
    device = opts.device
    print(f"Using device: {device}")

    
    if device == "cuda" and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        if "3050" in torch.cuda.get_device_name(0) and "Ti" in torch.cuda.get_device_name(0):
            print("Detected lower VRAM GPU, enabling memory optimizations")

    dataset = make_dataset(
        opts.dataset, opts.data_params, opts.format, opts.format_params
    )

    model = HookedTransformer.from_pretrained(opts.model_name, device=device)

    for i in range (len(dataset)):
        dataloader = dataset.to_dataloader(model, opts.batch_size, indices=[i])
        
        for batch in dataloader:
            print(f"[DEBUG] Batch size: {len(batch)}")
            for idx, item in enumerate(batch):
                print(f"[DEBUG] Item {idx}: len={len(item)}, types={[type(x) for x in item]}")
                if len(item) >= 3 and hasattr(item[0], 'shape'):
                    print(f"[DEBUG] Input tokens shape: {item[0].shape}")
                    print(f"[DEBUG] First few tokens: {item[0][0][:10] if len(item[0].shape) > 1 else item[0][:10]}")
                if len(item) >= 2:
                    print(f"[DEBUG] Clean/corrupted comparison available: {type(item[0])}, {type(item[1])}")
        
        # Print batch size info
        print(f"Batch size: {opts.batch_size}")

        model.cfg.use_split_qkv_input = True
        model.cfg.use_attn_result = True
        model.cfg.use_hook_mlp_in = True

        pure_metric = get_metric(opts.patching_metric)
        extraction = get_extraction(opts.extraction)

        metric = extraction_schema(extraction, model)(pure_metric)

        g = Graph.from_model(model)
        print(f"[DEBUG] Initial graph has {len(g.nodes)} nodes and {len(g.edges)} edges")
        
        attribute(model, g, dataloader, metric, method="EAP-IG", ig_steps=opts.ig_steps)
        
        # Count edges in graph
        edges_in_graph = sum(1 for e in g.edges.values() if e.in_graph)
        print(f"[DEBUG] After attribution: {edges_in_graph} edges in graph")
        
        # Check edge scores
        scored_edges = [e for e in g.edges.values() if e.score is not None]
        if scored_edges:
            scores = [e.score for e in scored_edges]
            print(f"[DEBUG] Edge scores: min={min(scores):.4f}, max={max(scores):.4f}, mean={sum(scores)/len(scores):.4f}")
        else:
            print("[DEBUG] No edges have scores assigned")
        
        g.apply_topn(200, absolute=False)
        edges_in_graph_after_topn = sum(1 for e in g.edges.values() if e.in_graph)
        print(f"[DEBUG] After topn: {edges_in_graph_after_topn} edges in graph")
        
        g.to_json(f"{i//n_thoughts}_th_thought_{i}_th_variation.png.json")
        g.prune_dead_nodes()
        
        active_nodes = sum(1 for n in g.nodes.values() if n.in_graph)
        print(f"[DEBUG] After pruning: {active_nodes} active nodes")

        print(f"[DEBUG] Starting evaluation...")
        baseline = evaluate_baseline(model, dataloader, metric)
        print(f"[DEBUG] Baseline loss: {baseline.mean().item():.4f}")
        
        results = evaluate_graph(model, g, dataloader, metric)
        print(f"[DEBUG] Circuit results: {results.mean().item():.4f}")

        diff = (results - baseline).mean().item()
        print(f"[DEBUG] Circuit loss difference: {diff:.4f}")

        print(f"The circuit incurred extra {diff} loss.")
        
        # Save the circuit stability score to a file for ToT to read
        score_data = {
            "variation_index": i,
            "circuit_loss": diff,
            "baseline_loss": baseline.mean().item(),
            "circuit_results": results.mean().item()
        }
        
        with open(f"{i//n_thoughts}_th_thought_{i}_th_variation_score.json", "w") as f:
            json.dump(score_data, f)

        gz = g.to_graphviz()
        gz.draw(f"{i//n_thoughts}_th_thought_{i}_th_variation.png", prog="dot")



if __name__ == "__main__":
    main()
