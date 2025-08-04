import os
import sys
import json
import pickle
import random
import argparse
import torch
from functools import partial
from pathlib import Path
from ..cdatasets import DatasetBuilder, PromptFormatter
from ..eap import Graph, attribute, evaluate_baseline, evaluate_graph
from .utils import (
    seed_everything,
    parse_key_value_pairs,
    make_dataset,
    get_metric,
    get_extraction,
    extraction_schema,
)
import torch.nn.functional as F
from transformer_lens import HookedTransformer

# imports same from circuit_discovery.py


#small alteration in parse_args from circuit_discovery.py, I just added the output_dir argument, that way we can have a folder for each experiment
#we can remove this if we decide it's not needed, but if we do, just make sure to remove it from everywhere it's used
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="model")
    parser.add_argument("thought_variation_path", type=str, help="path to thought variation file")
    parser.add_argument("output_dir", type=str, help="output directory for results")
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
    parser.add_argument("--top_n_nodes", type=int, default=200, help="number of top nodes to keep")
    args = parser.parse_args()
    args.data_params = parse_key_value_pairs(args.data_params)
    args.format_params = parse_key_value_pairs(args.format_params)
    return args



## new function to load the thought variations from JSON format
def load_thought_variations(file_path):
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data,list):
                return data
            else:
                return [data]
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def create_dataset(thought_variation, dataset_class, data_params, format_class, format_params):

    dataset = dataset_class(**data_params)
    formatter = format_class(**format_params)
    
    dataset.data = [thought_variation]
    dataset.formatter = formatter
    return dataset

def process_single_variation(variation, variation_idx, opts, model, dataset_class, format_class, output_dir):    
    dataset = create_dataset(variation, dataset_class, opts.data_params, format_class, opts.format_params)
    dataloader = dataset.to_dataloader(model, batch_size=1)
    #using previous function and creating dataloader w/ batch_size 1, as mentioned in the github issue 


    #most everything below this line is from circuit_discovery.py's main function
    pure_metric = get_metric(opts.patching_metric)
    extraction = get_extraction(opts.extraction)
    metric = extraction_schema(extraction, model)(pure_metric)
    
    g = Graph.from_model(model)
    attribute(model, g, dataloader, metric, method="EAP-IG", ig_steps=opts.ig_steps)
    g.apply_topn(opts.top_n_nodes, absolute=False)
    
    base_name = Path(opts.thought_variation_path).stem
    json_filename = output_dir / f"{base_name}_variation_{variation_idx}_circuit.json"
    png_filename = output_dir / f"{base_name}_variation_{variation_idx}_circuit.png"
    
    g.to_json(str(json_filename))
    g.prune_dead_nodes()
    
    baseline = evaluate_baseline(model, dataloader, metric)
    results = evaluate_graph(model, g, dataloader, metric)
    diff = (results - baseline).mean().item()
    print(f"The circuit incurred extra {diff} loss.")
    
    gz = g.to_graphviz()
    gz.draw(str(png_filename), prog="dot")
    
    #can change return or make it something completely different if needed
    return (json_filename, png_filename)


# way slimmer than the circuit.discovery.py main, just because most of the code was moved to the process_single_variation function
def main():
    opts = parse_args()
    seed_everything(opts.seed)
    
    #added for output folder
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    

    device = opts.device
    print(f"Using device: {device}")
    if device == "cuda" and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    thought_variations = load_thought_variations(opts.thought_variation_path)    

    dataset_class = DatasetBuilder.ids[opts.dataset]
    format_class = PromptFormatter.ids[opts.format]


    model = HookedTransformer.from_pretrained(opts.model_name, device=device)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    all_results = []
    for i, variation in enumerate(thought_variations):
        try:
            result = process_single_variation(variation, i, opts, model, dataset_class, format_class, output_dir)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing variation {i}: {e}")
            continue
    
    return all_results

if __name__ == "__main__":
    main()