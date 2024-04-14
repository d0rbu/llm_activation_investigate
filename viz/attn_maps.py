import os
import argparse
import matplotlib.pyplot as plt
import torch as th
from typing import Sequence
from itertools import product
from tqdm import tqdm
from exp.get_activations import ATTN_PATH


def cosine_sim_heatmap(
    dataset: str = "ivanzhouyq/RedPajama-Tiny",
    model_name: str = "meta-llama/Llama-2-7b-hf",
    output_dir: str = "out",
    figures_dir: str = "figures",
    truncate_data: int = 0,
) -> None:
    data_path = os.path.join(output_dir, model_name, dataset, f"{ATTN_PATH}.pt")
    assert os.path.exists(data_path), f"Data path {data_path} does not exist"

    figures_out_dir = os.path.join(figures_dir, model_name, dataset, "attn_maps")
    os.makedirs(figures_out_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    data = th.load(data_path)
    attn, padding_mask = data["activation"], data["padding"]  # (L, B, H, T, T), (B, T)
    del data
    
    if truncate_data > 0:
        max_token = min(truncate_data, attn.shape[3])
        attn = attn[..., :max_token, :max_token]
        padding_mask = padding_mask[:, :max_token]

    print("Creating attention maps...")
    for layer, sample_idx in tqdm(product(range(attn.shape[0]), range(attn.shape[1])), total=attn.shape[0] * attn.shape[1]):
        # average attention map for a given sample
        avg_attn_map = attn[layer, sample_idx].mean(dim=0)  # (T, T)

        fig, ax = plt.subplots()
        cax = ax.matshow(avg_attn_map)
        fig.colorbar(cax)
        plt.title(f"Attention Map (Layer {layer}, Sample {sample_idx})")
        plt.xlabel("Key Token")
        plt.ylabel("Query Token")
        plt.savefig(os.path.join(figures_out_dir, f"attn_layer={layer}_sample={sample_idx}.png"))
        print(f"Saved attn map for layer {layer} sample {sample_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ivanzhouyq/RedPajama-Tiny")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--figures_dir", type=str, default="figures")
    parser.add_argument("--truncate_data", type=int, default=0)

    args = parser.parse_args()

    cosine_sim_heatmap(
        dataset=args.dataset,
        model_name=args.model_name,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        truncate_data=args.truncate_data,
    )
