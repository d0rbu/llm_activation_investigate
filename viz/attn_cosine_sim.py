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
    base_layers: Sequence[int] | None = None,
) -> None:
    data_path = os.path.join(output_dir, model_name, dataset, f"{ATTN_PATH}.pt")
    assert os.path.exists(data_path), f"Data path {data_path} does not exist"

    figures_out_dir = os.path.join(figures_dir, model_name, dataset)
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
    else:
        max_token = attn.shape[3]
    
    if base_layers is None:
        base_layers = tuple(range(attn.shape[0]))

    # Heatmap data
    cosine_sim_heatmaps = th.zeros(attn.shape[0], attn.shape[0], max_token)  # (L, L, T)

    print("Creating heatmaps...")
    for base_layer, predicted_layer, token_idx in tqdm(product(base_layers, range(attn.shape[0]), range(max_token)), total=len(base_layers) * attn.shape[0] * max_token):
        # for each batch that is not a padding token, compute the cosine similarity of average attention across heads
        base_attn_vectors = attn[base_layer, :, :, token_idx, :token_idx + 1].mean(dim=1)  # (B, token_idx + 1)
        predicted_attn_vectors = attn[predicted_layer, :, :, token_idx, :token_idx + 1].mean(dim=1)  # (B, token_idx + 1)

        # mask padding samples
        mask = padding_mask[:, token_idx]  # (B,)

        # cosine similarity
        cosine_sim = th.nn.functional.cosine_similarity(base_attn_vectors, predicted_attn_vectors, dim=-1)  # (B,)
        cosine_sim = cosine_sim[mask]

        # average cosine similarity
        cosine_sim_heatmaps[base_layer, predicted_layer, token_idx] = cosine_sim.mean()

        # save if we are done with the heatmap
        if predicted_layer == attn.shape[0] - 1 and token_idx == max_token - 1:
            fig, ax = plt.subplots()
            cax = ax.matshow(cosine_sim_heatmaps[base_layer], cmap="RdBu_r")
            fig.colorbar(cax)
            plt.title(f"Cosine Similarity Heatmap (Layer {base_layer})")
            plt.xlabel("Token")
            plt.ylabel("Layer")
            plt.savefig(os.path.join(figures_out_dir, f"cosine_sim_heatmap_layer_{base_layer}.png"))
            print(f"Saved heatmap for layer {base_layer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ivanzhouyq/RedPajama-Tiny")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--figures_dir", type=str, default="figures")
    parser.add_argument("--truncate_data", type=int, default=0)
    parser.add_argument("--base_layers", type=int, nargs="+", default=None)

    args = parser.parse_args()

    cosine_sim_heatmap(
        dataset=args.dataset,
        model_name=args.model_name,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        truncate_data=args.truncate_data,
        base_layers=args.base_layers,
    )
