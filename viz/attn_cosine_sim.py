import os
import argparse
import matplotlib.pyplot as plt
import torch as th
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

    figures_out_dir = os.path.join(figures_dir, model_name, dataset)
    os.makedirs(figures_out_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    data = th.load(data_path)
    attn, padding = data["activation"], data["padding"]  # (L, B, H, T, T), (B, T)
    del data

    # Heatmap data
    cosine_sim_heatmap = th.zeros(attn.shape[0], attn.shape[0])

    print("Creating heatmaps...")
    for base_layer, predicted_layer in tqdm(product(range(attn.shape[0]), repeat=2), total=attn.shape[0] ** 2):
        pass

    # Plot the heatmap separately and save
    fig, ax = plt.subplots()
    cax = ax.matshow(cosine_sim_heatmap, cmap="RdBu_r")
    fig.colorbar(cax)
    plt.title("Cosine Similarity Heatmap")
    plt.xlabel("Layer")
    plt.ylabel("Layer")
    plt.savefig(os.path.join(figures_out_dir, "mse_heatmap.png"))


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
