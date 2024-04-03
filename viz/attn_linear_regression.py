import os
import argparse
import matplotlib.pyplot as plt
import torch as th
from itertools import product
from tqdm import tqdm
from exp.get_activations import ATTN_PATH


def linear_regression_heatmap(
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
    flattened_attns = th.load(data_path)  # (L, H, B)
    if truncate_data > 0:
        flattened_attns = flattened_attns[..., :truncate_data]

    # Heatmap data
    mse_heatmap = th.zeros(flattened_attns.shape[0] - 1, flattened_attns.shape[0])
    r2_heatmap = th.zeros(flattened_attns.shape[0] - 1, flattened_attns.shape[0])

    print("Creating heatmaps...")
    for base_layer, predicted_layer in tqdm(product(range(flattened_attns.shape[0]), repeat=2), total=flattened_attns.shape[0] ** 2):
        if predicted_layer <= base_layer:
            continue

        # Multiple linear regression
        X = flattened_attns[base_layer].T
        y = flattened_attns[predicted_layer].T

        # Compute coefficients
        X = th.cat([X, th.ones_like(X)], dim=-1)
        beta = th.linalg.lstsq(X, y).solution

        # Compute predictions
        y_pred = X @ beta

        # Compute residuals
        residuals = y - y_pred

        # Compute R^2
        R2 = 1 - residuals.var() / y.var()

        # Compute MSE
        MSE = residuals.pow(2).mean()

        mse_heatmap[base_layer, predicted_layer] = MSE
        r2_heatmap[base_layer, predicted_layer] = R2

    # Plot each heatmap separately and save
    fig, ax = plt.subplots()
    cax = ax.matshow(mse_heatmap, cmap="RdBu_r")
    fig.colorbar(cax)
    plt.title("MSE Heatmap")
    plt.xlabel("Predicted Layer")
    plt.ylabel("Base Layer")
    plt.savefig(os.path.join(figures_out_dir, "mse_heatmap.png"))

    fig, ax = plt.subplots()
    cax = ax.matshow(r2_heatmap, cmap="RdBu", vmin=0, vmax=1)
    fig.colorbar(cax)
    plt.title("R^2 Heatmap")
    plt.xlabel("Predicted Layer")
    plt.ylabel("Base Layer")
    plt.savefig(os.path.join(figures_out_dir, "r2_heatmap.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ivanzhouyq/RedPajama-Tiny")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--figures_dir", type=str, default="figures")
    parser.add_argument("--truncate_data", type=int, default=0)

    args = parser.parse_args()

    linear_regression_heatmap(
        dataset=args.dataset,
        model_name=args.model_name,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        truncate_data=args.truncate_data,
    )
