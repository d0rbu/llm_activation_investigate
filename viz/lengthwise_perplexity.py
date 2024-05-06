import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence
from itertools import product
from tqdm import tqdm
from collections import defaultdict


NO_GROUPBY = {
    "all": [None]
}
MOVING_AVERAGE_WINDOW = 128
MAX_PERPLEXITY = 1e4


def get_lengthwise_perplexity(row: pd.Series) -> float:
    perplexities = row["perplexities"]

    # avg over batch
    max_len = max(len(p) for p in perplexities)
    avg_perplexities = []
    for i in range(max_len):
        avg_perplexity = 0
        num_perplexity = 0
        for p in perplexities:
            if i < len(p):
                avg_perplexity += p[i]
                num_perplexity += 1

        avg_perplexities.append(avg_perplexity / num_perplexity)
    
    avg_perplexities = np.array(avg_perplexities)

    # remove outliers
    avg_perplexities[avg_perplexities > MAX_PERPLEXITY] = MAX_PERPLEXITY

    # get moving average
    avg_perplexities = np.convolve(avg_perplexities, np.ones(MOVING_AVERAGE_WINDOW) / MOVING_AVERAGE_WINDOW, mode="same")

    return avg_perplexities

def graph_lengthwise_perplexity(
    output_dir: str = "out",
    figures_dir: str = "figures",
    groupby: Sequence[str] = [],
    truncate_first: int = 0,
    model_names: Sequence[str] = ["meta-llama/Llama-2-7b-hf"],
    show: bool = False,
) -> None:
    os.makedirs(figures_dir, exist_ok=True)

    # Load data
    print(f"Loading perplexity data...")
    with open(os.path.join(output_dir, f"skip_attn_perplexity.json"), "r") as f:
        data = json.load(f)
        data = [row for row in data if row["model"] in model_names]
        data = pd.DataFrame(data)

    # Group by
    groupby_params = {}
    for param in groupby:
        if param not in data:
            continue

        groupby_params[param] = data[param].unique()
    
    # If there is nothing to group by, just use the whole data
    if not groupby_params:
        groupby_params = NO_GROUPBY

    for groupby_vals in product(*[groupby_params[param] for param in groupby]):
        groupby_filter = True
        if groupby_params != NO_GROUPBY:
            for param, val in zip(groupby, groupby_vals):
                groupby_filter &= data[param] == val
        else:
            groupby_filter = slice(None)

        group_data = data[groupby_filter]

        # Get perplexities
        scores = group_data.apply(get_lengthwise_perplexity, axis=1)
        legend = group_data["topk"].apply(lambda x: f"Top {x}")

        param_strings = [f"{param}={val}" for param, val in zip(groupby, groupby_vals)]
        if groupby_params == NO_GROUPBY:
            title_params = ""
            filename_params = ""
        else:
            title_params = " " + ", ".join(groupby_vals)
            filename_params = "_" + "_".join(param_strings).replace("/", "-")

        # Plot topk vs scores, log scale
        fig, ax = plt.subplots()
        for score in scores:
            ax.plot(score[truncate_first:], linestyle="-")
        ax.legend(legend[truncate_first:])
        ax.set_yscale("log")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Perplexity")
        # ax.set_title(f"{special_attn_name}{title_params}")
        ax.set_title(f"{title_params}")
        plt.savefig(os.path.join(figures_dir, f"skip_attn_lengthwise_perplexity{filename_params}.png"))

        if show:
            plt.show()

        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--figures_dir", type=str, default="figures")
    parser.add_argument("--groupby", type=str, nargs="+", default=[])
    parser.add_argument("--truncate_first", type=int, default=0)
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--model_names", type=str, nargs="+", default=["meta-llama/Llama-2-7b-hf"])

    args = parser.parse_args()

    graph_lengthwise_perplexity(
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        groupby=args.groupby,
        truncate_first=args.truncate_first,
        model_names=args.model_names,
        show=args.show,
    )
