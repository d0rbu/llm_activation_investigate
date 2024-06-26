import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence
from itertools import product
from tqdm import tqdm
from collections import defaultdict


SPECIAL_ATTN_TYPES = ["skip_attn", "reuse_attn"]
MEASURES = ["Perplexity", "Average LM Harness Accuracy", "Average LM Harness Perplexity", "LM Harness Accuracies"]
NO_GROUPBY = {
    "all": [None]
}


# takes row of a dataframe and extracts the perplexity
def get_score_perplexity(row: pd.Series) -> float:
    return row["avg_perplexity"]


def get_score_lm_harness_accuracy(row: pd.Series) -> float:
    task_results = row["task_results"]
    sum_acc = 0
    n_acc = 0
    for task, results in task_results.items():
        if "acc,none" in results:
            sum_acc += results["acc,none"]
            n_acc += 1

    return sum_acc / n_acc if n_acc > 0 else 0.0


def get_score_lm_harness_accuracies(group_data: pd.DataFrame) -> tuple[list[str], list[list[float]]]:
    sorted_data = group_data.sort_values("topk")
    legend = []
    scores = {}
    for row in tqdm(sorted_data.itertuples(), total=len(sorted_data)):
        task_results = row.task_results
        for task, results in task_results.items():
            if "acc,none" in results:
                if task not in scores:
                    scores[task] = []
                scores[task].append(results["acc,none"])
                if task not in legend:
                    legend.append(task)
            elif "exact_match,get-answer" in results:
                if task not in scores:
                    scores[task] = []
                scores[task].append(results["exact_match,get-answer"])
                if task not in legend:
                    legend.append(task)
    
    scores = [scores[task] for task in legend]

    return legend, scores


def get_score_lm_harness_perplexity(row: pd.Series) -> float:
    task_results = row["task_results"]
    sum_pplx = 0
    n_pplx = 0
    for task, results in task_results.items():
        if "word_perplexity,none" in results:
            sum_pplx += results["word_perplexity,none"]
            n_pplx += 1

    return sum_pplx / n_pplx if n_pplx > 0 else 0.0


def graph_special_attns(
    output_dir: str = "out",
    figures_dir: str = "figures",
    groupby: Sequence[str] = [],
    truncate_first: int = 0,
    show: bool = False,
) -> None:
    for file_suffix, measure_name in zip(["dataset_pplx", "harness_eval", "harness_eval", "harness_eval"], MEASURES):
        # search for all files with the given suffix
        data_paths = []
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith(f"{file_suffix}.json"):
                    data_paths.append(os.path.join(root, file))
        
        for special_attn_type, special_attn_name in zip(SPECIAL_ATTN_TYPES, ["Skip Attention"]):
            os.makedirs(figures_dir, exist_ok=True)

            # Load data
            print(f"Loading {special_attn_name} {measure_name} data...")
            with open(os.path.join(output_dir, f"{special_attn_type}_{file_suffix}.json"), "r") as f:
                data = json.load(f)
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
                legend = None

                # Get scores
                if measure_name == MEASURES[0]:
                    scores = group_data.apply(get_score_perplexity, axis=1)
                elif measure_name == MEASURES[1]:
                    scores = group_data.apply(get_score_lm_harness_accuracy, axis=1)
                elif measure_name == MEASURES[2]:
                    scores = group_data.apply(get_score_lm_harness_perplexity, axis=1)
                elif measure_name == MEASURES[3]:
                    legend, scores = get_score_lm_harness_accuracies(group_data)

                param_strings = [f"{param}={val}" for param, val in zip(groupby, groupby_vals)]
                if groupby_params == NO_GROUPBY:
                    title_params = ""
                    filename_params = ""
                else:
                    title_params = " " + ", ".join(groupby_vals)
                    filename_params = "_" + "_".join(param_strings).replace("/", "-")

                # Plot topk vs scores, log scale
                fig, ax = plt.subplots()
                if legend is None:
                    ax.plot(group_data["topk"][truncate_first:], scores[truncate_first:], marker="o", linestyle="-")
                else:
                    for score in scores:
                        ax.plot(group_data["topk"][truncate_first:], score[truncate_first:], marker="o", linestyle="-")
                    ax.legend(legend)
                ax.set_xscale("log")
                ax.set_xlabel("Top K")
                ax.set_ylabel(measure_name)
                # ax.set_title(f"{special_attn_name}{title_params}")
                ax.set_title(f"{title_params}")
                plt.savefig(os.path.join(figures_dir, f"{special_attn_type} {measure_name} {filename_params}.png"))

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

    args = parser.parse_args()

    graph_special_attns(
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        groupby=args.groupby,
        truncate_first=args.truncate_first,
        show=args.show,
    )
