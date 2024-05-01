import argparse
import json
import os
import math
import torch as th
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, MistralForCausalLM
from tqdm import tqdm
from typing import Sequence
from itertools import product
from core.skip_attn import convert_to_skip_attn_llama, convert_to_skip_attn_mistral
from core.reuse_attn import convert_to_reuse_attn_llama, convert_to_reuse_attn_mistral
from core.vanilla import convert_to_vanilla_attn_llama, convert_to_vanilla_attn_mistral


FORWARD_FNS = {
    LlamaForCausalLM: (convert_to_vanilla_attn_llama, convert_to_skip_attn_llama, convert_to_reuse_attn_llama),
    MistralForCausalLM: (convert_to_vanilla_attn_mistral, convert_to_skip_attn_mistral, convert_to_reuse_attn_mistral),
}


def harness_eval(
    model_names: Sequence[str] = ["meta-llama/Llama-2-7b-hf"],
    tasks: Sequence[str] = ["winogrande", "wikitext", "hellaswag"],
    output_dir: str = "out",
    batch_size: int | None = None,
    topk_values: Sequence[int] = [2 ** i for i in range(13)],
) -> None:
    skip_attn_output_path = os.path.join(output_dir, "skip_attn_harness_eval.json")
    reuse_attn_output_path = os.path.join(output_dir, "reuse_attn_harness_eval.json")
    _topk_values = topk_values

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    from lm_eval import evaluator
    from lm_eval.tasks import initialize_tasks
    from lm_eval.models.huggingface import HFLM

    initialize_tasks(verbosity = "INFO")

    for special_fn_idx, output_location in enumerate([skip_attn_output_path, reuse_attn_output_path]):
        if os.path.exists(output_location):
            with open(output_location, "r") as f:
                results = json.load(f)
        else:
            results = []

        if special_fn_idx == 0:
            topk_values = _topk_values
        elif special_fn_idx == 1:
            topk_values = [0]

        current_model = None
        for topk, model_name in tqdm(product(topk_values, model_names), total=len(topk_values) * len(model_names), desc="Evaluating models on lm harness eval tasks"):
            current_tasks = set([task for task in tasks])

            # check if results already exist
            found_task_results = {}
            for i, row in enumerate(results):
                if row["model"] == model_name and row["topk"] == topk:
                    found_task_results = row["task_results"]
                    found_result_tasks = set(found_task_results.keys())
                    current_tasks -= found_result_tasks

                    del results[i]
                    break

            if len(current_tasks) == 0:
                continue

            if current_model != model_name:
                # load model
                current_model = model_name
                model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", device_map="auto")

                # change model forward function to use special attention
                if type(model) in FORWARD_FNS:
                    convert_fn = FORWARD_FNS[type(model)][special_fn_idx + 1]
                else:
                    raise ValueError(f"Model type {type(model)} not supported for special attention functions")

                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model.resize_token_embeddings(len(tokenizer))
                model.eval()

            convert_fn(model, topk=topk)
            lm = HFLM(pretrained=model, tokenizer=tokenizer)

            result = {
                "model": model_name,
                "topk": topk,
                "task_results": found_task_results,
            }

            with th.no_grad():
                raw_results = evaluator.simple_evaluate(
                    model=lm,
                    tasks=list(current_tasks),
                    batch_size="auto",
                )
                result["task_results"].update(raw_results["results"])

            results.append(result)
            with open(output_location, "w") as f:
                json.dump(results, f, indent=4)


def perplexity(
    model_names: Sequence[str] = ["meta-llama/Llama-2-7b-hf"],
    dataset_names: Sequence[str] = ["ivanzhouyq/RedPajama-Tiny"],
    output_dir: str = "out",
    batch_size: int | None = None,
    truncate_dataset: int = 0,
    max_length: int = 0,
    topk_values: Sequence[int] = [2 ** i for i in range(13)],
) -> None:
    skip_attn_output_path = os.path.join(output_dir, "skip_attn_perplexity.json")
    reuse_attn_output_path = os.path.join(output_dir, "reuse_attn_perplexity.json")
    _topk_values = topk_values

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for special_fn_idx, output_location in enumerate([skip_attn_output_path, reuse_attn_output_path]):
        if os.path.exists(output_location):
            with open(output_location, "r") as f:
                results = json.load(f)
        else:
            results = []

        current_model = None
        current_dataset = None
        model_batch_sizes = {
            model_name: None
            for model_name in model_names
        }

        if special_fn_idx == 0:
            topk_values = _topk_values
        elif special_fn_idx == 1:
            topk_values = [0]

        for topk, dataset_name, model_name in tqdm(product(topk_values, dataset_names, model_names), total=len(topk_values) * len(dataset_names) * len(model_names), desc="Evaluating models on datasets"):
            # check if results already exist
            if any(row["model"] == model_name and ("dataset" not in row or row["dataset"] == dataset_name) and row["topk"] == topk for row in results):
                continue

            if current_model != model_name:
                # load model
                current_model = model_name
                model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", device_map="auto")

                # change model forward function to use skip attention
                if type(model) in FORWARD_FNS:
                    convert_fn = FORWARD_FNS[type(model)][special_fn_idx + 1]
                else:
                    raise ValueError(f"Model type {type(model)} not supported for special attention functions")

                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model.resize_token_embeddings(len(tokenizer))
                model.eval()

                if max_length <= 0:
                    max_length = model.config.max_position_embeddings

            convert_fn(model, topk=topk)

            if current_dataset != dataset_name:
                # load dataset
                current_dataset = dataset_name
                dataset = datasets.load_dataset(dataset_name)["train"]

            if model_batch_sizes[model_name] is None:
                # find largest batch size that works
                print("Finding optimal batch size...")

                if batch_size is None:
                    model_batch_size = 64
                    while model_batch_size > 0:
                        try:
                            print(f"Attempting batch size {model_batch_size}...")

                            test_batch = dataset[:model_batch_size]
                            test_batch = tokenizer(test_batch["text"], padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(model.device)

                            th.cuda.empty_cache()

                            model(**test_batch)

                            break
                        except RuntimeError as e:
                            model_batch_size //= 2
                            print(e)
                    else:
                        raise ValueError("Failed at batch size 0")

                    del test_batch
                else:
                    model_batch_size = batch_size

                print(f"Settled on batch size {model_batch_size}")
                model_batch_sizes[model_name] = model_batch_size
            else:
                model_batch_size = model_batch_sizes[model_name]

            # data loader
            print("Creating data loader...")
            data_loader = th.utils.data.DataLoader(
                dataset, batch_size=batch_size, num_workers=8,
            )
            if truncate_dataset <= 0:
                num_batches = len(data_loader)
            else:
                num_batches = math.ceil(truncate_dataset / batch_size)

            result = {
                "model": model_name,
                "dataset": dataset_name,
                "topk": topk,
                "perplexities": [],
                "avg_perplexity": 0.0,
                "losses": [],
                "avg_loss": 0.0,
            }

            with th.no_grad():
                print("Computing perplexities...")

                for i, batch in tqdm(enumerate(data_loader), total=num_batches, desc=f"Topk={topk}", leave=False):
                    batch = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(model.device)
                    outputs = model(**batch, labels=batch["input_ids"])
                    loss = outputs[0]  # (,)
                    perplexity = th.exp(loss)

                    if (loss.isnan().sum() > 0):
                        import pdb; pdb.set_trace()

                    result["perplexities"].append(perplexity.item())
                    result["losses"].append(loss.item())

                    del outputs, batch, loss, perplexity
                    th.cuda.empty_cache()

                result["avg_perplexity"] = sum(result["perplexities"]) / len(result["perplexities"])
                result["avg_loss"] = sum(result["losses"]) / len(result["losses"])

            results.append(result)
            with open(output_location, "w") as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", type=str, nargs="+", default=["meta-llama/Llama-2-7b-hf"])
    parser.add_argument("--datasets", type=str, nargs="+", default=["ivanzhouyq/RedPajama-Tiny"])
    parser.add_argument("--tasks", type=str, nargs="+", default=["winogrande", "wikitext", "hellaswag", "gpqa", "gsm8k", "humaneval", "piqa", "sciq"])
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--truncate_dataset", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=0)
    parser.add_argument("--topk", type=int, nargs="+", default=[2 ** i for i in range(13)])  # 1, 2, 4, ..., 4096

    args = parser.parse_args()

    if len(args.datasets) > 0 and args.datasets[0] != "null":
        print("Starting dataset perplexity measurements...")
        perplexity(
            model_names=args.model_names,
            dataset_names=args.datasets,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            truncate_dataset=args.truncate_dataset,
            max_length=args.max_length,
            topk_values=args.topk,
        )
        th.cuda.empty_cache()

    if len(args.tasks) > 0 and args.tasks[0] != "null":
        print("Starting lm harness eval measurements...")
        harness_eval(
            model_names=args.model_names,
            tasks=args.tasks,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            topk_values=args.topk,
        )
