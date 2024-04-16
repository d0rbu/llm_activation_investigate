import argparse
import json
import os
import math
import torch as th
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm
from typing import Sequence
from itertools import product
from core.skip_attn import convert_to_regular_attn_llama, convert_to_skip_attn_llama


FORWARD_FNS = {
    LlamaForCausalLM: (convert_to_regular_attn_llama, convert_to_skip_attn_llama)
}


def skip_attention_perplexity(
    model_names: Sequence[str] = ["meta-llama/Llama-2-7b-hf"],
    dataset_names: Sequence[str] = ["ivanzhouyq/RedPajama-Tiny"],
    output_dir: str = "out",
    batch_size: int | None = None,
    truncate_dataset: int = 0,
    max_length: int = 0,
) -> None:
    output_location = os.path.join(output_dir, "skip_attn_pplx.json")
    results = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_location):
        with open(output_location, "r") as f:
            results = json.load(f)

    current_model = None
    current_dataset = None
    model_batch_sizes = {
        model_name: None
        for model_name in model_names
    }
    for dataset_name, model_name in tqdm(product(dataset_names, model_names), total=len(dataset_names) * len(model_names), desc="Evaluating models on datasets"):
        # check if results already exist
        if any(row["model"] == model_name and row["dataset"] == dataset_name for row in results):
            continue

        if current_model != model_name:
            # load model
            current_model = model_name
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

            # change model forward function to use skip attention
            if type(model) in FORWARD_FNS:
                normal_forward_fn, skip_attn_fn = FORWARD_FNS[type(model)]
            else:
                raise ValueError(f"Model type {type(model)} not supported for skip attention")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
            model.eval()

            if max_length <= 0:
                max_length = model.config.max_position_embeddings

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
            "perplexities": [],
            "avg_perplexity": 0.0,
            "perplexities_skip_attn": [],
            "avg_perplexity_skip_attn": 0.0,
            "losses": [],
            "avg_loss": 0.0,
            "losses_skip_attn": [],
            "avg_loss_skip_attn": 0.0,
        }

        with th.no_grad():
            print("Computing vanilla perplexities...")
            normal_forward_fn(model)  # convert model to use normal attention
            for i, batch in tqdm(enumerate(data_loader), total=num_batches, desc="Computing perplexities"):
                batch = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(model.device)
                outputs = model(**batch)
                loss = outputs[0].mean(dim=-1).mean(dim=-1)  # (B,)
                perplexity = th.exp(loss)

                result["perplexities"].extend(perplexity.tolist())
                result["losses"].extend(loss.tolist())

            result["avg_perplexity"] = sum(result["perplexities"]) / len(result["perplexities"])
            result["avg_loss"] = sum(result["losses"]) / len(result["losses"])

            print("Computing skip attention perplexities...")
            skip_attn_fn(model)  # convert model to use skip attention

            for i, batch in tqdm(enumerate(data_loader), total=num_batches, desc="Computing skip attention perplexities"):
                batch = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt", max_length=max_length)
                outputs = model(**batch)
                loss = outputs[0].mean(dim=-1).mean(dim=-1)  # (B,)
                perplexity = th.exp(loss)

                result["perplexities_skip_attn"].extend(perplexity.tolist())
                result["loss_skip_attn"].extend(loss.tolist())

            result["avg_perplexity_skip_attn"] = sum(result["perplexities_skip_attn"]) / len(result["perplexities_skip_attn"])
            result["avg_loss_skip_attn"] = sum(result["losses_skip_attn"]) / len(result["losses_skip_attn"])

        results.append(result)
        with open(output_location, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", type=str, nargs="+", default=["meta-llama/Llama-2-7b-hf"])
    parser.add_argument("--datasets", type=str, nargs="+", default=["ivanzhouyq/RedPajama-Tiny"])
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--truncate_dataset", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=0)

    args = parser.parse_args()

    skip_attention_perplexity(
        model_names=args.model_names,
        dataset_names=args.datasets,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        truncate_dataset=args.truncate_dataset,
        max_length=args.max_length,
    )
