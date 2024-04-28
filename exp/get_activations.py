import argparse
import os
import datasets
import math
import torch as th
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from tempfile import TemporaryDirectory


ATTN_PATH = "attentions"
HIDDEN_PATH = "hidden_states"

def get_activations(
    get_attentions: bool = False,
    get_hidden_states: bool = False,
    dataset: str = "ivanzhouyq/RedPajama-Tiny",
    model_name: str = "meta-llama/Llama-2-7b-hf",
    output_dir: str | None = "out",  # pass None to prevent saving to disk
    return_activations: bool = False,
    batch_size: int | None = None,
    truncate_dataset: int = 0,
    max_length: int = 0,
) -> None | tuple[th.Tensor]:
    assert get_attentions or get_hidden_states, "At least one of get_attentions or get_hidden_states must be True"
    assert output_dir is not None or return_activations, "If output_dir is None, return_activations must be True"

    if output_dir is not None and get_attentions:
        attn_output_path = os.path.join(output_dir, model_name, dataset, f"{ATTN_PATH}.pt")
        os.makedirs(os.path.dirname(attn_output_path), exist_ok=True)
    else:
        attn_output_path = None
    
    if output_dir is not None and get_hidden_states:
        hidden_output_path = os.path.join(output_dir, model_name, dataset, f"{HIDDEN_PATH}.pt")
        os.makedirs(os.path.dirname(hidden_output_path), exist_ok=True)
    else:
        hidden_output_path = None

    # Load dataset
    print("Loading dataset...")
    dataset = datasets.load_dataset(dataset)["train"]

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Load model
    print("Loading model...")
    model = AutoModel.from_pretrained(
        model_name, device_map="balanced", output_attentions=get_attentions, output_hidden_states=get_hidden_states, return_dict=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    
    if max_length <= 0:
        max_length = model.config.max_position_embeddings

    with TemporaryDirectory() as tmp_dir, th.no_grad():
        # Find largest batch size that works
        print("Finding optimal batch size...")
        if batch_size is None:
            batch_size = 128
        while batch_size > 0:
            try:
                print(f"Attempting batch size {batch_size}...")

                test_batch = dataset[:batch_size]
                test_batch = tokenizer(test_batch["text"], padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(model.device)

                th.cuda.empty_cache()

                model(**test_batch)

                break
            except RuntimeError as e:
                batch_size //= 2
                print(e)
        else:
            raise ValueError("Failed at batch size 0")

        del test_batch
        print(f"Settled on batch size {batch_size}")

        # Data loader
        print("Creating data loader...")
        data_loader = th.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=8,
        )
        if truncate_dataset <= 0:
            num_batches = len(data_loader)
        else:
            num_batches = math.ceil(truncate_dataset / batch_size)

        # Get activations in batches
        print("Getting activations...")
        if attn_output_path is None:
            attn_tmp_dir = None
        else:
            attn_tmp_dir = os.path.join(tmp_dir, ATTN_PATH)
            os.makedirs(attn_tmp_dir, exist_ok=True)

        if hidden_output_path is None:
            hidden_tmp_dir = None
        else:
            hidden_tmp_dir = os.path.join(tmp_dir, HIDDEN_PATH)
            os.makedirs(hidden_tmp_dir, exist_ok=True)

        samples = 0
        for i, batch in tqdm(zip(range(num_batches), data_loader), total=min(len(data_loader), num_batches)):
            batch = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt", max_length=max_length, return_attention_mask=get_attentions).to(model.device)
            samples += batch["input_ids"].shape[0]

            th.cuda.empty_cache()
            outputs = model(**batch)

            if get_attentions:
                attentions = [attn.cpu() for attn in outputs.attentions]
                attentions = th.stack(attentions, dim=0)  # (L, B, H, T, T)

                th.save(attentions, os.path.join(attn_tmp_dir, f"{i}.pt"))

                del attentions, outputs.attentions

                padding_mask = batch.attention_mask.bool().cpu()  # (B, T)

                th.save(padding_mask, os.path.join(attn_tmp_dir, f"{i}_padding.pt"))

                causal_mask = th.ones(1, batch["input_ids"].shape[1], batch["input_ids"].shape[1]).bool().cpu()  # (1, T, T)
                causal_mask = th.tril(causal_mask)  # lower triangular mask
                causal_mask = causal_mask.expand(batch["input_ids"].shape[0], -1, -1)  # (B, T, T)

                padding_mask = padding_mask.unsqueeze(-1)  # (B, T, 1)
                padding_mask = padding_mask.expand(-1, -1, padding_mask.shape[-2])  # (B, T, T)

                attention_mask = th.logical_and(causal_mask, padding_mask)  # (B, T, T)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(0)  # (1, B, 1, T, T)

                th.save(attention_mask, os.path.join(attn_tmp_dir, f"{i}_mask.pt"))

                del causal_mask, padding_mask, attention_mask

            if get_hidden_states:
                padding_mask = batch.attention_mask.unsqueeze(0).unsqueeze(-1).bool().cpu()  # (1, B, T, 1)

                th.save(padding_mask, os.path.join(hidden_tmp_dir, f"{i}_mask.pt"))

                hidden_states = th.stack([hidden_state.cpu() for hidden_state in outputs.hidden_states])  # (L, B, T, D)

                th.save(hidden_states, os.path.join(hidden_tmp_dir, f"{i}.pt"))

                del hidden_states, padding_mask
            
            del outputs

        del data_loader, model, tokenizer, dataset, batch
        th.cuda.empty_cache()

        # Collate activations
        print("Collating activations...")
        if get_attentions:
            attn_activations = [th.load(os.path.join(attn_tmp_dir, f"{i}.pt")) for i in range(num_batches)]
            attn_activations = th.cat(attn_activations, dim=1)  # (L, B, H, T, T)

            attn_masks = [th.load(os.path.join(attn_tmp_dir, f"{i}_mask.pt")) for i in range(num_batches)]
            attn_masks = th.cat(attn_masks, dim=1)  # (1, B, 1, T, T)

            attn_padding = [th.load(os.path.join(attn_tmp_dir, f"{i}_padding.pt")) for i in range(num_batches)]
            attn_padding = th.cat(attn_padding, dim=0)  # (B, T)

            if attn_output_path is not None:
                th.save({
                    "activation": attn_activations,
                    "mask": attn_masks,
                    "padding": attn_padding,
                }, attn_output_path)
        else:
            attn_activations = None

        if get_hidden_states:
            hidden_activations = [th.load(os.path.join(hidden_tmp_dir, f"{i}.pt")) for i in range(num_batches)]
            hidden_activations = th.cat(hidden_activations, dim=1)  # (L, B, T, D)

            hidden_masks = [th.load(os.path.join(hidden_tmp_dir, f"{i}_mask.pt")) for i in range(num_batches)]
            hidden_masks = th.cat(hidden_masks, dim=1)  # (1, B, T, 1)

            if hidden_output_path is not None:
                th.save({
                    "activation": hidden_activations,
                    "mask": hidden_masks,
                }, hidden_output_path)
        else:
            hidden_activations = None

        if return_activations:
            return tuple(activation for activation in (attn_activations, hidden_activations) if activation is not None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--get_attentions", action="store_true")
    parser.add_argument("--get_hidden_states", action="store_true")
    parser.add_argument("--dataset", type=str, default="ivanzhouyq/RedPajama-Tiny")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--truncate_dataset", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=0)

    args = parser.parse_args()

    get_activations(
        get_attentions=args.get_attentions,
        get_hidden_states=args.get_hidden_states,
        dataset=args.dataset,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        truncate_dataset=args.truncate_dataset,
        max_length=args.max_length,
    )
