import argparse
import os
import datasets
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
    model_name: str = "meta-llama/Llama-7b-hf",
    output_dir: str | None = "out",  # pass None to prevent saving to disk
    return_activations: bool = False,
) -> None | tuple[th.Tensor]:
    assert get_attentions or get_hidden_states, "At least one of get_attentions or get_hidden_states must be True"
    assert output_dir is not None or return_activations, "If output_dir is None, return_activations must be True"

    if output_dir is not None and get_attentions:
        attn_output_path = os.path.join(output_dir, model_name, dataset, f"{ATTN_PATH}.pt")
    else:
        attn_output_path = None
    
    if output_dir is not None and get_hidden_states:
        hidden_output_path = os.path.join(output_dir, model_name, dataset, f"{HIDDEN_PATH}.pt")
    else:
        hidden_output_path = None

    # Load dataset
    print("Loading dataset...")
    dataset = datasets.load_dataset(dataset)["train"]

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model
    print("Loading model...")
    model = AutoModel.from_pretrained(
        model_name, device_map="auto", output_attentions=get_attentions, output_hidden_states=get_hidden_states, return_dict=True
    )
    model.eval()

    # Find largest batch size that works
    print("Finding optimal batch size...")
    batch_size = len(dataset)
    with TemporaryDirectory() as tmp_dir, th.no_grad():
        while True:
            try:
                test_batch = dataset[:batch_size]
                test_batch = tokenizer(test_batch["text"], padding=True, truncation=True, return_tensors="pt")

                th.cuda.empty_cache()

                model(**test_batch)

                break
            except RuntimeError:
                batch_size //= 2
        del test_batch

        # Data loader
        print("Creating data loader...")
        data_loader = th.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=datasets.utils.collate_batch, num_workers=4
        )
        num_batches = len(data_loader)

        # Get activations in batches
        print("Getting activations...")
        attn_tmp_dir = None if attn_output_path is None else os.path.join(tmp_dir, ATTN_PATH)
        hidden_tmp_dir = None if hidden_output_path is None else os.path.join(tmp_dir, HIDDEN_PATH)
        os.makedirs(attn_tmp_dir, exist_ok=True)
        os.makedirs(hidden_tmp_dir, exist_ok=True)
        for i, batch in tqdm(enumerate(data_loader)):
            batch = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt", return_attention_mask=get_attentions)

            th.cuda.empty_cache()
            outputs = model(**batch)

            if get_attentions:
                causal_mask = th.ones(1, 1, batch["input_ids"].shape[1], batch["input_ids"].shape[1]).bool().cpu()  # (1, 1, T, T)
                causal_mask = th.tril(causal_mask)  # lower triangular mask
                causal_mask = causal_mask.expand(model.config.num_attention_heads, batch["input_ids"].shape[0], -1, -1)  # (H, B, T, T)

                padding_mask = batch.attention_mask.unsqueeze(0).unsqueeze(-1).bool().cpu()  # (1, B, T, 1)
                padding_mask = padding_mask.expand(model.config.num_attention_heads, -1, -1, padding_mask.shape[-2])  # (H, B, T, T)

                attention_mask = th.logical_and(causal_mask, padding_mask)  # (H, B, T, T)

                flattened_attentions = [attn.cpu()[attention_mask].view(model.config.num_attention_heads, -1) for attn in outputs.attentions]
                flattened_attentions = th.stack(flattened_attentions, dim=0)  # (L, H, B)

                th.save(flattened_attentions, os.path.join(attn_tmp_dir, f"{i}.pt"))
            if get_hidden_states:
                hidden_states = th.stack([hidden_state.cpu() for hidden_state in outputs.hidden_states])  # (L, B, T, D)

                th.save(hidden_states, os.path.join(hidden_tmp_dir, f"{i}.pt"))

        del data_loader, model, tokenizer, dataset, outputs, batch
        th.cuda.empty_cache()

        # Collate activations
        print("Collating activations...")
        if get_attentions:
            attn_activations = [th.load(os.path.join(attn_tmp_dir, f"{i}.pt")) for i in range(num_batches)]
            attn_activations = th.cat(attn_activations, dim=-1)  # (L, H, B)

            if attn_output_path is not None:
                th.save(attn_activations, attn_output_path)
        else:
            attn_activations = None

        if get_hidden_states:
            hidden_activations = [th.load(os.path.join(hidden_tmp_dir, f"{i}.pt")) for i in range(num_batches)]
            hidden_activations = th.cat(hidden_activations, dim=1)  # (L, B, T, D)

            if hidden_output_path is not None:
                th.save(hidden_activations, hidden_output_path)
        else:
            hidden_activations = None

        if return_activations:
            return tuple(activation for activation in (attn_activations, hidden_activations) if activation is not None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--get_attentions", action="store_true")
    parser.add_argument("--get_hidden_states", action="store_true")
    parser.add_argument("--dataset", type=str, default="ivanzhouyq/RedPajama-Tiny")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-7b-hf")
    parser.add_argument("--output_dir", type=str, default="out")

    args = parser.parse_args()

    get_activations(
        get_attentions=args.get_attentions,
        get_hidden_states=args.get_hidden_states,
        dataset=args.dataset,
        model_name=args.model_name,
        output_dir=args.output_dir,
    )
