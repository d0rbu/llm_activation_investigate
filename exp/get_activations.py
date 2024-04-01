import argparse
import os
import datasets
import torch as th
from tempfile import TemporaryDirectory


ACTIVATIONS_FILENAME = "activations.pt"

def get_activations(
    dataset: str = "ivanzhouyq/RedPajama-Tiny",
    model_name: str = "meta-llama/Llama-7b-hf",
    output_dir: str | None = "out",  # pass None to prevent saving to disk
    return_activations: bool = False,
) -> None | tuple[th.Tensor]:
    output_path = None if output_dir is None else os.path.join(output_dir, model_name, dataset, ACTIVATIONS_FILENAME)


if __name__ == "__main__":
    get_activations()
