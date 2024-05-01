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
from exp.attn_pplx import FORWARD_FNS


def serve(
    model_name: 
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--topk", type=int, default=32)

    args = parser.parse_args()
    
    serve(
        model_name=args.model_name,
        topk=args.topk,
    )
