import torch as th
from transformers import LlamaModel, LlamaTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import logger
from transformers.cache_utils import StaticCache, DynamicCache, Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import Self, Sequence


def is_in_slices(
    index: int,
    slices: Sequence[slice],
    total_length: int,
) -> bool:
    for s in slices:
        start = 0
        if isinstance(s.start, int):
            if 0 <= s.start < total_length:
                start = s.start
            elif -total_length <= s.start < 0:
                start = total_length + s.start
            else:
                raise ValueError(f"Invalid start index {s.start} for total length {total_length}")

        stop = total_length
        if isinstance(s.stop, int):
            if 0 <= s.stop < total_length:
                stop = s.stop
            elif -total_length <= s.stop < 0:
                stop = total_length + s.stop
            else:
                raise ValueError(f"Invalid stop index {s.stop} for total length {total_length}")
        
        step = s.step if s.step is not None else 1

        if start <= index < stop and (index - start) % step == 0:
            return True

    return False
