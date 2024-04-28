from transformers import LlamaModel, LlamaForCausalLM


def convert_to_vanilla_attn_llama(
    causal_model: LlamaForCausalLM,
) -> None:
    bound_forward = LlamaModel.forward.__get__(causal_model.model, causal_model.model.__class__)
    setattr(causal_model.model, "forward", bound_forward)
