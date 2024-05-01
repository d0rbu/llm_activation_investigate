from transformers import LlamaModel, LlamaForCausalLM, MistralModel, MistralForCausalLM


def convert_to_vanilla_attn_llama(
    causal_model: LlamaForCausalLM,
) -> None:
    bound_forward = LlamaModel.forward.__get__(causal_model.model, causal_model.model.__class__)
    setattr(causal_model.model, "forward", bound_forward)


def convert_to_vanilla_attn_mistral(
    causal_model: MistralForCausalLM,
) -> None:
    bound_forward = MistralModel.forward.__get__(causal_model.model, causal_model.model.__class__)
    setattr(causal_model.model, "forward", bound_forward)
