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


def convert_to_skip_attn_llama(
    causal_model: LlamaForCausalLM,
    base_attn_layer: int = 2,
    predicted_attn_layers: Sequence[slice] = (slice(3, -1),),
    topk: int = 8,
) -> None:
    # https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/llama/modeling_llama.py#L940
    def skip_attn_forward(
        self: Self,
        input_ids: th.LongTensor = None,
        attention_mask: th.Tensor | None = None,
        position_ids: th.LongTensor | None = None,
        past_key_values: list[th.FloatTensor] | None = None,
        inputs_embeds: th.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: th.LongTensor | None = None,
    ) -> tuple | BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = th.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        top_tokens_mask = None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                attn_mask = top_tokens_mask if is_in_slices(i, predicted_attn_layers, len(self.layers)) else causal_mask  # (B, 1, T, T)

                if is_in_slices(i, predicted_attn_layers, len(self.layers)):  # if we are doing skip attention for this layer
                    pass

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=i == base_attn_layer,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            if i == base_attn_layer:
                token_attn = layer_outputs[1]  # (B, H, T_O, T_I)
                token_attn = token_attn.sum(dim=1)  # (B, T_O, T_I)  how much each token is paid attention to across heads

                k = min(topk, token_attn.shape[-1])
                top_tokens = token_attn.topk(k, dim=-1).indices  # (B, T_O, topk)

                dtype, device = hidden_states.dtype, hidden_states.device
                min_dtype = th.finfo(dtype).min
                top_tokens_mask = th.full_like(token_attn, min_dtype, dtype=dtype, device=device)  # (B, T_O, T_I)
                top_tokens_mask.scatter_(-1, top_tokens, 0)  # (B, T_O, T_I), top tokens are 0, rest are -inf

                # logical and with causal mask
                top_tokens_mask = top_tokens_mask.unsqueeze(1)  # (B, 1, T_O, T_I)
                top_tokens_mask[causal_mask < 0] = causal_mask[causal_mask < 0]  # (B, 1, T_O, T_I)

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    bound_forward = skip_attn_forward.__get__(causal_model.model, causal_model.model.__class__)
    setattr(causal_model.model, "forward", bound_forward)


def convert_to_regular_attn_llama(
    causal_model: LlamaForCausalLM,
) -> None:
    bound_forward = LlamaModel.forward.__get__(causal_model.model, causal_model.model.__class__)
    setattr(causal_model.model, "forward", bound_forward)
