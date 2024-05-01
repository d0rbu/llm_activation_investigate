import torch as th
from transformers import LlamaModel, LlamaTokenizer, LlamaForCausalLM, MistralForCausalLM
from transformers.models.llama.modeling_llama import logger
from transformers.models.mistral.modeling_mistral import _prepare_4d_causal_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask
from transformers.cache_utils import StaticCache, DynamicCache, Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from core.util import is_in_slices
from typing import Self, Sequence


def convert_to_skip_attn_llama(
    causal_model: LlamaForCausalLM,
    base_attn_layer: int = 2,
    predicted_attn_layers: Sequence[slice] = (slice(3, -1),),
    topk: int = 8,
    **kwargs,
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


def convert_to_skip_attn_mistral(
    causal_model: MistralForCausalLM,
    base_attn_layer: int = 2,
    predicted_attn_layers: Sequence[slice] = (slice(3, -1),),
    topk: int = 8,
    **kwargs,
) -> None:
    # https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/mistral/modeling_mistral.py#L932
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
    ) -> tuple | BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

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
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                attn_mask = top_tokens_mask if is_in_slices(i, predicted_attn_layers, len(self.layers)) else attention_mask

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=i == base_attn_layer,
                    use_cache=use_cache,
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
                top_tokens_mask[attention_mask < 0] = attention_mask[attention_mask < 0]  # (B, 1, T_O, T_I)

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
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

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
