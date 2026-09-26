"""Layer-local PACT reduction with proportional attention for Qwen decoders."""

from __future__ import annotations

import functools

import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from baselines.pact.reference import reduce_at_layer


class PactManipulator:
    def __init__(self, language_model, visual_positions, *, layer=4, prune_keep_fraction=0.55,
                 cutoff=0.21, target_visual_tokens=None):
        self.language_model = language_model
        self.visual_positions = list(visual_positions)
        self.layer = layer
        self.prune_keep_fraction = prune_keep_fraction
        self.cutoff = cutoff
        self.target_visual_tokens = target_visual_tokens
        self._original_forward = None
        self.log_token_sizes = None
        self.kept_visual_tokens = None
        self.chosen_cutoff = None

    def patch(self):
        if self._original_forward is not None:
            raise RuntimeError("PACT manipulator is already installed")
        self._original_forward = self.language_model.forward
        self.language_model.forward = functools.update_wrapper(
            functools.partial(self._forward, self.language_model), self._original_forward)

    def unpatch(self):
        if self._original_forward is not None:
            self.language_model.forward = self._original_forward
            self._original_forward = None

    @staticmethod
    def _weighted_mask(base, sizes, query_len, key_len, device, dtype):
        if base is None:
            base = torch.zeros((1, 1, query_len, key_len), device=device, dtype=dtype)
            if query_len == key_len and query_len > 1:
                future = torch.triu(torch.ones((query_len, key_len), device=device, dtype=torch.bool), diagonal=1)
                base.masked_fill_(future, torch.finfo(dtype).min)
        else:
            base = base.clone()
        if base.ndim != 4:
            raise RuntimeError("PACT proportional attention requires a 4D additive attention mask")
        if len(sizes) > key_len:
            raise RuntimeError("PACT cluster sizes exceed attention key length")
        padded = torch.zeros(key_len, device=device, dtype=dtype)
        padded[:len(sizes)] = sizes.to(device=device, dtype=dtype)
        return base + padded.view(1, 1, 1, -1)

    @staticmethod
    def _layer_cache_length(cache, layer_index):
        if cache is None:
            return 0
        layers = getattr(cache, "layers", ())
        if layer_index >= len(layers):
            return 0
        layer = layers[layer_index]
        keys = getattr(layer, "keys", None)
        return int(keys.shape[-2]) if keys is not None else 0

    def _before_layer(self, index, hidden, decoder_layer, position_embeddings):
        return None

    def _reduce(self, hidden, decoder_layer, position_embeddings):
        return reduce_at_layer(
            hidden, self.visual_positions, decoder_layer,
            position_embeddings=position_embeddings,
            prune_keep_fraction=self.prune_keep_fraction, cutoff=self.cutoff,
            target_visual_tokens=self.target_visual_tokens)

    def _forward(self, text_model, *args, **kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        attention_mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
        position_ids = kwargs.get("position_ids", args[2] if len(args) > 2 else None)
        past_key_values = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
        inputs_embeds = kwargs.get("inputs_embeds", args[4] if len(args) > 4 else None)
        use_cache = kwargs.get("use_cache", args[5] if len(args) > 5 else None)
        extra = {k: v for k, v in kwargs.items()
                 if k not in {"input_ids", "attention_mask", "position_ids",
                              "past_key_values", "inputs_embeds", "use_cache"}}
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=text_model.config)
        if inputs_embeds is None:
            inputs_embeds = text_model.embed_tokens(input_ids)
        if position_ids is None:
            seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            positions = torch.arange(seen, seen + inputs_embeds.shape[1], device=inputs_embeds.device)
            position_ids = positions.reshape(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

        if isinstance(attention_mask, dict):
            masks = attention_mask
        else:
            mask_kwargs = dict(config=text_model.config, inputs_embeds=inputs_embeds,
                               attention_mask=attention_mask, past_key_values=past_key_values,
                               position_ids=text_position_ids)
            masks = {"full_attention": create_causal_mask(**mask_kwargs)}
            if getattr(text_model, "has_sliding_layers", False):
                masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden = inputs_embeds
        embeddings = text_model.rotary_emb(hidden, position_ids)
        is_prefill = hidden.shape[1] > 1 and bool(self.visual_positions)
        for index, decoder_layer in enumerate(text_model.layers):
            if is_prefill:
                self._before_layer(index, hidden, decoder_layer, embeddings)
            if is_prefill and index == self.layer:
                reduction = self._reduce(hidden, decoder_layer, embeddings)
                hidden = reduction.hidden_states
                keep = torch.as_tensor(reduction.kept_positions, dtype=torch.long, device=hidden.device)
                position_ids = position_ids[..., keep.to(position_ids.device)]
                cos, sin = embeddings
                axis = -2
                embeddings = (cos.index_select(axis, keep.to(cos.device)),
                              sin.index_select(axis, keep.to(sin.device)))
                if text_position_ids is not None:
                    text_position_ids = text_position_ids[:, keep.to(text_position_ids.device)]
                new_masks = {}
                for name, mask in masks.items():
                    if mask is None:
                        new_masks[name] = None
                    else:
                        idx = keep.to(mask.device)
                        new_masks[name] = mask.index_select(-2, idx).index_select(-1, idx)
                masks = new_masks
                self.log_token_sizes = reduction.log_token_sizes
                self.kept_visual_tokens = sum(pos in set(self.visual_positions)
                                              for pos in reduction.kept_positions)
                self.chosen_cutoff = reduction.cutoff

            layer_types = getattr(text_model.config, "layer_types", None)
            mask = masks[layer_types[index] if layer_types is not None else "full_attention"]
            if self.log_token_sizes is not None and index >= self.layer:
                key_len = self._layer_cache_length(past_key_values, index) + hidden.shape[1]
                if not is_prefill:
                    mask = None
                mask = self._weighted_mask(mask, self.log_token_sizes, hidden.shape[1],
                                           key_len, hidden.device, hidden.dtype)
            hidden = decoder_layer(
                hidden, attention_mask=mask, position_embeddings=embeddings,
                position_ids=text_position_ids, past_key_values=past_key_values,
                use_cache=use_cache, **extra)
        hidden = text_model.norm(hidden)
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=past_key_values)
