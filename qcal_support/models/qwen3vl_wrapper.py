"""Qwen2-VL / Qwen2.5-VL model wrapper for BEA experiments."""

import re
import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def _load_model_cls(model_path: str):
    """Return the correct ForConditionalGeneration class for the given path."""
    path_lower = str(model_path).lower()
    if "qwen3" in path_lower:
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    if "qwen2.5" in path_lower or "qwen2_5" in path_lower:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    return Qwen2VLForConditionalGeneration


class Qwen2VLWrapper:
    """Wrapper for Qwen2-VL / Qwen2.5-VL model."""

    def __init__(self, model_path="Qwen/Qwen2-VL-7B-Instruct", device="cuda"):
        self.device = device
        print(f"Loading model from {model_path}...")
        model_cls = _load_model_cls(model_path)
        self.model = model_cls.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=os.environ.get("QWEN_ATTN_IMPLEMENTATION", "eager"),
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.eval()
        print("Model loaded.")

    @torch.no_grad()
    def generate(self, image, prompt, max_new_tokens=256):
        """Generate response for image + prompt."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, return_tensors="pt"
        ).to(self.model.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        # Decode only the generated tokens
        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate_grounding(self, image, expression):
        """Generate grounding bbox for a referring expression."""
        prompt = (
            f"Please locate the object described by: \"{expression}\". "
            "Output the bounding box coordinates as [x1, y1, x2, y2] "
            "where values are normalized to 0-1000."
        )
        return self.generate(image, prompt, max_new_tokens=128)

    @torch.no_grad()
    def generate_qa(self, image, question):
        """Generate answer for a visual question."""
        return self.generate(image, question, max_new_tokens=64)

    @torch.no_grad()
    def generate_multi(self, images, prompt, max_new_tokens=256,
                       max_pixels_per_image=None):
        """Generate response for multiple images + prompt."""
        content = []
        for img in images:
            entry = {"type": "image", "image": img}
            if max_pixels_per_image is not None:
                entry["max_pixels"] = max_pixels_per_image
            content.append(entry)
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, return_tensors="pt"
        ).to(self.model.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    @staticmethod
    def parse_bbox(text, img_width=1, img_height=1):
        """Parse bounding box from model output text.

        Tries multiple formats:
        - [x1, y1, x2, y2] with values 0-1000
        - (x1, y1, x2, y2)
        - bare numbers
        Returns [x1, y1, x2, y2] normalized to image dimensions, or None.
        """
        patterns = [
            # Qwen2-VL format: (x1,y1),(x2,y2)
            r'\((\d+),(\d+)\),\((\d+),(\d+)\)',
            r'\[(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\]',
            r'\((\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\)',
            r'(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)',
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                vals = [int(m.group(i)) for i in range(1, 5)]
                # Qwen2-VL outputs coords in 0-1000 range
                return [
                    vals[0] / 1000 * img_width,
                    vals[1] / 1000 * img_height,
                    vals[2] / 1000 * img_width,
                    vals[3] / 1000 * img_height,
                ]
        return None

    # ---- Token-level manipulation API ----

    @torch.no_grad()
    def prepare_inputs(self, image, prompt):
        """Prepare model inputs without running generate. Returns inputs dict."""
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        return self.processor(
            text=[text], images=image_inputs, return_tensors="pt"
        ).to(self.model.device)

    @torch.no_grad()
    def prepare_inputs_multi(self, images, prompt, max_pixels_per_image=None):
        """Prepare model inputs for multiple images.

        Args:
            images: list of PIL images
            prompt: text prompt
            max_pixels_per_image: if set, each image is capped at this many
                pixels before tokenisation. Use to prevent sequence-length
                overflow when many images are present (e.g. 6144 = 96×64).
        """
        content = []
        for img in images:
            entry = {"type": "image", "image": img}
            if max_pixels_per_image is not None:
                entry["max_pixels"] = max_pixels_per_image
            content.append(entry)
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        return self.processor(
            text=[text], images=image_inputs, return_tensors="pt"
        ).to(self.model.device)

    @torch.no_grad()
    def extract_visual_embeddings(self, inputs):
        """Extract visual embeddings from prepared inputs.

        Returns:
            visual_embeds: tensor [total_visual_tokens, hidden_dim]
            image_grid_thw: tensor [num_images, 3]
        """
        vis_out = self.model.model.visual(
            inputs["pixel_values"],
            grid_thw=inputs["image_grid_thw"]
        )
        return vis_out.pooler_output, inputs["image_grid_thw"]

    @torch.no_grad()
    def generate_from_cache(self, inputs, past_key_values, max_new_tokens=256):
        """Decode-only generation reusing KV cache from a prior forward pass.

        The past_key_values must come from a forward pass on the same inputs
        (same input_ids / pixel_values, same sequence length).  This skips
        the expensive prefill step and goes straight to autoregressive decode.

        Args:
            inputs: original processor outputs (input_ids, attention_mask, …)
            past_key_values: KV cache returned by FastVScorer.score() when
                return_past_key_values=True
            max_new_tokens: generation length

        Returns:
            Generated text string
        """
        input_ids = inputs["input_ids"]          # [1, seq_len]
        attention_mask = inputs["attention_mask"] # [1, seq_len]

        # HuggingFace generate() with past_key_values expects input_ids to
        # contain only the tokens NOT yet in the cache.  The scoring forward
        # covered the full sequence [0, seq_len), so we pass the last token
        # as the first decode step input.
        last_token_ids = input_ids[:, -1:]  # [1, 1]

        gen_kwargs = {
            "input_ids": last_token_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
        }
        output_ids = self.model.generate(**gen_kwargs)
        return self.processor.decode(output_ids[0], skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate_with_modified_embeds(self, inputs, modified_visual_embeds,
                                      max_new_tokens=256):
        """Generate using modified visual embeddings.

        Args:
            inputs: original processor outputs (for input_ids, attention_mask, etc.)
            modified_visual_embeds: [total_visual_tokens, hidden_dim] modified embeddings
            max_new_tokens: generation length

        Returns:
            Generated text string
        """
        # Build inputs_embeds: text embeddings + modified visual embeddings
        input_ids = inputs["input_ids"]
        text_embeds = self.model.model.language_model.get_input_embeddings()(input_ids)

        # Get placeholder mask and scatter modified visual embeds
        image_mask, _ = self.model.model.get_placeholder_mask(
            input_ids, inputs_embeds=text_embeds,
            image_features=modified_visual_embeds
        )
        inputs_embeds = text_embeds.masked_scatter(image_mask, modified_visual_embeds)

        # Build generation kwargs
        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": inputs.get("attention_mask"),
            "mm_token_type_ids": inputs.get("mm_token_type_ids"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "video_grid_thw": inputs.get("video_grid_thw"),
            "max_new_tokens": max_new_tokens,
        }
        # Remove None values
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        output_ids = self.model.generate(**gen_kwargs)
        # With inputs_embeds, output_ids contains ONLY generated tokens
        return self.processor.decode(output_ids[0], skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate_with_token_manipulation_v2(self, manip_result, max_new_tokens=256):
        """Generate using V2 token manipulation (actual token removal).

        Args:
            manip_result: dict from TokenManipulatorV2.apply()
            max_new_tokens: generation length

        Returns:
            Generated text string
        """
        new_ids = manip_result["new_input_ids"]
        new_vis = manip_result["new_visual_embeds"]
        position_ids = manip_result["position_ids"]

        # Build inputs_embeds
        text_embeds = self.model.model.language_model.get_input_embeddings()(new_ids)

        if new_vis.shape[0] > 0:
            mask, _ = self.model.model.get_placeholder_mask(
                new_ids, inputs_embeds=text_embeds, image_features=new_vis)
            inputs_embeds = text_embeds.masked_scatter(mask, new_vis)
        else:
            inputs_embeds = text_embeds

        # Set rope_deltas from GAP-aware position_ids so the model
        # computes correct positions for autoregressively generated tokens.
        self.model.model.rope_deltas = manip_result.get("rope_deltas", None)

        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": manip_result["new_attention_mask"],
            "position_ids": position_ids,
            "max_new_tokens": max_new_tokens,
        }

        output_ids = self.model.generate(**gen_kwargs)
        return self.processor.decode(output_ids[0], skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate_progressive(self, manip_result, schedule,
                             merged_h, merged_w, grid_size=6,
                             max_new_tokens=256):
        """Generate with progressive layer-wise token pruning.

        Group 0 allocation is applied via V2 (token removal in inputs_embeds).
        Deeper groups use hooks to progressively remove more tokens.

        Args:
            manip_result: dict from TokenManipulatorV2.apply() (group 0 allocation)
            schedule: dict {layer_idx: {tile_id: tier}} from progressive_schedule()
            merged_h, merged_w: spatial dimensions after vision encoder merge
            grid_size: tile grid size
            max_new_tokens: generation length

        Returns:
            Generated text string
        """
        from models.progressive_manipulator import ProgressiveTokenManipulator

        new_ids = manip_result["new_input_ids"]
        new_vis = manip_result["new_visual_embeds"]
        position_ids = manip_result["position_ids"]

        # Build inputs_embeds (same as V2)
        text_embeds = self.model.model.language_model.get_input_embeddings()(new_ids)

        if new_vis.shape[0] > 0:
            mask, _ = self.model.model.get_placeholder_mask(
                new_ids, inputs_embeds=text_embeds, image_features=new_vis)
            inputs_embeds = text_embeds.masked_scatter(mask, new_vis)
        else:
            inputs_embeds = text_embeds

        self.model.model.rope_deltas = manip_result.get("rope_deltas", None)

        # Find visual token range in the new sequence
        config = self.model.config
        img_mask = (new_ids[0] == config.image_token_id)
        img_positions = torch.where(img_mask)[0]
        if len(img_positions) > 0:
            vis_start = img_positions[0].item()
            vis_end = img_positions[-1].item() + 1
        else:
            vis_start = vis_end = 0

        # Install progressive hooks
        ptm = ProgressiveTokenManipulator(
            self, schedule, vis_start, vis_end,
            merged_h, merged_w, grid_size,
            kept_orig_indices=manip_result.get("kept_orig_indices"),
        )
        ptm.install_hooks()

        try:
            gen_kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": manip_result["new_attention_mask"],
                "position_ids": position_ids,
                "max_new_tokens": max_new_tokens,
            }
            output_ids = self.model.generate(**gen_kwargs)
            return self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
        finally:
            ptm.remove_hooks()

    @torch.no_grad()
    def generate_baseline_progressive(self, inputs, vis_start, vis_end,
                                      boundary_spec, rank_fn,
                                      max_new_tokens=256,
                                      vis_token_positions=None):
        """Generate with progressive baseline pruning (PyramidDrop/FitPrune).

        Unlike BEA progressive which uses pre-computed allocations, this
        re-ranks visual tokens at each boundary using current hidden states.

        Args:
            inputs: prepared model inputs (from prepare_inputs)
            vis_start: start index of visual tokens in input_ids
            vis_end: end index (exclusive)
            boundary_spec: {layer_idx: num_tokens_to_drop}
            rank_fn: callable for ranking tokens at each boundary
            max_new_tokens: generation length
            vis_token_positions: optional list of actual IMAGE_TOKEN_ID
                positions for multi-image (non-contiguous visual segments)

        Returns:
            Generated text string
        """
        from baselines.common.progressive_manipulator import BaselineProgressiveManipulator

        if not boundary_spec:
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated = output_ids[0, inputs["input_ids"].shape[1]:]
            return self.processor.decode(generated, skip_special_tokens=True).strip()

        input_ids = inputs["input_ids"]
        text_embeds = self.model.model.language_model.get_input_embeddings()(input_ids)

        vis_out = self.model.model.visual(
            inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
        vis_embeds = vis_out.pooler_output

        if vis_embeds.shape[0] > 0:
            mask, _ = self.model.model.get_placeholder_mask(
                input_ids, inputs_embeds=text_embeds, image_features=vis_embeds)
            inputs_embeds = text_embeds.masked_scatter(mask, vis_embeds)
        else:
            inputs_embeds = text_embeds

        self.model.model.rope_deltas = None

        manip = BaselineProgressiveManipulator(
            self, boundary_spec, vis_start, vis_end, rank_fn=rank_fn,
            vis_token_positions=vis_token_positions)
        manip.patch()

        try:
            gen_kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": inputs.get("attention_mask"),
                "position_ids": inputs.get("position_ids"),
                "max_new_tokens": max_new_tokens,
            }
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
            output_ids = self.model.generate(**gen_kwargs)
            return self.processor.decode(
                output_ids[0], skip_special_tokens=True).strip()
        finally:
            manip.unpatch()
