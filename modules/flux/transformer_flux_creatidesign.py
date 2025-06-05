# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FluxTransformer2DLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from modules.flux.attention_processor_flux_creatidesign import (
    Attention,
    AttentionProcessor,
    DesignFluxAttnProcessor2_0,
    FluxAttnProcessor2_0_NPU,
    FusedFluxAttnProcessor2_0,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, FluxPosEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from modules.semantic_layout.layout_encoder import ObjectLayoutEncoder,ObjectLayoutEncoder_noFourier
from modules.common.lora import LoRALinearLayer

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name





@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0, rank=16,network_alpha=16,lora_weight=1.0,attention_type="design"):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)


        if is_torch_npu_available():
            processor = FluxAttnProcessor2_0_NPU()
        else:
            processor = DesignFluxAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

        self.attention_type = attention_type
        self.rank = rank
        self.network_alpha = network_alpha
        self.lora_weight = lora_weight
        if attention_type == "design":
            self.layernorm_subject = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) # layernorm for subject
            self.norm_subject_lora = nn.Sequential(
                nn.SiLU(),
                LoRALinearLayer(dim, dim*3, self.rank, self.network_alpha) # lora for adalinear of subject
            )
            self.layernorm_object_bbox = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) # layernorm for object
            self.norm_object_lora = nn.Sequential(
                nn.SiLU(),
                LoRALinearLayer(dim, dim*3, self.rank, self.network_alpha) # lora for adalinear of object
            )
    def single_block_adaln_lora_forward(self, x, temb, adaln, adaln_lora, layernorm, lora_weight):
        norm_x, x_gate = adaln(x, emb=temb)
        lora_shift_msa, lora_scale_msa, lora_gate_msa = adaln_lora(temb).chunk(3, dim=1)
        norm_x = norm_x + lora_weight * (layernorm(x)* (1 + lora_scale_msa[:, None]) + lora_shift_msa[:, None])
        x_gate = x_gate + lora_weight * lora_gate_msa
        return norm_x, x_gate
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        subject_hidden_states = None,
        subject_rotary_emb = None,
        object_bbox_hidden_states = None,
        object_rotary_emb = None,
        design_scale = 1.0,
        attention_mask=None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        
        # handle hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        #creatidesign
        use_subject = True if self.attention_type == "design" and subject_hidden_states is not None and design_scale!=0.0 else False
        use_object = True if self.attention_type == "design" and object_bbox_hidden_states is not None and design_scale!=0.0 else False
        # handle subejct_hidden_states
        if use_subject:
            residual_subject_hidden_states = subject_hidden_states
            norm_subject_hidden_states, subject_gate = self.single_block_adaln_lora_forward(subject_hidden_states, temb, self.norm, self.norm_subject_lora, self.layernorm_subject,  self.lora_weight)
            mlp_subject_hidden_states = self.act_mlp(self.proj_mlp(norm_subject_hidden_states))
        if use_object:
            residual_object_bbox_hidden_states = object_bbox_hidden_states
            norm_object_bbox_hidden_states, object_gate = self.single_block_adaln_lora_forward(object_bbox_hidden_states, temb, self.norm, self.norm_object_lora, self.layernorm_object_bbox,  self.lora_weight)
            mlp_object_bbox_hidden_states = self.act_mlp(self.proj_mlp(norm_object_bbox_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output, subject_attn_output, object_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            subject_hidden_states=norm_subject_hidden_states,
            subject_rotary_emb=subject_rotary_emb,
            object_bbox_hidden_states=norm_object_bbox_hidden_states,
            object_rotary_emb=object_rotary_emb,
            attention_mask = attention_mask,
            **joint_attention_kwargs,
        )
        # handle hidden states
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        #handle subject_hidden_states
        if use_subject:
            subject_hidden_states = torch.cat([subject_attn_output, mlp_subject_hidden_states], dim=2)
            subject_gate = subject_gate.unsqueeze(1)
            subject_hidden_states = subject_gate * self.proj_out(subject_hidden_states)
            subject_hidden_states = residual_subject_hidden_states + subject_hidden_states

        #handle object_bbox_hidden_states
        if use_object:
            object_bbox_hidden_states = torch.cat([object_attn_output, mlp_object_bbox_hidden_states], dim=2)
            object_gate = object_gate.unsqueeze(1)
            object_bbox_hidden_states = object_gate * self.proj_out(object_bbox_hidden_states)
            object_bbox_hidden_states = residual_object_bbox_hidden_states + object_bbox_hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states, subject_hidden_states, object_bbox_hidden_states


@maybe_allow_in_graph
class FluxTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Args:
        dim (`int`):
            The embedding dimension of the block.
        num_attention_heads (`int`):
            The number of attention heads to use.
        attention_head_dim (`int`):
            The number of dimensions to use for each attention head.
        qk_norm (`str`, defaults to `"rms_norm"`):
            The normalization to use for the query and key tensors.
        eps (`float`, defaults to `1e-6`):
            The epsilon value to use for the normalization.
    """

    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, qk_norm: str = "rms_norm", eps: float = 1e-6, rank=16, network_alpha=16, lora_weight=1.0,attention_type="design"
    ):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)

        self.norm1_context = AdaLayerNormZero(dim)

        if hasattr(F, "scaled_dot_product_attention"):
            processor = DesignFluxAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

        # creatidesign
        self.attention_type = attention_type
        self.rank = rank
        self.network_alpha = network_alpha
        self.lora_weight = lora_weight

        if self.attention_type == "design":
            # lora for handle subject (img branch)
            self.norm1_subject_lora = nn.Sequential(
                nn.SiLU(),
                LoRALinearLayer(dim, dim*6, self.rank, self.network_alpha) # lora for adalinear
            )
            self.layernorm_subject = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) # norm for subject

            # lora for handle object (txt branch)
            self.norm1_object_lora = nn.Sequential(
                nn.SiLU(),
                LoRALinearLayer(dim, dim*6, self.rank, self.network_alpha) # lora for adalinear
            )
            self.layernorm_object = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) # norm for object

    def double_block_adaln_lora_forward(self, x, temb, adaln, adaln_lora, layernorm, lora_weight):
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = adaln(x, emb=temb)
        lora_shift_msa, lora_scale_msa, lora_gate_msa, lora_shift_mlp, lora_scale_mlp, lora_gate_mlp = adaln_lora(temb).chunk(6, dim=1)
        norm_x = norm_x + lora_weight * (layernorm(x)* (1 + lora_scale_msa[:, None]) + lora_shift_msa[:, None])
        x_gate_msa = x_gate_msa + lora_weight*lora_gate_msa
        x_shift_mlp = x_shift_mlp + lora_weight*lora_shift_mlp
        x_scale_mlp = x_scale_mlp + lora_weight*lora_scale_mlp
        x_gate_mlp = x_gate_mlp + lora_weight*lora_gate_mlp
        return norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        subject_hidden_states = None,
        subject_rotary_emb = None,
        object_bbox_hidden_states = None,
        object_rotary_emb = None,
        design_scale = 1.0,
        attention_mask=None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}

        
        use_subject = True if self.attention_type == "design" and subject_hidden_states is not None and design_scale!=0.0 else False
        use_object = True if self.attention_type == "design" and object_bbox_hidden_states is not None and design_scale!=0.0 else False
        if use_subject:
            # subject adalinear
            norm_subject_hidden_states, subject_gate_msa, subject_shift_mlp, subject_scale_mlp, subject_gate_mlp = self.double_block_adaln_lora_forward(
                subject_hidden_states, temb, self.norm1, self.norm1_subject_lora, self.layernorm_subject, self.lora_weight
            )
        if use_object:
            # object adalinear
            norm_object_bbox_hidden_states, object_gate_msa, object_shift_mlp, object_scale_mlp, object_gate_mlp = self.double_block_adaln_lora_forward(
                object_bbox_hidden_states, temb, self.norm1_context, self.norm1_object_lora, self.layernorm_object, self.lora_weight
            )

    
        attn_output, context_attn_output, subject_attn_output, object_attn_output = self.attn(  
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            subject_hidden_states=norm_subject_hidden_states if use_subject else None,
            subject_rotary_emb=subject_rotary_emb if use_subject else None,
            object_bbox_hidden_states=norm_object_bbox_hidden_states if use_object else None,
            object_rotary_emb=object_rotary_emb if use_object else None,
            attention_mask = attention_mask,
            **joint_attention_kwargs,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
       


        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        
        
        # process attention outputs for the `subject_hidden_states`.
        if use_subject:
            subject_attn_output = subject_gate_msa.unsqueeze(1) * subject_attn_output
            subject_hidden_states = subject_hidden_states + subject_attn_output
            norm_subject_hidden_states = self.norm2(subject_hidden_states)
            norm_subject_hidden_states = norm_subject_hidden_states * (1 + subject_scale_mlp[:, None]) + subject_shift_mlp[:, None]
            subject_ff_output = self.ff(norm_subject_hidden_states)
            subject_hidden_states = subject_hidden_states + subject_gate_mlp.unsqueeze(1) * subject_ff_output

        # process attention outputs for the `object_bbox_hidden_states`.
        if use_object:
            object_attn_output = object_gate_msa.unsqueeze(1) * object_attn_output
            object_bbox_hidden_states = object_bbox_hidden_states + object_attn_output
            norm_object_bbox_hidden_states = self.norm2_context(object_bbox_hidden_states)
            norm_object_bbox_hidden_states = norm_object_bbox_hidden_states * (1 + object_scale_mlp[:, None]) + object_shift_mlp[:, None]
            object_ff_output = self.ff_context(norm_object_bbox_hidden_states)
            object_bbox_hidden_states = object_bbox_hidden_states + object_gate_mlp.unsqueeze(1) * object_ff_output
        
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states, subject_hidden_states, object_bbox_hidden_states


class FluxTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, FluxTransformer2DLoadersMixin
):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        patch_size (`int`, defaults to `1`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `19`):
            The number of layers of dual stream DiT blocks to use.
        num_single_layers (`int`, defaults to `38`):
            The number of layers of single stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `4096`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        pooled_projection_dim (`int`, defaults to `768`):
            The number of dimensions to use for the pooled projection.
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
        attention_type="design",
        max_boxes_token_length=30,
        rank = 16,
        network_alpha = 16,
        lora_weight = 1.0,
        use_attention_mask = True,
        use_objects_masks_maps=True,
        use_subject_masks_maps=True,
        use_layout_encoder=True,
        drop_subject_bg=False,
        gradient_checkpointing=False,
        use_fourier_bbox=True,
        bbox_id_shift=True
    ):
        super().__init__()
        # #creatidesign
        self.attention_type = attention_type
        self.max_boxes_token_length = max_boxes_token_length
        self.rank = rank
        self.network_alpha = network_alpha
        self.lora_weight = lora_weight
        self.use_attention_mask = use_attention_mask
        self.use_objects_masks_maps= use_objects_masks_maps
        self.num_attention_heads=num_attention_heads
        self.use_layout_encoder = use_layout_encoder
        self.use_subject_masks_maps = use_subject_masks_maps
        self.drop_subject_bg = drop_subject_bg
        self.gradient_checkpointing = gradient_checkpointing
        self.use_fourier_bbox = use_fourier_bbox
        self.bbox_id_shift = bbox_id_shift


        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_type=self.attention_type,
                    rank=self.rank,
                    network_alpha=self.network_alpha,
                    lora_weight=self.lora_weight,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_type=self.attention_type,
                    rank=self.rank,
                    network_alpha=self.network_alpha,
                    lora_weight=self.lora_weight,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)


        if self.attention_type =="design":
            if self.use_layout_encoder:
                if self.use_fourier_bbox:
                    self.object_layout_encoder = ObjectLayoutEncoder(
                        positive_len=self.inner_dim, out_dim=self.inner_dim, max_boxes_token_length=self.max_boxes_token_length
                    )
                else:
                    self.object_layout_encoder = ObjectLayoutEncoder_noFourier(
                        in_dim=self.inner_dim, out_dim=self.inner_dim
                    )

            
    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedFluxAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedFluxAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
        design_kwargs: dict | None = None,
        design_scale =1.0
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
       

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            # logger.warning(
            #     "Passing `txt_ids` 3d torch.Tensor is deprecated."
            #     "Please remove the batch dimension and pass it as a 2d torch Tensor"
            # )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            # logger.warning(
            #     "Passing `img_ids` 3d torch.Tensor is deprecated."
            #     "Please remove the batch dimension and pass it as a 2d torch Tensor"
            # )
            img_ids = img_ids[0]

        attention_mask_batch = None 
        # handle design infos
        if self.attention_type=="design" and design_kwargs is not None:

            # handle objects
            objects_boxes = design_kwargs['object_layout']['objects_boxes'].to(dtype=hidden_states.dtype, device=hidden_states.device) # [B,10,4]
            objects_bbox_text_embeddings = design_kwargs['object_layout']['bbox_text_embeddings'].to(dtype=hidden_states.dtype, device=hidden_states.device) # [B,10,512,4096]
            objects_bbox_masks = design_kwargs['object_layout']['bbox_masks'].to(dtype=hidden_states.dtype, device=hidden_states.device) # [B,10]
            #token Truncation
            objects_bbox_text_embeddings = objects_bbox_text_embeddings[:,:,:self.max_boxes_token_length,:]# [B,10,30,4096]

            # [B,10,30,4096] -> [B*10,30,4096] -> [B*10,30,3072] -> [B,10,30,3072]
            B, N, S, C = objects_bbox_text_embeddings.shape
            objects_bbox_text_embeddings = objects_bbox_text_embeddings.reshape(-1, S, C) #[B*10,30,4096]
            objects_bbox_text_embeddings = self.context_embedder(objects_bbox_text_embeddings) #[B*10,30,3072]
            objects_bbox_text_embeddings = objects_bbox_text_embeddings.reshape(B, N, S, -1) # [B,10,30,3072]

            if self.use_layout_encoder:
                if self.use_fourier_bbox:
                    object_bbox_hidden_states = self.object_layout_encoder(
                        boxes=objects_boxes,
                        masks=objects_bbox_masks,
                        positive_embeddings=objects_bbox_text_embeddings,
                    )# [B,10,30,3072]
                else:
                    object_bbox_hidden_states = self.object_layout_encoder(
                        positive_embeddings=objects_bbox_text_embeddings,
                    )# [B,10,30,3072]
            else:
                object_bbox_hidden_states = objects_bbox_text_embeddings

            object_bbox_hidden_states = object_bbox_hidden_states.contiguous().view(B, N*S, -1)  # [B,300,3072]

            # bbox_id shift
            if self.bbox_id_shift:
                object_bbox_ids = -1 * torch.ones(object_bbox_hidden_states.shape[0], object_bbox_hidden_states.shape[1], 3).to(device=object_bbox_hidden_states.device, dtype=object_bbox_hidden_states.dtype)         
            else:
                object_bbox_ids = torch.zeros(object_bbox_hidden_states.shape[0], object_bbox_hidden_states.shape[1], 3).to(device=object_bbox_hidden_states.device, dtype=object_bbox_hidden_states.dtype) 
            if object_bbox_ids.ndim == 3:
                object_bbox_ids = object_bbox_ids[0] #[300,3]
            object_rotary_emb = self.pos_embed(object_bbox_ids)


            
            # handle subjects
            subject_hidden_states = design_kwargs['subject_contion']['condition_img']
            subject_hidden_states = self.x_embedder(subject_hidden_states)
            subject_ids = design_kwargs['subject_contion']['condition_img_ids']
            if subject_ids.ndim == 3:
                subject_ids = subject_ids[0]
            subject_rotary_emb = self.pos_embed(subject_ids)
            
            
            
            if self.use_attention_mask:
                num_objects = N 
                tokens_per_object = self.max_boxes_token_length
                total_object_tokens = object_bbox_hidden_states.shape[1]
                assert total_object_tokens == num_objects * tokens_per_object, "Total object tokens do not match expected value"
                encoder_tokens = encoder_hidden_states.shape[1]
                img_tokens = hidden_states.shape[1]
                subject_tokens = subject_hidden_states.shape[1]
                # Total number of tokens
                total_tokens = total_object_tokens + encoder_tokens + img_tokens + subject_tokens

                attention_mask_batch = torch.zeros((B,total_tokens, total_tokens), dtype=hidden_states.dtype,device=hidden_states.device)
                img_H, img_W = design_kwargs['object_layout']['img_token_h'], design_kwargs['object_layout']['img_token_w']
                objects_masks_maps = design_kwargs['object_layout']['objects_masks_maps'].to(dtype=hidden_states.dtype, device=hidden_states.device) # [B,512,512]
                subject_H,subject_W = design_kwargs['subject_contion']['subject_token_h'], design_kwargs['subject_contion']['subject_token_w']
                subject_masks_maps = design_kwargs['subject_contion']['subject_masks_maps'].to(dtype=hidden_states.dtype, device=hidden_states.device) # [B,512,512]    
                for m_idx in range(B):
                    # Create the base mask (all False/blocked)
                    attention_mask = torch.zeros((total_tokens, total_tokens), dtype=hidden_states.dtype,device=hidden_states.device)

                    # Define token ranges
                    o_ranges = []  # Ranges for each object
                    start_idx = 0
                    for i in range(num_objects):
                        end_idx = start_idx + tokens_per_object
                        o_ranges.append((start_idx, end_idx))
                        start_idx = end_idx
                    
                    encoder_range = (total_object_tokens, total_object_tokens + encoder_tokens)
                    img_range = (encoder_range[1], encoder_range[1] + img_tokens)
                    subject_range = (img_range[1], img_range[1] + subject_tokens)

                    # Fill in the mask

                    # 1. Object self-attention (diagonal o₁-o₁, o₂-o₂, o₃-o₃)
                    for o_start, o_end in o_ranges:
                        attention_mask[o_start:o_end, o_start:o_end] = True

                    # 2. Objects to img and img to objetcs
                    
                    if not self.use_objects_masks_maps:
                        # all objects can attend to img and img can attend to all objects
                        for o_start, o_end in o_ranges:
                            attention_mask[o_start:o_end, img_range[0]:img_range[1]] = True
                        # img can attend to all
                        attention_mask[img_range[0]:img_range[1], :] = True
                    else:
                        # all objects can only attend to the bbox area (defined by objects_mask) of img
                        for idx, (o_start, o_end )in enumerate(o_ranges):
                            mask = objects_masks_maps[m_idx][idx]
                            mask = torch.nn.functional.interpolate(mask[None, None, :, :], (img_H, img_W), mode='nearest-exact').flatten().unsqueeze(1).repeat(1, tokens_per_object) #shape: [img_tokens,tokens_per_object]
                            
                            # objects to img
                            attention_mask[o_start:o_end, img_range[0]:img_range[1]] = mask.transpose(-1, -2)

                            # img to objects
                            attention_mask[img_range[0]:img_range[1], o_start:o_end] = mask
                    

                    # img to img 
                    attention_mask[img_range[0]:img_range[1], img_range[0]:img_range[1]] = True
                        
                    # img to prompt 
                    attention_mask[img_range[0]:img_range[1], encoder_range[0]:encoder_range[1]] = True
                    
                    # img to subject
                    subject_mask = subject_masks_maps[m_idx][0]

                    if not self.use_subject_masks_maps:
                        # all img can attend to subject
                        attention_mask[img_range[0]:img_range[1], subject_range[0]:subject_range[1]] = True
                    else:
                        # img can only attend to the bbox area (defined by subject_mask) of subject
                        
                        subject_mask_img = torch.nn.functional.interpolate(subject_mask[None, None, :, :], (img_H, img_W), mode='nearest-exact').flatten().unsqueeze(1).repeat(1, subject_tokens) #shape: [img_tokens,subject_tokens]
                        
                        # img to objects
                        attention_mask[img_range[0]:img_range[1], subject_range[0]:subject_range[1]] = subject_mask_img
                    
                    

                    # 3. prompt to prompt, prompt to img, and prompt to subject

                    # prompt to prompt
                    attention_mask[encoder_range[0]:encoder_range[1], encoder_range[0]:encoder_range[1]] = True
                    # prompt to img
                    attention_mask[encoder_range[0]:encoder_range[1], img_range[0]:img_range[1]] = True

                    # prompt to subject
                    if not self.use_subject_masks_maps:
                        attention_mask[encoder_range[0]:encoder_range[1], subject_range[0]:subject_range[1]] = True
                    else:
                        subject_mask_prompt = torch.nn.functional.interpolate(subject_mask[None, None, :, :], (subject_H, subject_W), mode='nearest-exact').flatten().unsqueeze(1).repeat(1, encoder_tokens) #shape: [subject_tokens,encoder_tokens]
                        attention_mask[encoder_range[0]:encoder_range[1], subject_range[0]:subject_range[1]] = subject_mask_prompt.transpose(-1, -2)


                    # 4. subject to prompt, subject to img, subject to subject
                    # subject to prompt
                    if not self.use_subject_masks_maps:
                        attention_mask[subject_range[0]:subject_range[1], encoder_range[0]:encoder_range[1]] = True
                    else:
                        attention_mask[subject_range[0]:subject_range[1], encoder_range[0]:encoder_range[1]] = False
                    
                    # subject to img
                    if not self.use_subject_masks_maps:
                        attention_mask[subject_range[0]:subject_range[1], img_range[0]:img_range[1]] = True
                    else:
                        attention_mask[subject_range[0]:subject_range[1], img_range[0]:img_range[1]] = subject_mask_img.transpose(-1, -2) 
                    # subject to subject
                    if not self.use_subject_masks_maps:
                        attention_mask[subject_range[0]:subject_range[1], subject_range[0]:subject_range[1]] = True
                    else:
                        # blcok non-subject region
                        if not self.drop_subject_bg:
                            attention_mask[subject_range[0]:subject_range[1], subject_range[0]:subject_range[1]] = True
                        else:
                            attention_mask[subject_range[0]:subject_range[1], subject_range[0]:subject_range[1]] = subject_mask_img 
                        

                    attention_mask_batch[m_idx] = attention_mask
 
                attention_mask_batch = attention_mask_batch.unsqueeze(1).to(dtype=torch.bool, device=hidden_states.device)#[B,2860,2860]->[B,1,2860,2860]
               
             
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})


        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states, subject_hidden_states, object_bbox_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    subject_hidden_states,
                    subject_rotary_emb,
                    object_bbox_hidden_states,
                    object_rotary_emb,
                    design_scale,
                    attention_mask_batch,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states, subject_hidden_states, object_bbox_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    subject_hidden_states=subject_hidden_states,
                    subject_rotary_emb=subject_rotary_emb,
                    object_bbox_hidden_states=object_bbox_hidden_states,
                    object_rotary_emb=object_rotary_emb,
                    design_scale = design_scale,
                    attention_mask = attention_mask_batch,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, subject_hidden_states, object_bbox_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    subject_hidden_states,
                    subject_rotary_emb,
                    object_bbox_hidden_states,
                    object_rotary_emb,
                    design_scale,
                    attention_mask_batch,
                    **ckpt_kwargs,
                )

            else:
                hidden_states, subject_hidden_states, object_bbox_hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    subject_hidden_states=subject_hidden_states,
                    subject_rotary_emb=subject_rotary_emb,
                    object_bbox_hidden_states=object_bbox_hidden_states,
                    object_rotary_emb=object_rotary_emb,
                    design_scale=design_scale,
                    attention_mask = attention_mask_batch,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
