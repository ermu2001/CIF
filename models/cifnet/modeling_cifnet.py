# coding=utf-8
# Copyright 2022 Microsoft Research, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch ResNet model."""

import functools
import math
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import einops

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.backbone_utils import BackboneMixin
from .configuration_cifnet import CifNetConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ResNetConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "microsoft/resnet-50"
_EXPECTED_OUTPUT_SHAPE = [1, 2048, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "microsoft/resnet-50"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tiger cat"

RESNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/resnet-50",
    # See all resnet models at https://huggingface.co/models?filter=resnet
]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class CifNetConvLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: str = "relu"
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = ACT2FN[activation] if activation is not None else nn.Identity()

    def forward(self, input: Tensor) -> Tensor:
        hidden_states = self.convolution(input)
        hidden_states = self.normalization(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class CifNetEmbeddings(nn.Module):
    """
    ResNet Embeddings (stem) composed of a single aggressive convolution.
    """

    def __init__(self, config: CifNetConfig):
        super().__init__()
        embedding_kwargs = config.embedding_kwargs
        self.embedder = CifNetConvLayer(
            config.num_channels,
            embedding_kwargs['embedding_size'],
            kernel_size=embedding_kwargs['embedding_kernel_size_1'],
            stride=embedding_kwargs['embedding_stride_1'],
            activation=config.activation,
        )
        self.pooler = nn.MaxPool2d(
            kernel_size=embedding_kwargs['embedding_kernel_size_2'],
            stride=embedding_kwargs['embedding_stride_2'],
        )
        self.num_channels = config.num_channels

    def forward(self, pixel_values: Tensor) -> Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embedding = self.embedder(pixel_values)
        embedding = self.pooler(embedding)
        return embedding


class CifNetShortCut(nn.Module):
    """
    ResNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.normalization = nn.BatchNorm2d(out_channels)

    def forward(self, input: Tensor) -> Tensor:
        hidden_states = self.convolution(input)
        hidden_states = self.normalization(hidden_states)
        return hidden_states


class CifNetBasicLayer(nn.Module):
    """
    A classic ResNet's residual layer composed by two `3x3` convolutions.
    """

    def __init__(
            self,
            config: CifNetConfig,
            in_channels,
            out_channels
        ):
        super().__init__()
        self.config = config
        stride = 2 if out_channels > in_channels else 1 # up scale channel if downscale resolution 
        should_apply_shortcut = out_channels != in_channels
        self.shortcut = (
            CifNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
        )
        self.layer = nn.Sequential(
            CifNetConvLayer(in_channels, out_channels, config.main_kernel_size, stride=stride, activation=config.activation),
            CifNetConvLayer(out_channels, out_channels, config.main_kernel_size, activation=config.activation),
        )

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer(hidden_states)
        residual = self.shortcut(residual)
        hidden_states += residual
        return hidden_states


class CifNetBottleNeckLayer(nn.Module):
    """
    A classic ResNet's bottleneck layer composed by three `3x3` convolutions.

    The first `1x1` convolution reduces the input by a factor of `reduction` in order to make the second `3x3`
    convolution faster. The last `1x1` convolution remaps the reduced features to `out_channels`. If
    `downsample_in_bottleneck` is true, downsample will be in the first layer instead of the second layer.
    """

    def __init__(
        self,
        config: CifNetConfig,
        in_channels, 
        out_channels,
    ):
        super().__init__()
        self.config = config
        should_apply_shortcut = in_channels != out_channels
        stride = 2 if out_channels > in_channels else 1 # up scale channel if downscale resolution 
        reduces_channels = out_channels // config.bottleneck_kwargs['reduction']
        self.shortcut = (
            CifNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
        )
        self.layer = nn.Sequential(
            CifNetConvLayer(in_channels, reduces_channels, kernel_size=1, stride=1, activation=config.activation),
            CifNetConvLayer(reduces_channels, reduces_channels, kernel_size=config.main_kernel_size, stride=stride, activation=config.activation),
            CifNetConvLayer(reduces_channels, out_channels, kernel_size=1, stride=1, activation=config.activation),
        )

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer(hidden_states)
        residual = self.shortcut(residual)
        hidden_states += residual
        return hidden_states

class CifNetRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class CifNetSelfAttention(nn.Module):
    def __init__(
        self,
        config,
        hidden_size,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn_channels = config.attention_kwargs['attn_channels']
        self.num_heads = config.attention_kwargs['num_heads']
        self.attn_kernel_size = config.attention_kwargs['attn_kernel_size']
        self.attn_stride = config.attention_kwargs['attn_stride']
        self.attention_bias = config.attention_kwargs['attention_bias']
        self.attention_dropout = config.attention_kwargs['attention_dropout']
        self.max_position_embeddings = config.attention_kwargs['max_position_embeddings']
        # self.rope_theta = config.attention_kwargs['rope_theta']
        # self.rope_scaling_type = config.attention_kwargs['rope_scaling_type']
        # self.rope_scaling_factor = config.attention_kwargs['rope_scaling_factor']


        self.intermediate_size = self.num_heads * self.attn_channels
        # self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.attn_channels, bias=config.attention_bias)
        # self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.attn_channels, bias=config.attention_bias)
        # self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.attn_channels, bias=config.attention_bias)
        # self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

        self.q_proj = nn.Conv2d(self.hidden_size, self.intermediate_size, kernel_size=self.attn_kernel_size, stride=self.attn_stride, padding=self.attn_kernel_size // 2, bias=self.attention_bias)
        self.k_proj = nn.Conv2d(self.hidden_size, self.intermediate_size, kernel_size=self.attn_kernel_size, stride=self.attn_stride, padding=self.attn_kernel_size // 2, bias=self.attention_bias)
        self.v_proj = nn.Conv2d(self.hidden_size, self.intermediate_size, kernel_size=self.attn_kernel_size, stride=self.attn_stride, padding=self.attn_kernel_size // 2, bias=self.attention_bias)
        self.o_proj = nn.Conv2d(self.intermediate_size, self.hidden_size, kernel_size=self.attn_kernel_size, stride=self.attn_stride, padding=self.attn_kernel_size // 2, bias=self.attention_bias)

    def forward(
            self,
            hidden_states,
            output_attentions=False,
        ):
        # MHSA
        bsz, c, h, w = hidden_states.size()
        q_len = h*w
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = einops.rearrange(query_states, 'b (num_heads attn_channels) h w -> b num_heads (h w) attn_channels', num_heads=self.num_heads, attn_channels=self.attn_channels)
        key_states = einops.rearrange(key_states, 'b (num_heads attn_channels) h w -> b num_heads (h w) attn_channels', num_heads=self.num_heads, attn_channels=self.attn_channels)
        value_states = einops.rearrange(value_states, 'b (num_heads attn_channels) h w -> b num_heads (h w) attn_channels', num_heads=self.num_heads, attn_channels=self.attn_channels)

        # cos, sin = self.rotary_emb(value_states, )
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.attn_channels)
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.attn_channels):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.attn_channels)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.contiguous()

        attn_output = einops.rearrange(attn_output, "b num_heads (h w) attn_channels -> b (num_heads attn_channels) h w", h=h, w=w) 
        # attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output

class CifNetSelfAttentionLayer(nn.Module):
    def __init__(
        self,
        config,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.config = config
        should_apply_shortcut = in_channels != out_channels
        stride = 2 if out_channels > in_channels else 1 # up scale channel if downscale resolution 
        self.shortcut = (
            CifNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
        )
        self.in_conv = CifNetConvLayer(in_channels, out_channels, config.main_kernel_size, stride=stride, activation=config.activation)

        self.attention = CifNetSelfAttention(config, out_channels)
        self.activation = ACT2FN[config.activation]
        self.attention_norm = CifNetRMSNorm(out_channels)

        self.out_conv = CifNetConvLayer(out_channels, out_channels, config.main_kernel_size, stride=1, activation=config.activation)
        
    def forward(
        self,
        hidden_states,
    ):
        residual = hidden_states

        hidden_states = self.in_conv(hidden_states) 
        
        
        hidden_states = self.attention(hidden_states)
        hidden_states = einops.rearrange(hidden_states, "b c h w -> b h w c") 
        hidden_states = self.attention_norm(hidden_states)
        hidden_states = einops.rearrange(hidden_states, "b h w c-> b c h w") 
        hidden_states = self.activation(hidden_states)

        hidden_states = self.out_conv(hidden_states) 

        residual = self.shortcut(residual)
        hidden_states += residual
        return hidden_states

class CifNetStage(nn.Module):
    """
    A ResNet stage composed by stacked layers.
    """

    def __init__(
        self,
        config: CifNetConfig,
        in_channels,
        out_channels,
        depth,
    ):
        super().__init__()
        match config.layer_type:
            case "attention":
                layer = CifNetSelfAttentionLayer
            case "bottleneck":
                layer = CifNetBottleNeckLayer
            case "basic":
                layer = CifNetBasicLayer
            case _:
                raise ValueError(f"Not Valid config.layer_type {config.layer_type}")

        first_layer = layer(config, in_channels, out_channels)
        self.layers = nn.Sequential(
            first_layer, *[layer(config, in_channels, out_channels) for _ in range(depth - 1)]
        )

    def forward(self, input: Tensor) -> Tensor:
        hidden_states = input
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class CifNetEncoder(nn.Module):
    def __init__(self, config: CifNetConfig):
        super().__init__()
        self.stages = nn.ModuleList([])
        # based on `downsample_in_first_stage` the first layer of the first stage may or may not downsample the input
        self.stages.append(
            CifNetStage(
                config,
                config.embedding_kwargs['embedding_size'],
                config.hidden_sizes[0],
                1,
            )
        )
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for (in_channels, out_channels), depth in zip(in_out_channels, config.depths[1:]):
            self.stages.append(CifNetStage(config, in_channels, out_channels, depth,))

    def forward(
        self, hidden_states: Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> BaseModelOutputWithNoAttention:
        all_hidden_states = () if output_hidden_states else None
        for stage_module in self.stages:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = stage_module(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class CifNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CifNetConfig
    base_model_prefix = "resnet"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


RESNET_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ResNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

RESNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`ConvNextImageProcessor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare ResNet model outputting raw features without any specific head on top.",
    RESNET_START_DOCSTRING,
)
class CifNetModel(CifNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedder = CifNetEmbeddings(config)
        self.encoder = CifNetEncoder(config)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ) -> BaseModelOutputWithPoolingAndNoAttention:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embedder(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict
        )

        last_hidden_state = encoder_outputs[0]

        pooled_output = self.pooler(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@add_start_docstrings(
    """
    ResNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    RESNET_START_DOCSTRING,
)
class CifNetForImageClassification(CifNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.resnet = CifNetModel(config)
        # classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity(),
        )
        # initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ImageClassifierOutputWithNoAttention:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.resnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output

        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)


@add_start_docstrings(
    """
    ResNet backbone, to be used with frameworks like DETR and MaskFormer.
    """,
    RESNET_START_DOCSTRING,
)
class CifNetBackbone(CifNetPreTrainedModel, BackboneMixin):
    def __init__(self, config):
        super().__init__(config)
        super()._init_backbone(config)

        self.num_features = [config.embedding_size] + config.hidden_sizes
        self.embedder = CifNetEmbeddings(config)
        self.encoder = CifNetEncoder(config)

        # initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ) -> BackboneOutput:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        >>> model = AutoBackbone.from_pretrained(
        ...     "microsoft/resnet-50", out_features=["stage1", "stage2", "stage3", "stage4"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 2048, 7, 7]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        embedding_output = self.embedder(pixel_values)

        outputs = self.encoder(embedding_output, output_hidden_states=True, return_dict=True)

        hidden_states = outputs.hidden_states

        feature_maps = ()
        for idx, stage in enumerate(self.stage_names):
            if stage in self.out_features:
                feature_maps += (hidden_states[idx],)

        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )
