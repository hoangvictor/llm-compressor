from typing import Optional

import os
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, SD35AdaLayerNormZeroX

from vllm.model_executor.layers.quantization.utils.w8a8_utils import apply_fp8_linear, apply_int8_linear

base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
activations_scales = torch.load(f'{base_dir}/data/activations_scales.pt')
fp8_quant_errors = torch.load(f'{base_dir}/data/quant_errors_fp8.pt')
int8_quant_errors = torch.load(f'{base_dir}/data/quant_errors_int8.pt')

def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output


class QuantLinear(nn.Module):

    def __init__(
        self,
        module: nn.Linear,
        old_module_name: str = None,
        quant_mode: str = 'fp8',
    ):
        super().__init__()        

        max_abs_val = torch.max(torch.abs(module.weight.T), axis=0).values.to(torch.float32)

        finfo = torch.finfo(torch.float8_e4m3fn)
        iinfo = torch.iinfo(torch.int8)
        
        assert quant_mode in ['fp8', 'int8']

        self.quant_mode = quant_mode
        self.old_module_name = old_module_name
        
        self.input_scale = None # activations_scales[self.old_module_name].to(torch.float32).unsqueeze(0) # torch.max(activations_scales[self.old_module_name].to(torch.float32)).unsqueeze(0) # None
        if self.input_scale is not None:
            if quant_mode == 'fp8':
                self.input_scale= (self.input_scale / finfo.max)
            else:
                self.input_scale= (self.input_scale / iinfo.max)
            
        if self.quant_mode == 'fp8':
            self.weight_scale = (max_abs_val /  finfo.max)
            self.weight_scale = self.weight_scale.contiguous()
            self.weight = (module.weight.T / self.weight_scale).clamp(min=finfo.min, max=finfo.max).to(torch.float8_e4m3fn)
        elif self.quant_mode == 'int8':
            self.weight_scale = (max_abs_val / iinfo.max)
            self.weight_scale = self.weight_scale.contiguous()
            self.weight = (module.weight.T / self.weight_scale).clamp(min=iinfo.min, max=iinfo.max).to(torch.int8)

        if module._parameters['bias'] is not None:
            self.bias = module.bias
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_2d = input.reshape(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], self.weight.shape[1]]
        scales = self.input_scale.repeat(input.shape[0], 1) if self.input_scale is not None else None
        if self.quant_mode == 'fp8':
            output = apply_fp8_linear(input_2d, self.weight, input_scale=scales, weight_scale=self.weight_scale, bias=self.bias, use_per_token_if_dynamic=True)
        elif self.quant_mode == 'int8':
            output = apply_int8_linear(input_2d, self.weight, input_scale=scales, weight_scale=self.weight_scale, bias=self.bias)
        else:
            raise Exception(f"The forward pass of this quant mode `{self.quant_mode}` is not implemented yet.")
        return output.view(*output_shape)

    def extra_repr(self) -> str:
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, bias={self.bias is not None}, quant_mode={self.quant_mode}"


@maybe_allow_in_graph
class JointTransformerBlock(nn.Module):
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

    def __init__(self, old_module: nn.Module):
        super().__init__()

        self.use_dual_attention = old_module.use_dual_attention
        self.context_pre_only = old_module.context_pre_only
        
        self.norm1 = old_module.norm1
        self.norm1_context = old_module.norm1_context

        self.attn = old_module.attn
        self.attn2 = old_module.attn2

        self.norm2 = old_module.norm2
        self.ff = old_module.ff

        self.norm2_context = old_module.norm2_context
        self.ff_context = old_module.ff_context

        self._chunk_size = old_module._chunk_size
        self._chunk_dim = old_module._chunk_dim

    def forward(
        self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, temb: torch.FloatTensor
    ):
        if self.use_dual_attention:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(
                hidden_states, emb=temb
            )
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        if self.use_dual_attention:
            attn_output2 = self.attn2(hidden_states=norm_hidden_states2)
            attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
            hidden_states = hidden_states + attn_output2

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states