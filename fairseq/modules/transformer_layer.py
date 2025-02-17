# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.distributed.utils import get_global_rank
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import distributed_utils as dist_utils, utils
from fairseq.modules import gelu, LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.moe import Top1Gate, Top2Gate, MOELayer
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.fused_bias_gelu import fused_bias_gelu, has_fused_bias_gelu
from torch import Tensor

def _linear(x, weight, bias=None):
    return F.linear(x, weight, bias)


def _ffn(
    x,
    fc1,
    activation_fn,
    activation_dropout_module,
    fc2,
    dropout_module,
):
    x_shape = x.shape
    x = x.reshape(-1, x.size(-1))
    if has_fused_bias_gelu and activation_fn == gelu:
        x = _linear(x, fc1.weight)
        x = fused_bias_gelu(x, fc1.bias)
        x = activation_dropout_module(x)
        x = _linear(x, fc2.weight, fc2.bias)
    else:
        x = _linear(x, fc1.weight, fc1.bias)
        x = activation_fn(x)
        x = activation_dropout_module(x)
        x = _linear(x, fc2.weight, fc2.bias)
    x = x.view(x_shape)
    x = dropout_module(x)
    return x


class FeedForwardNetwork(nn.Module):
    """
        Feed Forward Network layer in the Transformer model
    """
    def __init__(self, args, embed_dim, ffn_dim, dropout_module=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.fc1 = self.build_fc1(
            self.embed_dim,
            ffn_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.dropout_module = FairseqDropout(
                args.dropout, module_name=self.__class__.__name__
            ) if not dropout_module else dropout_module

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def forward(self, x):
        return _ffn(
            x,
            fc1=self.fc1,
            activation_fn=self.activation_fn,
            activation_dropout_module=self.activation_dropout_module,
            fc2=self.fc2,
            dropout_module=self.dropout_module,
        )
        return x


class CMR_Gate(nn.Module):
    """
        Feed Forward Network layer in the Transformer model
    """
    def __init__(self, args, input_dim, ffn_dim, output_dim, dropout_module=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.fc1 = self.build_fc1(
            self.input_dim,
            ffn_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_dim,
            self.output_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.dropout_module = FairseqDropout(
                args.dropout, module_name=self.__class__.__name__
            ) if not dropout_module else dropout_module

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x
    
class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, is_moe_layer=False):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.is_moe_layer = is_moe_layer
        ffn_dim = args.encoder_ffn_embed_dim
        self.group_num = getattr(args,'group_num',8)
        self.language_divide = getattr(args,'language_divide',None)
        if self.is_moe_layer and getattr(args, "alternate_ffn_embed_dim", 0.0) > 0:
            ffn_dim = getattr(args, "alternate_ffn_embed_dim", 0.0)
        self.ffn_dim = ffn_dim
        # the second condition is for a "pseudo" MoE layer
        # (shared FFN with expert FFN dimension) that tries
        # to replicate FLOPs used by an expert MoE layer with perfectly balanced load
        if not self.is_moe_layer or getattr(args, "alternate_ffn_embed_dim", 0.0) > 0:
            self.activation_fn = utils.get_activation_fn(
                activation=getattr(args, 'activation_fn', 'relu') or "relu"
            )
            activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
            if activation_dropout_p == 0:
                # for backwards compatibility with models that use args.relu_dropout
                activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
            self.activation_dropout_module = FairseqDropout(
                float(activation_dropout_p), module_name=self.__class__.__name__
            )
            self.fc1 = self.build_fc1(
                self.embed_dim,
                ffn_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.fc2 = self.build_fc2(
                ffn_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
        else:
            gate=self.build_gate(args)
            experts = self.make_experts(args, self.embed_dim, ffn_dim, self.dropout_module)
            self.moe_layer = self.build_moe_layer(gate, experts, args)
            self.is_use_moe_cmr = args.use_moe_cmr

            if args.use_moe_cmr:
                from fairseq.modules.moe.moe_cmr_layer import CMRLayer
                self.share_expert = self.build_share_ffn(
                        args, self.embed_dim, ffn_dim, self.dropout_module
                    )
                self.moe_module = CMRLayer(self.moe_layer, self.share_expert, self.embed_dim, getattr(args, "moe_cmr_dropout", 0.0), use_cmr_lang_perception=getattr(args, "use_cmr_lang_perception", False))

                #self.cmr_gate = self.build_cmr_gate(args, self.embed_dim, self.dropout_module) # share or moe
                #self.max_temp, self.min_temp, self.temp_decay = [float(x) for x in args.moe_cmr_temp.split(",")]
            elif args.use_moe_cmr_group:
                from fairseq.modules.moe.moe_cmr_layer_group import CMRGroupLayer
                self.share_expert = self.build_share_ffn(
                        args, self.embed_dim, ffn_dim, self.dropout_module
                    )
                self.moe_module = CMRGroupLayer(self.moe_layer, self.share_expert, self.embed_dim, getattr(args, "moe_cmr_dropout", 0.0), use_cmr_lang_perception=getattr(args, "use_cmr_lang_perception", False),language_divide = self.language_divide)
            else:
                self.moe_module = self.moe_layer


        self.final_layer_norm = LayerNorm(self.embed_dim)
    # def build_cmr_gate(self, args, embed_dim, dropout_module):
    #     return CMR_Gate(args, embed_dim, embed_dim // 4, 2, dropout_module)

    def build_share_ffn(self, args, embed_dim, expert_ffn_dim, dropout_module):
        return FeedForwardNetwork(args, embed_dim, expert_ffn_dim, dropout_module)
    def build_gate(self, args):
        if args.moe_top1_expert:
            gate = Top1Gate(
                self.embed_dim,
                args.moe_expert_count,
                use_fp32=args.moe_gating_use_fp32,
                moe_eval_capacity_token_fraction=getattr(args, "moe_eval_capacity_token_fraction", 0.25),
                batch_prioritized_routing=getattr(args, "moe_batch_prioritized_routing", False),
                capacity_factor=getattr(args, "capacity_factor", 1.0),
                moe_expert_output_masking=getattr(args, "moe_expert_output_masking", 0.0),
                use_moe_lang_perception=getattr(args, "use_moe_lang_perception", False) or getattr(args, "use_encoder_moe_lang_perception", False)
            )
        else:
            gate = Top2Gate(
                self.embed_dim,
                args.moe_expert_count,
                args.moe_gating_use_fp32,
                args.moe_second_expert_policy,
                args.moe_normalize_gate_prob_before_dropping,
                getattr(args, "moe_eval_capacity_token_fraction", 0.25),
                getattr(args, "moe_batch_prioritized_routing", False),
                getattr(args, "capacity_factor", 1.0),
                moe_expert_output_masking=getattr(args, "moe_expert_output_masking", 0.0),
                use_moe_lang_perception=getattr(args, "use_moe_lang_perception", False) or getattr(args, "use_encoder_moe_lang_perception", False),
            )
        return gate

    def make_experts(self, args, embed_dim, expert_ffn_dim, dropout_module):
        return make_experts(args, embed_dim, expert_ffn_dim, dropout_module)

    def build_moe_layer(self, gate, experts, args):
        return MOELayer(gate, experts, args)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]
    def forward(self, 
                x, 
                encoder_padding_mask: Optional[Tensor],
                attn_mask: Optional[Tensor] = None,
                lang_embeddings: Optional[Tensor] = None,
                src_idx = None,
            ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        if not self.is_moe_layer or getattr(self.args, "alternate_ffn_embed_dim", 0.0) > 0:
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            l_aux = None
        else:
            # x - seq_len, batch_size, model_dim
            x = x.transpose(0, 1) # batch_size, seq_len, model_dim
            # sentence_embeddings = None
            # if self.moe_layer.gate.use_moe_lang_perception:
            #     assert len(x.shape) == 3
            #     sentence_padding_mask = torch.ones((x.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)
            #     if encoder_padding_mask is not None:
            #         sentence_padding_mask[encoder_padding_mask] = 0
            #     sentence_embeddings = (x * sentence_padding_mask.unsqueeze(-1)).sum(dim=1) / torch.clamp(sentence_padding_mask.sum(dim=1).unsqueeze(-1), min=1e-9)
            #     sentence_embeddings = sentence_embeddings.unsqueeze(1).expand(x.shape) # (bsz, seq, hidden)

            x_shape = x.shape
            if lang_embeddings is not None:
                lang_embeddings = lang_embeddings.expand(x_shape)
            # drop pad token
            if encoder_padding_mask is not None:
                nonpadding =~encoder_padding_mask.bool()
                x = x[nonpadding] # drop pad token
                if lang_embeddings is not None:
                    lang_embeddings = lang_embeddings[nonpadding]
            # else:
            #     # reshape x into (bsz*seq, dmodel)
            #     x = x.reshape(-1, x.shape[-1])
            #     if lang_embeddings is not None:
            #         lang_embeddings = lang_embeddings.reshape(-1, x.shape[-1])
 
            x, l_aux = self.moe_module(x, lang_embeddings=lang_embeddings,lang_idx = src_idx)

            if encoder_padding_mask is not None:
                new_x = torch.zeros(x_shape, device=x.device, dtype=x.dtype)
                new_x[~encoder_padding_mask.bool()] = x
                x = new_x
            else:
                x = x.reshape(x_shape)
            
            x = x.transpose(0, 1) # seq_len, batch_size, model_dim
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, l_aux


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, is_moe_layer=False,
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.normalize_before = args.decoder_normalize_before
        self.group_num = getattr(args,'group_num',8)
        self.language_divide = getattr(args,'language_divide',None)

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)

        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.is_moe_layer = is_moe_layer

        ffn_dim = args.decoder_ffn_embed_dim
        if self.is_moe_layer and getattr(args, "alternate_decoder_ffn_embed_dim", 0.0) > 0:
            ffn_dim = getattr(args, "alternate_decoder_ffn_embed_dim", 0.0)

        if not self.is_moe_layer or getattr(args, "alternate_decoder_ffn_embed_dim", 0.0) > 0:
            self.activation_fn = utils.get_activation_fn(
                activation=str(args.activation_fn)
                if getattr(args, "activation_fn", None) is not None
                else "relu"
            )
            activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
            if activation_dropout_p == 0:
                # for backwards compatibility with models that use args.relu_dropout
                activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
            self.activation_dropout_module = FairseqDropout(
                float(activation_dropout_p), module_name=self.__class__.__name__
            )
            self.fc1 = self.build_fc1(
                self.embed_dim,
                ffn_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.fc2 = self.build_fc2(
                ffn_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
        else:
            gate=self.build_gate(args)
            experts = self.make_experts(args, self.embed_dim, ffn_dim, self.dropout_module)
            self.moe_layer = self.build_moe_layer(gate, experts, args)
            self.is_use_moe_cmr = args.use_moe_cmr
            if args.use_moe_cmr:
                from fairseq.modules.moe.moe_cmr_layer import CMRLayer
                self.share_expert = self.build_share_ffn(
                        args, self.embed_dim, ffn_dim, self.dropout_module
                    )
                self.moe_module = CMRLayer(self.moe_layer, self.share_expert, self.embed_dim, getattr(args, "moe_cmr_dropout", 0.0), use_cmr_lang_perception=getattr(args, "use_cmr_lang_perception", False))

                #self.cmr_gate = self.build_cmr_gate(args, self.embed_dim, self.dropout_module) # share or moe
                #self.max_temp, self.min_temp, self.temp_decay = [float(x) for x in args.moe_cmr_temp.split(",")]
            elif args.use_moe_cmr_group:
                from fairseq.modules.moe.moe_cmr_layer_group import CMRGroupLayer
                self.share_expert = self.build_share_ffn(
                        args, self.embed_dim, ffn_dim, self.dropout_module
                    )
                self.moe_module = CMRGroupLayer(self.moe_layer, self.share_expert, self.embed_dim, getattr(args, "moe_cmr_dropout", 0.0), use_cmr_lang_perception=getattr(args, "use_cmr_lang_perception", False),language_divide = self.language_divide)

            else:
                self.moe_module = self.moe_layer

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

        self.args = args
    # def build_cmr_gate(self, args, embed_dim, dropout_module):
    #     return CMR_Gate(args, embed_dim, embed_dim // 4, 2, dropout_module)
    def build_share_ffn(self, args, embed_dim, expert_ffn_dim, dropout_module):
        return FeedForwardNetwork(args, embed_dim, expert_ffn_dim, dropout_module)
    def build_gate(self, args):
        if args.moe_top1_expert:
            gate = Top1Gate(
                self.embed_dim,
                args.moe_expert_count,
                use_fp32=args.moe_gating_use_fp32,
                moe_eval_capacity_token_fraction=getattr(args, "moe_eval_capacity_token_fraction", 0.25),
                batch_prioritized_routing=getattr(args, "moe_batch_prioritized_routing", False),
                capacity_factor=getattr(args, "capacity_factor", 1.0),
                moe_expert_output_masking=getattr(args, "moe_expert_output_masking", 0.0),
                use_moe_lang_perception=getattr(args, "use_moe_lang_perception", False) or getattr(args, "use_decoder_moe_lang_perception", False)
            )
        else:
            gate = Top2Gate(
                self.embed_dim,
                args.moe_expert_count,
                args.moe_gating_use_fp32,
                args.moe_second_expert_policy,
                args.moe_normalize_gate_prob_before_dropping,
                getattr(args, "moe_eval_capacity_token_fraction", 0.25),
                getattr(args, "moe_batch_prioritized_routing", False),
                getattr(args, "capacity_factor", 1.0),
                moe_expert_output_masking=getattr(args, "moe_expert_output_masking", 0.0),
                use_moe_lang_perception=getattr(args, "use_moe_lang_perception", False) or getattr(args, "use_decoder_moe_lang_perception", False),
            )
        return gate

    def build_moe_layer(self, gate, experts, args):
        return MOELayer(gate, experts, args)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        lang_embeddings: Optional[Tensor] = None,
        tgt_idx = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        if not self.is_moe_layer or getattr(self.args, "alternate_decoder_ffn_embed_dim", 0.0) > 0:
            x = _ffn(
                x,
                fc1=self.fc1,
                activation_fn=self.activation_fn,
                activation_dropout_module=self.activation_dropout_module,
                fc2=self.fc2,
                dropout_module=self.dropout_module,
            )
            l_aux = None
        else:
            # x - seq_len, batch_size, model_dim
            x = x.transpose(0, 1) # batch_size, seq_len, model_dim
            # sentence_embeddings = None
            # if self.moe_layer.gate.use_moe_lang_perception:
            #     if incremental_state is None: # training
            #         if self_attn_mask is not None:
            #             sentence_padding_mask = (self_attn_mask==0).unsqueeze(0).expand((x.shape[0], x.shape[1],x.shape[1]))
            #             sentence_padding_mask = sentence_padding_mask.type(x.dtype)
            #         else:
            #             sentence_padding_mask = torch.ones((x.shape[0], x.shape[1], x.shape[1]), device=x.device, dtype=x.dtype)
            #         sentence_embeddings = sentence_padding_mask.bmm(x) / sentence_padding_mask.sum(dim=-1, keepdim=True) # (bsz, seq, hidden_size)
            #         if self_attn_padding_mask is not None:
            #             sentence_embeddings[self_attn_padding_mask.bool()] = 0
            #     else: # inference
            #         # load the previous hidden state from incremental_state
            #         prev_hidden_state = self.encoder_attn.get_incremental_state(incremental_state, "prev_hidden_state")
            #         if prev_hidden_state is None: # the first step in decoding
            #             sentence_embeddings = x # (bsz, 1, hidden_size)
            #             self.encoder_attn.set_incremental_state(incremental_state, "prev_hidden_state", x)
            #         else:
            #             all_hidden_state = torch.cat([prev_hidden_state, x], dim=1) # (bsz, seq, hidden_size)
            #             sentence_embeddings = all_hidden_state.mean(dim=1, keepdim=True) # (bsz, 1, hidden_size)
            #             self.encoder_attn.set_incremental_state(incremental_state, "prev_hidden_state", all_hidden_state)
            
            x_shape = x.shape
            if lang_embeddings is not None:
                lang_embeddings = lang_embeddings.expand(x_shape)
            # drop pad token
            # if self.training and self_attn_padding_mask is not None:
            #     nonpadding =~self_attn_padding_mask.bool()
            #     x = x[nonpadding] # drop pad token
            #     if lang_embeddings is not None:
            #         lang_embeddings = lang_embeddings[nonpadding]
            # else:
            #     # reshape x into (bsz*seq, dmodel)
            #     x = x.reshape(-1, x.shape[-1])
            #     if lang_embeddings is not None:
            #         lang_embeddings = lang_embeddings.reshape(-1, x.shape[-1])

            x, l_aux = self.moe_module(x, lang_embeddings=lang_embeddings,lang_idx = tgt_idx)

            # if self.training and self_attn_padding_mask is not None:
            #     new_x = torch.zeros(x_shape, device=x.device, dtype=x.dtype)
            #     new_x[~self_attn_padding_mask.bool()] = x
            #     x = new_x
            # else:
            #     x = x.reshape(x_shape)

            x = x.transpose(0, 1) # seq_len, batch_size, model_dim
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None, l_aux

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn

    def make_experts(self, args, embed_dim, expert_ffn_dim, dropout_module):
        return make_experts(args, embed_dim, expert_ffn_dim, dropout_module)


def make_experts(args, embed_dim, expert_ffn_dim, dropout_module) -> nn.ModuleList:
    world_size = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()
    expert_list = []
    ddp_rank = dist_utils.get_data_parallel_rank()
    start_seed = torch.randint(1000000, (1,)).item()
    # at least as many experts than gpus
    if args.moe_expert_count >= world_size:
        assert args.moe_expert_count % world_size == 0, f'{args.moe_expert_count}, {world_size}'
        local_moe_expert_count = args.moe_expert_count // world_size
        for i in range(local_moe_expert_count):
            with utils.set_torch_seed(start_seed + ddp_rank * local_moe_expert_count + i):
                expert_list.append(FeedForwardNetwork(args, embed_dim, expert_ffn_dim, dropout_module))
    # less experts than gpus
    else:
        assert world_size % args.moe_expert_count == 0, f'{world_size}, {args.moe_expert_count}'
        # initialize each FFN with the same seed on different GPUs
        with utils.set_torch_seed(start_seed + ddp_rank % args.moe_expert_count):
            expert_list.append(FeedForwardNetwork(args, embed_dim, expert_ffn_dim, dropout_module))
    experts = nn.ModuleList(expert_list)
    return experts
