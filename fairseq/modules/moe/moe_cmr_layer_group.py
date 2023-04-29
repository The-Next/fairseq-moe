# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import json

class CMRGate(torch.nn.Module):
    def __init__(self, model_dim: int, p: float = 0.0):
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, 1)
        self.dropout = torch.nn.Dropout(p=p)

    def forward(
        self,
        input: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        lang_embeddings=None,
    ) -> torch.Tensor:
        if lang_embeddings is not None:
            input = torch.cat([input, lang_embeddings], dim=1)
        logits = self.wg(input)
        gates = logits.squeeze(-1).sigmoid()
        # gates = self.dropout(gates)
        if input_mask is not None and input_mask.any():
            nonpadding = ~input_mask.bool()
            gates = gates * nonpadding.to(gates.dtype)
        return gates


class CMRGroupLayer(torch.nn.Module):
    def __init__(
        self,
        moe_layer: torch.nn.Module,
        ffn_fn: Callable,
        model_dim: int,
        p: float = 0.0,
        use_cmr_lang_perception = False,
        language_divide = None,#这是一个json文件
        # lang_idx: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.moe_layer = moe_layer
        self.ffn_fn = ffn_fn
        self.use_cmr_lang_perception = use_cmr_lang_perception
        if use_cmr_lang_perception:
            self.gate = CMRGate(model_dim*2, p)
        else:
            self.gate = CMRGate(model_dim, p)
        # if lang_idx is not None:
        #     self.register_buffer("lang_idx", lang_idx)
        # else:
        #     self.lang_idx = None
        self.group_dict = language_divide
        if self.group_dict is not None:
            with open(self.group_dict,'r',encoding='utf-8') as fp:
                data = json.load(fp)
            self.dict = data

    def forward(
        self,
        *input: torch.Tensor,
        input_padding_mask=None,
        lang_idx: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert len(input) == 1, "only single input Tensor supported"

        if self.use_cmr_lang_perception:
            lang_embeddings = kwargs.get("lang_embeddings", None)
            assert lang_embeddings is not None
            gates = self.gate(input[0], input_padding_mask, lang_embeddings)
        else:
            gates = self.gate(input[0], input_padding_mask)
        
        
        lang_list = list(set(lang_idx.tolist()))
        final_x_moe = torch.zeros_like(input[0],dtype = input[0].dtype)
        for lang_id in lang_list:
            lang_id_mask = torch.eq(lang_idx,lang_id)#制作掩码矩阵
            lang_id_mask = lang_id_mask.unsqueeze(-1)#[b,1]
            step1 = lang_id_mask.repeat([1,input[0].size(1)])#[b,len]
            step1 = step1.unsqueeze(-1)#[b,len,1]
            step2 = step1.repeat([1,1,input[0].size(-1)])#[b,len,d]

            lang_matrix = step2#[b,len,d]
            x_moe,l_aux = self.moe_layer(
                *input,input_padding_mask = input_padding_mask,lang_matrix = lang_matrix,langs_info = self.dict[str(lang_id)],**kwargs
            )
            final_x_moe = final_x_moe + x_moe
        # x_moe, l_aux = self.moe_layer(
        #     *input, input_padding_mask=input_padding_mask, **kwargs
        # )
        print('fin',final_x_moe)
        assert 1 == 0

        x_ffn = self.ffn_fn(*input)

        share_gates = self.gate.dropout(1 - gates)
        moe_gates = self.gate.dropout(gates)
        x_out = x_ffn * share_gates.unsqueeze(-1) + x_moe * moe_gates.unsqueeze(-1)

        if input_padding_mask is None:
            input_padding_mask = torch.zeros_like(input[0][:, 0], dtype=torch.bool)

        used_budget = (gates * (~input_padding_mask)).sum()
        total_budget = (~input_padding_mask).sum()

        l_aux["cmr_gate_loss_num"] = used_budget
        l_aux["cmr_gate_loss_denom"] = total_budget

        # self.moe_layer.metadata["cmr_share_rate"] = (1 - gates).mean().data
        # self.record_cmr_choices(gates)
        #self.moe_layer.metadata["cmr_lang_gates"] = 0
        # if prefix_tokens is not None and self.lang_idx is not None:
        #     num_langs = self.lang_idx.shape[0]
        #     # map lang token indices to lang_idx
        #     batch_langs = prefix_tokens.new_zeros(gates.shape[0])
        #     # non-matches have value 0 in batch_langs
        #     lang_match = torch.where(
        #         prefix_tokens.expand(-1, num_langs) == self.lang_idx
        #     )
        #     batch_langs[lang_match[0]] = lang_match[1]

        #     out = gates.new_zeros(num_langs, gates.shape[0])
        #     out[batch_langs, torch.arange(gates.shape[0])] = 1
        #     out = F.normalize(out, p=1, dim=1, eps=1e-5)

        #     # per-lang, (soft) fraction of tokens routed to MOE layers
        #     self.moe_layer.metadata["cmr_lang_gates"] = out.mm(
        #         gates.mean(dim=1, keepdim=True)
        #     ).detach()
        return x_out, l_aux

    def record_cmr_choices(self, gates):
        self.moe_layer.metadata['cmr_choices'] = gates.data # (s,)