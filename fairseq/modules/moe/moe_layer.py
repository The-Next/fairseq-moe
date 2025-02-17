# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: This is a mirror of the code in
# https://github.com/facebookresearch/fairscale/tree/master/fairscale/nn/moe

import logging
import time
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.cuda import Event as CudaEvent
from torch.nn import Module, ModuleList
from fairseq import distributed_utils

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x
    from tutel import moe as tutel_moe
    from tutel.impls import communicate as C

    has_tutel, fused_cumsum_sub_one = True, tutel_moe.fast_cumsum_sub_one
except Exception:
    has_tutel, fused_cumsum_sub_one = False, lambda mask: torch.cumsum(mask, dim=0) - 1

logger = logging.getLogger(__name__)


# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.

# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, input_splits=None, output_splits=None) -> Tensor:  # type: ignore
        ctx.group = group
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        input = input.contiguous()
        if output_splits is None:
            output = torch.empty_like(input)
        else:
            output=input.new_empty(size=[sum(output_splits)] + list(input.size()[1:]))
        if dist.is_initialized():
            dist.all_to_all_single(output, input, output_splits, input_splits, group)
        else:
            assert group is None
            output = input
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[None, Tensor]:
        grad_output=grad_output.contiguous()
        result = torch.empty_like(grad_output) if ctx.input_splits is None else \
            grad_output.new_empty(size=[sum(ctx.input_splits)] + list(grad_output.size()[1:]))
        dist.all_to_all_single(result, 
                                grad_output, 
                                output_split_sizes=ctx.input_splits, 
                                input_split_sizes=ctx.output_splits, 
                                group=ctx.group)
        return (None, result, None, None)


class MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(self, gate: Module, experts: Union[Module, ModuleList], args, group: Optional[Any] = None, all2all_group: Optional[Any] = None) -> None:
        super().__init__()
        self.gate = gate
        if type(experts) == ModuleList:
            self.experts = cast(ModuleList, experts)
        else:
            self.experts = ModuleList([experts])
        self.expert_group = group if group is not None else distributed_utils.get_moe_group(args.moe_expert_count)
        self.all2all_group = all2all_group if all2all_group is not None else distributed_utils.get_all2all_group(args.moe_expert_count)
        for p in experts.parameters():
            p.expert = True  # type: ignore
        self.world_size = distributed_utils.get_world_size(self.expert_group)
        self.all2all_size = distributed_utils.get_world_size(self.all2all_group)
        self.num_local_experts = len(self.experts)
        self.args = args
        self.in_generation = False
        self.a2a_cuda_event_intervals = []
        self.a2a_cpu_time_ms = 0.0
        self.use_tutel=getattr(args, 'use_tutel', False) and has_tutel
        self.use_tutel_all2all=getattr(args, 'use_tutel_all2all', False) and has_tutel
    

    def forward(self, *input: Tensor, input_padding_mask=None,lang_matrix = None,langs_info = None,lang_list = None,lang_idx = None, **kwargs: Any) -> Tensor:
        assert len(input) == 1, "only single input Tensor supported"
        input = input[0]
        assert len(input.shape) == 3, "input Tensor must have dimensions: bsz,seq, dmodel"
        # if input_padding_mask is not None:
        #     assert len(input_padding_mask.shape) == 2, "input Tensor must have dimensions: bsz,seq"
        #     assert input_padding_mask.shape[0]==input.shape[0]
        #     if input_padding_mask.shape[1] != input.shape[1]:
        #         input_padding_mask=None
        # assert input.shape[0] % len(self.experts) == 0, "num tokens must be order of number of local experts"

        # Implement Algorithm 2 from GShard paper.
        # count_non_zero = None
        # if lang_matrix is not None:#屏蔽其他语言
        #     input = input * lang_matrix
        #     count_non_zero = torch.count_nonzero(lang_matrix).clone().item()/512#统计张量中不为零的个数，此为实际token数
            
            # count_non_zero = (lang_matrix != 0).sum().item()
        d_model = input.shape[2]
        # Pad to expected batch size
        input_shape = list(input.shape)
        expected_bsz = getattr(self.args, 'batch_size', 0) if self.training else getattr(self.args, 'batch_size_valid', 0)
        # This indicates that --batch-size or --max-sentences is not specified
        if expected_bsz is None:
            expected_bsz = 0
        expected_bsz=int(expected_bsz)
        # print('ip',input.shape,input.device)
        
        # Reshape into S tokens by dropping sequence dimension.
        reshaped_input = input.reshape(-1, d_model)
        reshaped_input_shape = reshaped_input.shape
        reshaped_input_padding_mask = input_padding_mask.reshape(-1) if input_padding_mask is not None else None

        # Doing padding here when --max-tokens is specified and not --batch-size or --max-sentences
        # Pro of --max-tokens: more flexible for MT variable sequence lengths
        # Con of --max-tokens: extra all-reduce needed to figure out optimal padding without running OOM
        reshape_input_len = reshaped_input_shape[0]
        if expected_bsz == 0:
            expected_dim = int(distributed_utils.all_reduce(
                reshaped_input_shape[0] * torch.ones((1,), dtype=torch.long, device=input.device),
                group=dist.group.WORLD,
                op="max",
            ).item())
            padded_input = torch.zeros(
                (expected_dim, reshaped_input_shape[1]),
                dtype=input.dtype, layout=input.layout, device=input.device)
            padded_input[:reshaped_input_shape[0], :] = reshaped_input
            reshaped_input = padded_input

            padded_input_padding_mask = torch.ones(
                (expected_dim,), dtype=torch.bool, device=padded_input.device
            )
            if reshaped_input_padding_mask is not None:
                padded_input_padding_mask[:reshaped_input_shape[0]] = reshaped_input_padding_mask
            else:
                padded_input_padding_mask[:reshaped_input_shape[0]] = False
                reshaped_input_padding_mask = padded_input_padding_mask


        
        lang_embeddings = kwargs.get("lang_embeddings", None)
        #sentence_embeddings = kwargs.get("sentence_embeddings", None)
        if self.gate.use_moe_lang_perception:
            #assert sentence_embeddings is not None
            assert lang_embeddings is not None
            assert lang_embeddings.shape[0] == input_shape[0], f"{lang_embeddings.shape}, {input_shape}"
            assert lang_embeddings.shape[1] == input_shape[1], f"{lang_embeddings.shape}, {input_shape}"
            padded_lang_embeddings = torch.zeros(
                (expected_dim, lang_embeddings.shape[1]),
                dtype=lang_embeddings.dtype, layout=lang_embeddings.layout, device=lang_embeddings.device)
            padded_lang_embeddings[:lang_embeddings.shape[0], :] = lang_embeddings
            lang_embeddings = padded_lang_embeddings
            # pad for sentence embeddings
            # sentence_embeddings = sentence_embeddings.reshape(-1, d_model)
            # padded_sentence_embeddings = torch.zeros(
            #     (expected_dim, sentence_embeddings.shape[1]),
            #     dtype=sentence_embeddings.dtype, layout=sentence_embeddings.layout, device=sentence_embeddings.device)

            # padded_sentence_embeddings[:sentence_embeddings.shape[0], :] = sentence_embeddings
            # sentence_embeddings = padded_sentence_embeddings
        
        if self.use_tutel:
            l_aux, self.metadata, C, E, indices_, locations_, gates_ = self.gate(reshaped_input, reshaped_input_padding_mask, has_tutel=True)
            S, M = reshaped_input.size(0), reshaped_input.size(1)

            if not hasattr(self, '_tutel_dispatcher'):
                self._tutel_dispatcher = tutel_moe.fast_dispatcher(E, C, M, dispatch_dtype=reshaped_input.dtype)
            self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
            dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
        else:
            l_aux, combine_weights, dispatch_mask, self.metadata = self.gate(reshaped_input, reshaped_input_padding_mask, has_tutel=False, lang_embeddings=lang_embeddings,langs_info = langs_info,lang_list = lang_list,lang_idx = lang_idx,reshape_input_len = reshape_input_len,expected_dim = expected_dim)

            dispatch_mask = dispatch_mask.to(input.dtype).permute(1, 2, 0)  # S,E,C -> E,C,S
            E, C, S = dispatch_mask.size()
            M = reshaped_input.size(1)
            assert reshaped_input.size() == (S, M)
            # einsum("sec,sm->ecm")
            dispatched_input = torch.mm(dispatch_mask.view(E*C, S), reshaped_input)  # -> (E*C),M

        if self.all2all_size > 1:
            dispatched_input = self.all_to_all_wrapper(dispatched_input)

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.all2all_size, self.num_local_experts, -1, d_model)
        chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.experts):
            expert_outputs += [expert(chunk)]
        expert_output = torch.cat(expert_outputs, dim=1)

        if self.all2all_size > 1:
            expert_output = self.all_to_all_wrapper(expert_output)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.all2all_size * self.num_local_experts, -1, d_model)

        if self.use_tutel:
            combined_output = self._tutel_dispatcher.decode(expert_output.view(E*C, M))
        else:
            # einsum("sec,ecm->sm")
            combined_output = combine_weights.view(S, E*C).mm(expert_output.view(E*C, M))
        
        # Remove padding here when --max-tokens is specified and not --batch-size or --max-sentences
        combined_output = combined_output[:reshaped_input_shape[0], :] # (bsz*seq, dmodel)
        combined_output = combined_output.reshape(input.shape) 
        combined_output = combined_output[:input_shape[0], :, :]

        self.record_all_to_all_stats()
        self.record_expert_choices(dispatch_mask[:, :, :reshaped_input_shape[0]]) # dispatch_mask: ecs
        return combined_output, {"moe_gate_loss": l_aux}

    def prepare_for_inference_(self):
        self.in_generation = True

    def all_to_all_wrapper(self, input: Tensor, input_splits=None, output_splits=None):
        dummy_a2a = getattr(self.args, 'dummy_a2a', False)
        if dummy_a2a:
            input = input.contiguous()
            output = input.detach().clone()
            return input
        # always record times, since it is not a lot of overhead
        # if we do not log it we simply clear it off in record_all_to_all_stats
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cpu_start = time.time() * 1000
        cuda_start.record()
        if self.use_tutel_all2all:
            assert input_splits is None and output_splits is None, "tutel does not support input/output splits"
            C.AllToAllStatus.init(self.all2all_group, 1, -1)
            output = \
                C.CurrentStreamAcquire.apply(
                    C.NcclStreamRelease.apply(
                        C.AllToAll2DAsync.apply(
                            C.NcclStreamAcquire.apply(
                                C.CurrentStreamRelease.apply(input, 0), 0)), 0), 0)

        else:
            output = _AllToAll.apply(self.all2all_group, input, input_splits, output_splits)
        cuda_end.record()
        cpu_end = time.time() * 1000
        self.a2a_cpu_time_ms += (cpu_end - cpu_start)
        self.a2a_cuda_event_intervals.append((cuda_start, cuda_end))
        return output

    def record_all_to_all_stats(self):
        # controlled via an argument as we want to minimize any impact from torch.cuda.synchronize()
        record_a2a_perf_stats = getattr(self.args, 'record_a2a_perf_stats', False)
        if record_a2a_perf_stats:
            torch.cuda.synchronize()
            self.metadata["all_to_all_cpu_time_ms"] = self.a2a_cpu_time_ms
            a2a_cuda_time_ms = 0.0
            for ev_start, ev_end in self.a2a_cuda_event_intervals:
                a2a_cuda_time_ms += ev_start.elapsed_time(ev_end)
            self.metadata["all_to_all_cuda_time_ms"] = a2a_cuda_time_ms
        # reset stats
        self.a2a_cpu_time_ms = 0.0
        self.a2a_cuda_event_intervals = []
    
    def record_expert_choices(self, dispatch_mask):
        self.metadata['expert_choices']=dispatch_mask.sum(dim=1) # (e, s)