# License-Identifier: "LLAMA 2 COMMUNITY LICENSE"
# License-Details: https://github.com/facebookresearch/llama/blob/main/LICENSE    
# Copyright (C) 2023 Meta.
# 
# @file llama_v2.py
# @date 28 August 2023
# @brief Modified LLaMA v2 (It don't use parallel library)
#
# @author Seungbaek Hong <sb92.hong@samsung.com>


import os
os.environ['HTTP_PROXY'] = 'http://10.112.1.184:8080'
os.environ['HTTPS_PROXY'] = 'http://10.112.1.184:8080'
os.environ['NO_PROXY']="127.0.0.1, localhost"
os.environ['SSL_CERT_FILE'] = '/etc/ssl/certs/ca-certtificates.crt'
os.environ['REQUESTS_CA_BUNDLE']='/etc/ssl/certs/ca-certificates.crt'

os.environ['SSL_CERT_FILE']='/etc/ssl/certs/ca-certificates.crt'

os.environ['CURL_CA_BUNDLE']='/etc/ssl/certs/ca-certificates.crt'

os.environ['DEFAULT_CA_BUNDLE_PATH']='/etc/ssl/certs/ca-certificates.crt'
# export no_proxy

import ssl
import urllib.request



import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import random


seed = 2021

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_TO_GENERATE = 100

@dataclass
class ModelArgs:
    dim: int = 2304
    n_layers: int = 28
    n_heads: int = 18
    n_kv_heads: Optional[int] = None
    vocab_size: int = 96000  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-6

    max_batch_size: int = 1
    max_seq_len: int = 1024


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1 # fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wk = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
        
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
#         if mask is not None:
#             scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous()
        output = output.view(bsz, seqlen, -1)
        output = self.wo(output)
        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        our_w1 = self.w1(x)
        our_w2 = self.w3(x)
        our_w3 = self.w2(F.silu(our_w1) * our_w2)
        return our_w3


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        
        self.output = nn.Linear(
            params.dim, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        # mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).type_as(h)
        # print(mask.dtype)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)        
        output = self.output(h).float()
        return output
    
#Load weights from huggingface weights format

state_dict = torch.load('/home/hs89lee/2ndHDD/pytorch_model.bin')
for key in list(state_dict.keys()):
    if 'rotary_emb' in key:
        state_dict.pop(key)
    elif key == 'model.embed_tokens.weight':
        state_dict[key.replace('model.embed_tokens.weight', 'tok_embeddings.weight')] = state_dict.pop(key)
    elif '_attn.' in key:
        state_dict[key.replace('model.', '')\
                   .replace('self_attn.q_proj', 'attention.wq')\
                   .replace('self_attn.k_proj', 'attention.wk')\
                   .replace('self_attn.v_proj', 'attention.wv')\
                   .replace('self_attn.o_proj', 'attention.wo')] = state_dict.pop(key)
    elif 'mlp.' in key:
        state_dict[key.replace('model.', '')\
                   .replace('mlp.', 'feed_forward.')\
                   .replace('gate_proj', 'w1')\
                   .replace('up_proj', 'w3')\
                   .replace('down_proj', 'w2')] = state_dict.pop(key)
    elif 'input_layernorm' in key or 'post_attention_layernorm' in key:
        state_dict[key.replace('model.', '')\
                   .replace('input_layernorm', 'attention_norm')\
                   .replace('post_attention_layernorm', 'ffn_norm')] = state_dict.pop(key)
    elif key in ['model.norm.weight', 'lm_head.weight']:
        state_dict[key.replace('model.', '')\
                   .replace('lm_head', 'output')] = state_dict.pop(key)    
        
# # print(state_dict.keys())

params = ModelArgs()
model = Transformer(params)

## Save weights to nntrainer format

# import numpy as np

# file = open("./llama_v2.bin", "wb")

# args = ModelArgs()

# def save_llama_to_bin(params, n_layer = 32, n_head = 32, args=[]):
#     def save_weight(weight):
#         np.array(weight).tofile(file)

#     def save_embedding(weight):
#         save_weight(weight)

#     def save_attention(weights, layer_name, n_head = 32):        
#         save_weight(params[layer_name + 'attention_norm' + '.weight'])
        
#         # split_size = (args.dim // n_head)

#         # for head_idx in range(1, n_head+1):
#         #     st_idx = (args.dim - split_size * head_idx)
#         #     end_idx = st_idx + split_size
# #        save_weight(params[layer_name + 'attention.wq' + '.weight'][st_idx:end_idx, :].permute(1, 0)) # It includes multiple heads
#         save_weight(params[layer_name + 'attention.wq' + '.weight'].permute(1, 0)) # It includes multiple heads

#         # for head_idx in range(1, n_head+1):
#         #     st_idx = (args.dim - split_size * head_idx)
#         #     end_idx = st_idx + split_size
# #        save_weight(params[layer_name + 'attention.wk' + '.weight'][st_idx:end_idx, :].permute(1, 0))
#         save_weight(params[layer_name + 'attention.wk' + '.weight'].permute(1, 0)) # It includes multiple heads
        
        
#         # for head_idx in range(1, n_head+1):            
#         #     st_idx = (args.dim - split_size * head_idx)
#         #     end_idx = st_idx + split_size
# #        save_weight(params[layer_name + 'attention.wv' + '.weight'][st_idx:end_idx, :].permute(1, 0))
#         save_weight(params[layer_name + 'attention.wv' + '.weight'].permute(1, 0)) # It includes multiple heads
            
#         save_weight(params[layer_name + 'attention.wo' + '.weight'].permute(1, 0))

#     def save_feed_forward(weights, layer_name):
#         save_weight(params[layer_name + 'ffn_norm' + '.weight'])        
        
#         save_weight(params[layer_name + 'feed_forward.w3' + '.weight'].permute(1, 0))
#         save_weight(params[layer_name + 'feed_forward.w1' + '.weight'].permute(1, 0))        
#         save_weight(params[layer_name + 'feed_forward.w2' + '.weight'].permute(1, 0))

#     # save weights of embedding layer
#     save_embedding(params['tok_embeddings.weight'])
    
#     # save weights of attention layers
#     for layer_idx in range(n_layer):
#         save_attention(params, 'layers.{}.'.format(layer_idx), n_head)
#         save_feed_forward(params, 'layers.{}.'.format(layer_idx))
        
#     save_weight(params['norm.weight'])
    
#     save_weight(params['output.weight'].permute(1, 0))

model.load_state_dict(state_dict);    
# save_llama_to_bin(model.state_dict(), n_layer = params.n_layers, n_head = params.n_heads, args=args)

X = torch.tensor([[ 7571,  3701,  3794, 14982,   423,   587,  4615,   422,  7077,   706, \
           484,  7478,  1392,   656,   257,  1907,  2641,   262, 25762,   286, \
           257,  6614,  8972,   878,   340,   373,  7530,   284,  1011,   572, \
            13,   383, 38880,  7411,   262, 10654,   290,   763,    12,    79, \
         23439, 23681,   981,   262,  6614,   373,   852,  5597,   329,   257, \
          2026,    12, 11374,  7002,   422, 12517,   284, 13790,   541,   333, \
           938,  1755,    13,  5747, 14982,   423,   587,  4587,   455,  1068, \
           706,   262, 10654,   286,  5474,  9552,    21,  1157, 13832,   326, \
           262,   763,    12,    79, 23439,   550,  2984, 20709,  9586,   290, \
          7425,   683,    11,   262,  3782,   286,  3794,  2098,    13,  1052, \
          3701,  3794, 10654,  3667,   257,   763,    12,    79, 23439,  2984, \
         20709,  9586,   290,  7425,   683,  1141,   281, 38880,   287,   262, \
         25762,   357,  7753,     8,  1052,  3701,  3794,  6523,  1297,   262, \
          7533,    25,   564,   246, 10265,   262, 14982,   423,   587,  4587, \
           455,  1068,    13,  1052, 12069,   468,   587,  6149,   656,   428, \
            13,   447,   247,   383, 18091, 17424,   262, 38880,   373,  3614, \
           284,   257, 17755,  4578,    11,   290,   612,   373,   645,  3518, \
          3685,    13,   383,  3782,   286,  3794,    11, 28411,   257,  2723, \
            11,  2098,   326,   262, 10654,   373, 18513,   706,   339,  1965, \
           262,   763,    12,    79, 23439,   284,  1700,   564,   246, 34666, \
          1011,    12,  2364,  5538,   447,   247,   329,   262,  5474,    11, \
          1390,   262,  1271,   286, 10405,   319,  3096,    11,  1011,    12, \
          2364,  3463,   290,  5252,    13,  5455,   286,  3393,  6447,   262, \
          4519,   287, 12517,    11,   543,   561,   423,  2957,   284,   262, \
         25395,   286,   262,  5474,    11,   262, 10654, 13112,   262,  6614, \
           284, 13790,   541,   333,   290,   788,  7981,  3701,  3794,  3085, \
            13,  3942, 22548,  2828,   423,  5611,   281,  3645,   656,   262, \
          4519,   284,  5004,  1771,   597,   286,   262,  4671,  2950,   815, \
           307, 30654,    13, 17905,  1950,   326,   262, 10654,   373, 18513, \
           706,  4737,   262,   763,    12,    79, 23439,   284,  1700,  1321, \
           878,  1011,    12,  2364,   764,   317,   989,   416,   262,  3782, \
           286,  3794,   531,   326,   262,   763,    12,    79, 23439,   468, \
          7452,  2092, 14227,   287,   262,  1613,    13,  7683,   812,  2084, \
           339,  1297,   262, 10654,   286,   257,  5474,   284,  8420,   262, \
         25762,    11,   564,   246, 28956,   262,  5788,   319,   465, 10147, \
         19908,   447,   247,   290,  1907,   683,    11,   981,   257,  8224, \
          5717,   734,   812,  2084,   422,  1194, 10654, 11434,   262,   763, \
            12,    79, 23439,   447,   247,    82,  5110,  1535,   290,  4752, \
           339,   373,   564,   246,    81,  2507,   290,   555,  9423,  3383, \
           447,   247,    13,  4586,  1755,   447,   247,    82,  4519,  2058, \
           379,   257,  8564,   640,   329,   262,  5068, 22548,  2831,  1708, \
           262, 13574,  7411,  2679, 48819,  5474,   604,    52,  3865,  1495, \
            13, 40466,  1975,  2681,    12,  1941,    12,   727,   763,    12, \
            79, 23439, 33728, 40753,  4224, 14593, 14997,   262,  6614,   656, \
           262,  4141, 46312,   784,  5170,  2506,   319,  3096,   784,   706, \
         22656,   262, 10654,   503,   286,   262, 25762,   319,   257,  5474, \
           422, 15142,   284,   360,   385,   325,   335, 24263,    13,  2679, \
          7533, 44406,  2098,   326, 40753,  4224, 16499,   262,  5230,   329, \
          1321,   319,  7341,   290,  8862,  1262,   262,  1438,   564,   246, \
         22308,  7959,   346,   447,   247,    13,   198,  5492, 35743,   428, \
            13,   198]], dtype=torch.long)




# torch.set_printoptions(threshold=100000*30)
output = model(X, 0)
# print(torch.argmax(output, -1))
# print(output.shape)
# for i in range(len(output[0])):
#     print(output[:,i].shape)
#     print(output[:,i])

# print(torch.argmax(output, -1).shape)
# print(torch.argmax(output, -1)[-1, -1].shape)
# print(torch.argmax(output, -1)[-1, -1].dtype)

X = torch.tensor([[torch.argmax(output, -1)[-1, -1]]], dtype=torch.long)
print(X)
for i in range(NUM_TO_GENERATE):
    output = model(X, len(X) + i)
    # print(output.shape)
    # print("check shape")
    # print(torch.argmax(output, -1).shape)
    X = torch.tensor([[torch.argmax(output, -1)[-1, -1]]], dtype=torch.long)
    print(X)



# print(model(X, 0))
