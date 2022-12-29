#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
#
# @file transLayer_v2.py
# @date 19 October 2021
# @brief Rewrite parameters in the order consistent with nntrainer for the torch model
# @author Jihoon lee <jhoon.it.lee@samsung.com>

import torch
from collections.abc import Iterable
from zoneout import Zoneout

optimize = True

__all__ = ["params_translated"]

# type to parameter mapper containing (classes, function)
handler_book = []

##
# Decorater to register class mapping to a function.
# This is to imitate function overloadding
def register_for_(classes):
    for already_registered_classes, _ in handler_book:
        if not isinstance(classes, Iterable):
            classes = (classes, )

        for cls_ in classes:
            if isinstance(cls_, already_registered_classes):
                raise ValueError("class is already registered %s" % cls_.__name__)

    def wrapper(func):
        handler_book.append((classes, func))
        return func

    return wrapper


def default_translate_(model):
    yield from model.named_parameters(recurse=False)

@register_for_(torch.nn.Linear)
def fc_translate(model):
    params = [(name, tensor.detach()) for name, tensor in model.named_parameters()]
    def transpose_(weight):
        return (weight[0], weight[1].transpose(1, 0))
    if len(params) == 2:
        new_params = [transpose_(params[0]), params[1]]
    else:
        new_params = [transpose_(params[0])]
    yield from new_params

@register_for_(torch.nn.BatchNorm1d)
def bn1d_translate(model):
    gamma, beta = [(name, tensor.detach()) for name, tensor in model.named_parameters()]
    mu, var, _ = [(name, tensor.detach()) for name, tensor in model.named_buffers()]
    yield from [mu, var, gamma, beta]

@register_for_(torch.nn.LayerNorm)
def layer_normalization_translate(model):
    gamma, beta = [(name, tensor.detach()) for name, tensor in model.named_parameters()]
    yield from [gamma, beta]

@register_for_((Zoneout))
def zoneout_translate(model):
    params = [(name, tensor.detach()) for name, tensor in model.named_parameters()]
    hidden_state = ("hidden_state", torch.stack(model.hidden_state_zoneout_mask, dim=0))
    cell_state = ("cell_state", torch.stack(model.cell_state_zoneout_mask, dim=0))

    # [hidden, input] -> [input, hidden]
    def transpose_(weight):
        return (weight[0], weight[1].transpose(1, 0))

    new_params = [transpose_(params[0]), transpose_(params[1]), params[2], params[3], hidden_state, cell_state]
    yield from new_params

@register_for_((torch.nn.LSTM))
def lstm_translate(model):
    params = [(name, tensor.detach()) for name, tensor in model.named_parameters()]
    # [hidden, input] -> [input, hidden]
    def transpose_(weight):
        return (weight[0], weight[1].transpose(1, 0))

    new_params = [transpose_(params[0]), transpose_(params[1]), params[2], params[3]]
    if model.bidirectional:
        reverse_params = [transpose_(params[4]), transpose_(params[5]), params[6], params[7]]
        new_params += reverse_params

    yield from new_params

@register_for_((torch.nn.RNNCell, torch.nn.LSTMCell))
def rnn_lstm_translate(model):
    params = [(name, tensor.detach()) for name, tensor in model.named_parameters()]
    # [hidden, input] -> [input, hidden]
    def transpose_(weight):
        return (weight[0], weight[1].transpose(1, 0))

    new_params = [transpose_(params[0]), transpose_(params[1]), params[2], params[3]]
    yield from new_params

@register_for_((torch.nn.GRUCell))
def gru_translate(model):
    params = [(name, tensor.detach()) for name, tensor in model.named_parameters()]

    # [hidden, input] -> [input, hidden]
    def transpose_(weight):
        return (weight[0], weight[1].transpose(1, 0))

    # resetgate, inputgate, newgate -> inputgate, resetgate, newgate
    def reorder_weights(params):
        reordered_weights = []
        for (name, weight) in params: # param = ("name", weight)
            weight = weight.hsplit(3)
            reordered_weights.append((name, torch.hstack((weight[1], weight[0], weight[2])))) # reorder
        return reordered_weights

    transposed_params = [transpose_(params[0]), transpose_(params[1]), params[2], params[3]]
    new_params = reorder_weights(transposed_params)

    yield from new_params

if optimize:
    @register_for_((torch.nn.MultiheadAttention))
    def multi_head_attention_translate(model):
        def transpose_(weight):
            return (weight[0], weight[1].transpose(1, 0))

        params = [(name, tensor.detach()) for name, tensor in model.named_parameters()]

        getParamByName = lambda name: list(filter(lambda param: param[0] == name, params))[0]

        if model._qkv_same_embed_dim:
            in_proj_weight = getParamByName('in_proj_weight')
            w_q, w_k, w_v = in_proj_weight[1].chunk(3)
            q_proj_weight = ('q_proj_weight', w_q)
            k_proj_weight = ('k_proj_weight', w_k)
            v_proj_weight = ('v_proj_weight', w_v)
        else:
            q_proj_weight = getParamByName('q_proj_weight')
            k_proj_weight = getParamByName('k_proj_weight')
            v_proj_weight = getParamByName('v_proj_weight')

        q_p_w = q_proj_weight[1]
        k_p_w = k_proj_weight[1]
        v_p_w = v_proj_weight[1]
        *q_p_ws, = q_p_w.chunk(model.num_heads)
        *k_p_ws, = k_p_w.chunk(model.num_heads)
        *v_p_ws, = v_p_w.chunk(model.num_heads)
        q_proj_weights = [('q_proj_weight', q_p_w) for q_p_w in q_p_ws]
        k_proj_weights = [('k_proj_weight', k_p_w) for k_p_w in k_p_ws]
        v_proj_weights = [('v_proj_weight', v_p_w) for v_p_w in v_p_ws]

        if model.in_proj_bias is not None:
            in_proj_bias = getParamByName('in_proj_bias')
            w_q, w_k, w_v = in_proj_bias[1].chunk(3)
            q_proj_bias = ('q_proj_bias', w_q)
            k_proj_bias = ('k_proj_bias', w_k)
            v_proj_bias = ('v_proj_bias', w_v)
            q_p_b = q_proj_bias[1]
            k_p_b = k_proj_bias[1]
            v_p_b = v_proj_bias[1]
            *q_p_bs, = q_p_b.chunk(model.num_heads)
            *k_p_bs, = k_p_b.chunk(model.num_heads)
            *v_p_bs, = v_p_b.chunk(model.num_heads)
            q_proj_biases = [('q_proj_bias', q_b) for q_b in q_p_bs]
            k_proj_biases = [('k_proj_bias', k_b) for k_b in k_p_bs]
            v_proj_biases = [('v_proj_bias', v_b) for v_b in v_p_bs]

        out_proj_weight = getParamByName('out_proj.weight')

        if model.in_proj_bias is not None:
            out_proj_bias = getParamByName('out_proj.bias')

        if model.in_proj_bias is None:
            new_params = []
            new_params += [transpose_(q) for q in q_proj_weights]
            new_params += [transpose_(k) for k in k_proj_weights]
            new_params += [transpose_(v) for v in v_proj_weights]
            new_params += [transpose_(out_proj_weight)]
        else:
            new_params = []
            for i in range(len(q_proj_weights)):
                new_params += [transpose_(q_proj_weights[i]), q_proj_biases[i]]
            for i in range(len(k_proj_weights)):
                new_params += [transpose_(k_proj_weights[i]), k_proj_biases[i]]
            for i in range(len(v_proj_weights)):
                new_params += [transpose_(v_proj_weights[i]), v_proj_biases[i]]
            new_params += [transpose_(out_proj_weight), out_proj_bias]

        yield from new_params

else :
    @register_for_((torch.nn.MultiheadAttention))
    def multi_head_attention_translate(model):
        def transpose_(weight):
            return (weight[0], weight[1].transpose(1, 0))

        params = [(name, tensor.detach()) for name, tensor in model.named_parameters()]

        getParamByName = lambda name: list(filter(lambda param: param[0] == name, params))[0]

        if model._qkv_same_embed_dim:
            in_proj_weight = getParamByName('in_proj_weight')
            w_q, w_k, w_v = in_proj_weight[1].chunk(3)
            q_proj_weight = ('q_proj_weight', w_q)
            k_proj_weight = ('k_proj_weight', w_k)
            v_proj_weight = ('v_proj_weight', w_v)
        else:
            q_proj_weight = getParamByName('q_proj_weight')
            k_proj_weight = getParamByName('k_proj_weight')
            v_proj_weight = getParamByName('v_proj_weight')

        if model.in_proj_bias is not None:
            in_proj_bias = getParamByName('in_proj_bias')
            w_q, w_k, w_v = in_proj_bias[1].chunk(3)
            q_proj_bias = ('q_proj_bias', w_q)
            k_proj_bias = ('k_proj_bias', w_k)
            v_proj_bias = ('v_proj_bias', w_v)

        out_proj_weight = getParamByName('out_proj.weight')

        if model.in_proj_bias is not None:
            out_proj_bias = getParamByName('out_proj.bias')

        if model.in_proj_bias is None:
            new_params = [transpose_(q_proj_weight), transpose_(k_proj_weight), transpose_(v_proj_weight), transpose_(out_proj_weight)]
        else:
            new_params = [transpose_(q_proj_weight), q_proj_bias, transpose_(k_proj_weight), k_proj_bias, transpose_(v_proj_weight), v_proj_bias, transpose_(out_proj_weight), out_proj_bias]

        yield from new_params

@register_for_(torch.nn.TransformerEncoderLayer)
def transformer_encoder_translate(model):
    self_attn, linear1, dropout1, linear2, norm1, norm2, dropout2, dropout3 = [child for name, child in model.named_children()]
    modules = [self_attn, norm1, linear1, linear2, norm2]
    ret = []

    for module in modules:
        for registered_classes, fn in handler_book:
            if isinstance(module, registered_classes):
                module = fn(module)
                module = list((n, t) for n, t in module)
                ret += module
                break
    yield from ret

@register_for_(torch.nn.TransformerDecoderLayer)
def transformer_decoder_translate(model):
    self_attn, multihead_attn, linear1, dropout1, linear2, norm1, norm2, norm3, dropout2, dropout3, dropout4 = [(name, child) for name, child in model.named_children()]
    modules = [self_attn, norm1, multihead_attn, norm2, linear1, linear2, norm3]
    ret = []

    multihead_attn_kv = []
    if optimize:
        for module in modules:
            for registered_classes, fn in handler_book:
                if isinstance(module[1], registered_classes):
                    multihead_attn = True if module[0] == "multihead_attn" else False
                    if multihead_attn:
                        in_proj_bias = True if module[1].in_proj_bias is not None else False
                        nheads = module[1].num_heads
                    module = fn(module[1])
                    module = list((n, t) for n, t in module)

                    if multihead_attn:
                        if in_proj_bias:
                            multihead_attn_kv = module[2 * nheads:3 * 2 * nheads]
                            ret += module[:2 * nheads] + module[3 * 2 * nheads:]
                        else:
                            multihead_attn_kv = module[nheads:nheads:3 * nheads]
                            ret += module[:nheads] + module[3 * nheads:]
                    else:
                        ret += module
                    break
        ret = multihead_attn_kv + ret
    else:
        for module in modules:
            for registered_classes, fn in handler_book:
                if isinstance(module[1], registered_classes):
                    module = fn(module[1])
                    module = list((n, t) for n, t in module)
                    ret += module
                    break

    yield from ret

@register_for_(torch.nn.Transformer)
def transformer_translate(model):
    encoder, decoder = [child for name, child in model.named_children()]

    encoder_layers, encoder_norm = [child for name, child in encoder.named_children()]
    encoder_layers = [child for name, child in encoder_layers.named_children()]

    encoder_list = encoder_layers
    encoder_list.append(encoder_norm)

    decoder_layers, decoder_norm = [child for name, child in decoder.named_children()]
    decoder_layers = [child for name, child in decoder_layers.named_children()]

    decoder_list = decoder_layers
    decoder_list.append(decoder_norm)

    encoder_ret_list = []
    for module in encoder_list:
        for registered_classes, fn in handler_book:
            if isinstance(module, registered_classes):
                module = fn(module)
                module = list((n, t) for n, t in module)
                encoder_ret_list += module
                break

    decoder_ret_list = []
    mha_kv_list = []

    if optimize:
        for module in decoder_list:
            for registered_classes, fn in handler_book:
                if isinstance(module, registered_classes):
                    # print(module)
                    is_decoder_layer = True if isinstance(module, torch.nn.modules.transformer.TransformerDecoderLayer) else False
                    if is_decoder_layer:
                        # in_proj_bias = True if module[1].in_proj_bias else False
                        in_proj_bias = True
                        
                    module = fn(module)
                    module = list((n, t) for n, t in module)
                    # for tensor in module:
                    #     print(tensor[1].shape)

                    if is_decoder_layer:
                        if in_proj_bias:
                            mha_kv_list += module[:2 * 2 * model.nhead]
                            decoder_ret_list += module[2 * 2 * model.nhead:]
                        else:
                            mha_kv_list += module[:2 * model.nhead]
                            decoder_ret_list += module[2 * model.nhead:]
                    else:
                        decoder_ret_list += module
                    break
        decoder_ret_list = mha_kv_list + decoder_ret_list
    else:
        for module in decoder_list:
            for registered_classes, fn in handler_book:
                if isinstance(module, registered_classes):
                    module = fn(module)
                    module = list((n, t) for n, t in module)
                    decoder_ret_list += module
                    break

    # print("encoder")
    # for tensor in encoder_ret_list:
    #     print(tensor[1].shape)
    # print("decoder")
    # for tensor in decoder_ret_list:
    #     print(tensor[1].shape)

    yield from encoder_ret_list + decoder_ret_list

def translate(model):
    for child in model.children():
        for registered_classes, fn in handler_book:
            if isinstance(child, registered_classes):
                yield from fn(child)
                break
        else: # default case
            yield from translate(child)
    yield from default_translate_(model)

params_translated = translate

