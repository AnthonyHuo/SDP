import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '/home/yixiao/yixiao/sparse_diff/SDP')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from task_moe.moe import MoE
import time
from einops import rearrange

from typing import Union, Optional, Tuple, Callable, List
import logging
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
import torch.nn.functional as F
from torch import nn, Tensor
import copy
from mixture_of_experts.task_moe import TaskMoE
import math
import numpy as np
logger = logging.getLogger(__name__)



class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor,task_id, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt
        loss = 0.0
        for mod in self.layers:
            output,aux_loss,probs = mod(output,task_id, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            loss+=aux_loss

        if self.norm is not None:
            output = self.norm(output)

        return output,loss,probs
class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectivaly. Otherwise it's done after.
            Default: ``False`` (after).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, n_tasks: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.task_moe_layer = TaskMoE(
            input_size = d_model,
            head_size = dim_feedforward // 16,
            num_experts = 16,
            k = 8,
            bias=True,
            acc_aux_loss=True,
            w_MI=0.0005, #0.0005
            w_finetune_MI=0,
            task_num=n_tasks,
            activation=nn.Sequential(
                nn.GELU(),
            ),
            noisy_gating=False,
        )
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, task_id, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            # x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            output,aux_loss,probs = self._ff_block(self.norm3(x),task_id)
            x = x + output
            # aux_loss = self._ff_block(self.norm3(x),task_id)[1]
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            # x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x,task_id))

        return x,aux_loss,probs

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor,task_id) -> Tensor:
        # x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x,aux_loss,probs= self.task_moe_layer(x,task_id)
        return self.dropout3(x),aux_loss,probs


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))




class DSCBlock(nn.Module):
    def __init__(self,in_size=3, expand_size = 32, out_size=16, kernel_size=7, stride=1):
        super(DSCBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=expand_size // 4, num_channels=expand_size)
        
        self.act1 = nn.SiLU() 
        
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                                    padding=kernel_size // 2, groups=expand_size, bias=False)
        
        self.gn2 = nn.GroupNorm(num_groups=expand_size // 4, num_channels=expand_size)
        self.act2 = nn.SiLU() 

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn3 = nn.GroupNorm(num_groups=out_size // 4, num_channels=out_size)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.gn3(x)
        return x
    
    
class VisionEncoder(nn.Module):
    def __init__(self):
        super(VisionEncoder, self).__init__()
        num_experts = 8
        k = 2
        gate_input_size = 16
        expert_input_size = (3,80,80)
        expert_output_size = (16,40,40)
        module = DSCBlock(in_size=3, expand_size = 32, out_size=16, kernel_size=7, stride=2)
        task_num = 8
        self.patch_size = 4
        
        self.conv_moe1 = MoE(gate_input_size, 
                expert_input_size,
                expert_output_size,
                module, 
                num_experts, 
                k, 
                task_num=task_num,
                w_MI=0, w_H=0, w_finetune_MI=0, limit_k=0, w_topk_loss=0.0, noisy_gating=False, gating_activation=None, task_id=None)
        
        
        # self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        
        expert_input_size = (16,40,40)
        expert_output_size = (32,40,40)
        module = DSCBlock(in_size=16, expand_size = 64, out_size=32, kernel_size=3, stride=1)
        
        self.conv_moe2 = MoE(gate_input_size, 
                expert_input_size,
                expert_output_size,
                module, 
                num_experts, 
                k, 
                task_num=task_num,
                w_MI=0, w_H=0, w_finetune_MI=0, limit_k=0, w_topk_loss=0.0, noisy_gating=False, gating_activation=None, task_id=None)
        
        
        expert_input_size = (32,40,40)
        expert_output_size = (64,20,20)
        module = DSCBlock(in_size=32, expand_size = 128, out_size=64, kernel_size=3, stride=2)
        
        
        self.conv_moe3 = MoE(gate_input_size, 
                expert_input_size,
                expert_output_size,
                module, 
                num_experts, 
                k, 
                task_num=task_num,
                w_MI=0, w_H=0, w_finetune_MI=0, limit_k=0, w_topk_loss=0.0, noisy_gating=False, gating_activation=None, task_id=None)
        
        
        
        expert_input_size = (64,20,20)
        expert_output_size = (64,20,20)
        module = DSCBlock(in_size=64, expand_size = 256, out_size=64, kernel_size=3, stride=1)
        
        
        self.conv_moe4 = MoE(gate_input_size, 
                expert_input_size,
                expert_output_size,
                module, 
                num_experts, 
                k, 
                task_num=task_num,
                w_MI=0, w_H=0, w_finetune_MI=0, limit_k=0, w_topk_loss=0.0, noisy_gating=False, gating_activation=None, task_id=None)
        
        
        expert_input_size = (64,20,20)
        expert_output_size = (32,20,20)
        module = DSCBlock(in_size=64, expand_size = 128, out_size=32, kernel_size=3, stride=1)
        
        
        self.conv_moe5 = MoE(gate_input_size, 
                expert_input_size,
                expert_output_size,
                module, 
                num_experts, 
                k, 
                task_num=task_num,
                w_MI=0, w_H=0, w_finetune_MI=0, limit_k=0, w_topk_loss=0.0, noisy_gating=False, gating_activation=None, task_id=None)
        
        
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, 25, 512))


        # self.linear = nn.Linear(512, 128)

        
        # n_emb = 128
        # n_head = 8
        
        
        # decoder_layer = TransformerDecoderLayer(
        #     d_model=n_emb,
        #     nhead=n_head,
        #     n_tasks=task_num,
        #     dim_feedforward=4*n_emb,
        #     dropout=0.1,
        #     activation='gelu',
        #     batch_first=True,
        #     norm_first=True # important for stability
        # )
        # self.decoder = TransformerDecoder(
        #     decoder_layer=decoder_layer,
        #     num_layers=4
        # )
        
        
        

    def forward(self, gate_input, expert_input, task_bh):
        
        
        out1, loss1 = self.conv_moe1(gate_input, expert_input, task_bh)
        out2, loss2 = self.conv_moe2(gate_input, out1, task_bh)
        out3, loss3 = self.conv_moe3(gate_input, out2, task_bh)
        out4, loss4 = self.conv_moe4(gate_input, out3, task_bh)
        out5, loss5 = self.conv_moe5(gate_input, out4, task_bh)
                
        out = out5
        patch_size = self.patch_size
        patches = out.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = torch.permute(patches,(0,2,3,4,5,1))
        patches = rearrange(patches, "b p1 p2 h w c -> b (p1 p2) (h w c)")
        
        patches = patches + self.patch_pos_embed[:,:,:]
                
        # patches = self.linear(patches)
        
        # patches, loss3, _ = self.decoder(
        #         tgt = patches,
        #         task_id = task_bh,
        #         memory = patches,)
        
        
                
        loss = loss1 + loss2 + loss3 + loss4 + loss5
        
        return patches, loss
    

class RobotEncoder(nn.Module):
    def __init__(self, robot_state_size):
        super(RobotEncoder, self).__init__()
        self.n_emb = 512
        num_experts = 8
        k = 2
        gate_input_size = 16
        expert_input_size = (robot_state_size,)
        expert_output_size = (self.n_emb,)
        
        module = FourierEmbedding(input_dim = robot_state_size, 
                                  hidden_dim = self.n_emb, 
                                  num_freq_bands = 16)
        
        
        task_num = 8
        self.state_moe = MoE(gate_input_size, 
                expert_input_size,
                expert_output_size,
                module, 
                num_experts, 
                k, 
                task_num=task_num,
                w_MI=0, w_H=0, w_finetune_MI=0, limit_k=0, w_topk_loss=0.0, noisy_gating=False, gating_activation=None, task_id=None)
        
        
    def forward(self, gate_input, expert_input, task_bh):
        out, loss = self.state_moe(gate_input, expert_input, task_bh)
        return out, loss





class StateEncoder(nn.Module):
    def __init__(self):
        super(StateEncoder, self).__init__()
        
        self.time_steps = 2
        
        self.state_keys = ['agentview_image', 
                      'robot0_eye_in_hand_image',
                      'robot0_eef_pos',
                      'robot0_eef_quat',
                      'robot0_gripper_qpos']
        
        self.obs_keys = ['agentview_image', 
                      'robot0_eye_in_hand_image']
        
        self.robot_keys = ['robot0_eef_pos',
                      'robot0_eef_quat',
                      'robot0_gripper_qpos']
        
        self.encoders = nn.ModuleDict([
            ['agentview_image', VisionEncoder()],
            ['robot0_eye_in_hand_image', VisionEncoder()],
            ['robot0_eef_pos', RobotEncoder(robot_state_size=3)],
            ['robot0_eef_quat', RobotEncoder(robot_state_size=4)],
            ['robot0_gripper_qpos', RobotEncoder(robot_state_size=2)],
        ])
        
        self.n_emb = 512
        self.pos_emb = nn.Parameter(torch.zeros(1, self.time_steps, self.n_emb))
        
        
        n_emb = 512
        n_head = 8
        task_num = 8
        
        decoder_layer = TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            n_tasks=task_num,
            dim_feedforward=4*n_emb,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True # important for stability
        )
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=6
        )
        
        self.gate_emb = nn.Parameter(torch.zeros(task_num, 16))
        
        # agentview_image torch.Size([128, 3, 84, 84]) after [3,80,80]
        # robot0_eye_in_hand_image torch.Size([128, 3, 84, 84])
        # robot0_eef_pos torch.Size([128, 3])
        # robot0_eef_quat torch.Size([128, 4])
        # robot0_gripper_qpos torch.Size([128, 2])
        
    def forward(self, expert_input_dic, task_bh):        
        bs = expert_input_dic['agentview_image'].size(0)
        gate_input = self.gate_emb[task_bh:task_bh+1].repeat(bs, 1)
        loss = 0
        tmp_out = []
        time_index = [[] for _ in range(self.time_steps)]
        count = 0
        for key in self.encoders:
            out, tmp_loss = self.encoders[key](gate_input, expert_input_dic[key], task_bh)
            if out.dim() == 2:
                num_per_time_step = 1
            elif out.dim() == 3:
                num_per_time_step = out.size(1)
            else:
                raise Exception('State Encoder: out should be in 2 or 3 dimensions.')
            
            for i in range(self.time_steps):
                time_index[i].extend(list(np.arange(count+i*num_per_time_step, count+(i+1)*num_per_time_step)))
            count += self.time_steps * num_per_time_step
                                
            loss += tmp_loss
            
            out = out.reshape(bs // self.time_steps, self.time_steps, *out.size()[1:])
            if out.dim() == 3:
                out = out + self.pos_emb
            elif out.dim() == 4:
                out = out + self.pos_emb[:,:,None,:]
                out = out.reshape(bs // self.time_steps, -1, self.n_emb)

            tmp_out.append(out)
            
        tmp_out = torch.cat(tmp_out, dim=1)

        out, tmp_loss, _ = self.decoder(
                tgt = tmp_out,
                task_id = task_bh,
                memory = tmp_out,)
        
        loss += tmp_loss
        
        out_list = []
        for i in range(self.time_steps):
            out_list.append(torch.mean(tmp_out[:,time_index[i],:], dim=1, keepdim=True))
        
        out = torch.cat(out_list, dim=1)
                                
        return out, loss
        
        


class FourierEmbedding(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_freq_bands: int) -> None:
        super(FourierEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.freqs = nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None
        self.mlps = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
                for _ in range(input_dim)])
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self,
                continuous_inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
        x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
        continuous_embs: List[Optional[torch.Tensor]] = [None] * self.input_dim
        if continuous_inputs.dim() == 2:
            for i in range(self.input_dim):
                continuous_embs[i] = self.mlps[i](x[:, i])
        elif continuous_inputs.dim() == 3:
            for i in range(self.input_dim):
                continuous_embs[i] = self.mlps[i](x[:,:, i])
        else:
            raise Exception("Fourier embedding error: the input must have 2 or 3 dimensions.")
        x = torch.stack(continuous_embs).sum(dim=0)
        return self.to_out(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':

    state_encoder = StateEncoder()
    
    # agentview_image torch.Size([128, 3, 84, 84]) after [3,80,80]
    # robot0_eye_in_hand_image torch.Size([128, 3, 84, 84])
    # robot0_eef_pos torch.Size([128, 3])
    # robot0_eef_quat torch.Size([128, 4])
    # robot0_gripper_qpos torch.Size([128, 2])
    
    bs = 128
    input_dic = {}
    gate_input = torch.randn(size=(bs, 16))
    input_dic['agentview_image'] = torch.zeros(bs, 3, 80, 80)
    input_dic['robot0_eye_in_hand_image'] = torch.zeros(bs, 3, 80, 80)
    input_dic['robot0_eef_pos'] = torch.zeros(bs, 3)
    input_dic['robot0_eef_quat'] = torch.zeros(bs, 4)
    input_dic['robot0_gripper_qpos'] = torch.zeros(bs, 2)
    
    task_bh = torch.tensor([7])
    
    ps = time.time()
    out, loss = state_encoder(input_dic, task_bh)
    print(time.time()-ps)
    
    print(loss)
    
    # Get the number of parameters
    num_params = count_parameters(state_encoder.encoders['robot0_eef_quat'])
    print(f'The robot0_eef has {num_params/1000000} trainable parameters')
    
    num_params = count_parameters(state_encoder.encoders['agentview_image'])
    print(f'The agentview_image has {num_params/1000000} trainable parameters')
    
    num_params = count_parameters(state_encoder)
    print(f'The state_encoder has {num_params/1000000} trainable parameters')
    
    # for key in out_dic.keys():
    #     print(key, out_dic[key].size())
    
    print(out.size())