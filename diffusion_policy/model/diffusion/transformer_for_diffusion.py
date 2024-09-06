from typing import Union, Optional, Tuple, Callable
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
import torch.nn.functional as F
from torch import nn, Tensor
import copy
from mixture_of_experts.task_moe import TaskMoE
from task_moe.moe import MoE
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
        # Implementation of Feedforward model
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
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            output,aux_loss,probs = self._ff_block(self.norm3(x),task_id)
            x = x + output
            # aux_loss = self._ff_block(self.norm3(x),task_id)[1]
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
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


class TransformerForDiffusion(ModuleAttrMixin):
    def __init__(self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            n_obs_steps: int = None,
            cond_dim: int = 0,
            n_tasks: int = 1,
            n_layer: int = 12,
            n_head: int = 12,
            n_emb: int = 768,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal_attn: bool=False,
            time_as_cond: bool=True,
            obs_as_cond: bool=False,
            n_cond_layers: int = 0
        ) -> None:
        super().__init__()

        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon
        
        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps

        self.n_emb = n_emb
        self.output_dim = output_dim
        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None
        # print('_'*20)
        # print(cond_dim, n_emb)
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4*n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
                # print('_'*20)
            else:
                # print('*'*20)
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
                
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Get the number of parameters
            num_params = count_parameters(self.encoder)
            print(f'The encoder has {num_params/1000000} trainable parameters')
            
            # decoder
            decoder_layer = TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                n_tasks=n_tasks,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True # important for stability
            )
            self.decoder = TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )
            
            # Get the number of parameters
            num_params = count_parameters(self.decoder)
            print(f'The decoder has {num_params/1000000} trainable parameters')
            
            
            ## encoder 2M decoder 158M
            ## decoder 512 latent size; 512*4 for mlp size.
            
        else:
            # encoder only BERT
            encoder_only = True

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)
            
            if time_as_cond and obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(S),
                    indexing='ij'
                )
                mask = t >= (s-1) # add one dimension since time is the first token in cond
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        
        
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
        

        # self.encoder = None
        # self.cond_obs_emb = None
        # self.action_head = MoE(
        #     gate_input_size = n_emb, 
        #     expert_input_size = (n_emb,),
        #     expert_output_size = (output_dim,),
        #     module = nn.Sequential(nn.LayerNorm(n_emb), nn.Linear(n_emb, output_dim)), 
        #     num_experts = 8, 
        #     k = 2, 
        #     w_MI=0.0005, #0.0005
        #     w_finetune_MI=0,
        #     fixed_task_num=0,
        #     noisy_gating=False,
        # )

        
        
            
        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            SinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            # raise RuntimeError("Unaccounted module {}".format(module))
            pass
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith("experts.weight"):
                    no_decay.add(fpn)
        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups


    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        cond: Optional[torch.Tensor]=None,
        task_id: Union[torch.Tensor, float, int]=None, 
        **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        task_id: int
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)

        # process input
        input_emb = self.input_emb(sample)

        if self.encoder_only:
            # BERT
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T+1,n_emb)
            x = self.encoder(src=x, mask=self.mask)
            # (B,T+1,n_emb)
            x = x[:,1:,:]
            # (B,T,n_emb)
        else:
            # encoder
            cond_embeddings = time_emb
            if self.obs_as_cond:
                if self.cond_obs_emb is None:
                    cond_obs_emb = cond
                else:
                    cond_obs_emb = self.cond_obs_emb(cond)
                # cond_obs_emb = cond
                # (B,To,n_emb)
                # print(cond_embeddings.size(), cond_obs_emb.size())
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[
                :, :tc, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(cond_embeddings + position_embeddings)
            if self.encoder is not None:
                x = self.encoder(x)
            memory = x
            # (B,T_cond,n_emb)
            
            # decoder
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T,n_emb)
            x,loss,probs = self.decoder(
                tgt=x,
                task_id=task_id,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask,
            )
            # (B,T,n_emb)
        
        # batch_size, time_steps, x_emb_size = x.size()
        # x = x.reshape(-1, x_emb_size)
        # x, head_mi_loss = self.action_head(x,x,task_id)
        # x = x.reshape(batch_size, time_steps, self.output_dim)
        # loss = loss + head_mi_loss

        x = self.ln_f(x)
        x = self.head(x)

        # (B,T,n_out)
        return x,loss,probs


def test():
    # GPT with time embedding
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        # cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)
    

    # GPT with time embedding and obs cond
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)

    # GPT with time embedding and obs cond and encoder
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)

    # BERT with time embedding token
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        # cond_dim=10,
        # causal_attn=True,
        time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)

