import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
def clone_module(module, memo=None):
    module= deepcopy(module)
    return module
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

    **Description**

    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().

    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.

    **Arguments**

    * **module** (Module) - Module to be cloned.

    **Return**

    * (Module) - The cloned module.

    **Example**

    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """
    # NOTE: This function might break in future versions of PyTorch.

    # TODO: This function might require that module.forward()
    #       was called in order to work properly, if forward() instanciates
    #       new variables.
    # TODO: We can probably get away with a shallowcopy.
    #       However, since shallow copy does not recurse, we need to write a
    #       recursive version of shallow copy.
    # NOTE: This can probably be implemented more cleanly with
    #       clone = recursive_shallow_copy(model)
    #       clone._apply(lambda t: t.clone())

    if memo is None:
        # Maps original data_ptr to the cloned tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[param_ptr] = cloned

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone

class ParallelExperts(nn.Module):
    def __init__(self, num_experts, expert_module:nn.Module) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([clone_module(expert_module) for _ in range(num_experts)])
    
    def forward(self, inputs, expert_size):
        experts = self.experts
        expert_size = expert_size.tolist()

        input_list = inputs.split(expert_size, dim=0)
        output_list = []
        for i, expert in enumerate(experts):
            output_list.append(expert(input_list[i]))
        
        return torch.cat(output_list, dim=0)
    

@torch.jit.script
def compute_gating(k: int, probs: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
    zeros = torch.zeros_like(probs)
    gates = zeros.scatter(1, top_k_indices, top_k_gates)
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    nonzeros = top_k_gates.nonzero().squeeze(-1)
    top_k_experts_nonzero = top_k_experts[nonzeros]
    _, _index_sorted_experts = top_k_experts_nonzero.sort(0)
    expert_size = (gates > 0).long().sum(0)
    index_sorted_experts = nonzeros[_index_sorted_experts]
    batch_index = index_sorted_experts.div(k, rounding_mode='trunc')
    batch_gates = top_k_gates[index_sorted_experts]
    return batch_gates, batch_index, expert_size, gates, index_sorted_experts



class MoE(nn.Module):
    def __init__(self, gate_input_size, expert_input_size, expert_output_size, module:nn.Module , num_experts, k, w_MI=0, w_H=0, w_finetune_MI=0, limit_k=0, w_topk_loss=0.0, task_num=9, noisy_gating=True, gating_activation=None, task_id=None,**kwargs):
        super(MoE, self).__init__()
        self.debug = True
        
        self.task_num = task_num
        self.w_topk_loss = w_topk_loss
        self.w_MI = w_MI
        self.w_H = w_H
        self.w_finetune_MI = w_finetune_MI
        self.gate_input_size = gate_input_size
        self.expert_input_size = expert_input_size
        self.expert_output_size = expert_output_size
        self.task_id = task_id
        self.use_conv_router = False
        self.input_size_len = 1
        self.limit_k = max(k, limit_k)

        self.experts = ParallelExperts(num_experts, module)

        
        self.num_experts = num_experts
        self.bias = True
        self.k = min(k, self.num_experts)
        self.activation = kwargs.get('activation', None)
        self.noisy_gating = noisy_gating
        
        
        if gating_activation is None:
            gating_activation = nn.GELU()

        
        if not self.use_conv_router:
            self.f_gate = nn.ModuleList([nn.Sequential(
                                        nn.Linear(self.gate_input_size, self.gate_input_size//4),
                                        gating_activation,
                                        nn.Linear(self.gate_input_size//4,
                                                2 * (self.num_experts) if noisy_gating else (self.num_experts),
                                                bias=True),
                                    
                                    ) for _ in range(self.task_num)])
        else:
            self.f_gate = nn.ModuleList([nn.Sequential(
                                        nn.Conv2d(self.gate_input_size[0], self.gate_input_size[0], 1),
                                        gating_activation,
                                        nn.Conv2d(self.gate_input_size[0],
                                                2 * (self.num_experts) if noisy_gating else self.num_experts,
                                                kernel_size=(self.gate_input_size[1], self.gate_input_size[2]),
                                                bias=True)) for _ in range(self.task_num)])

        
        for i in range(self.task_num):
            nn.init.zeros_(self.f_gate[i][-1].weight)

        self.register_buffer('PTE', torch.zeros(1, self.num_experts))
        self.register_buffer('PE', torch.zeros(self.num_experts))
        self.momentum = 0.0
        self.register_buffer('times',torch.zeros(1))

        self.task_gate_freq = [0] * 1
        self.topk_acc_probs = [0] * 1
        self.token_probs = [0] * 1

    def get_MIloss(self, logits, probs, gates, task_bh):

        if not self.training:
            return 0.0

        top_k_gates, _ = probs.topk(self.k, dim=1)
        self.token_probs[task_bh] = self.token_probs[task_bh] * 0.95 + top_k_gates.mean(0).detach()*0.05

        self.task_gate_freq[task_bh] = self.task_gate_freq[task_bh]*0.95 + ((gates > 0).float().sum(0)).detach()*0.05

        self.topk_acc_probs[task_bh] = self.topk_acc_probs[task_bh]*0.95 + (probs.mean(0)).detach()*0.05
        
        PT = 1 / 1 # since we want each task to have equal weight

        # probs = P(E|T) in this batch
        # P(T,E) = P(E|T) * P(T) 
        self.PTE[task_bh] = self.PTE[task_bh] * self.momentum + (1-self.momentum) * (probs.mean(0).detach() * PT)


        # entropy loss
        # loss = 0.
        loss = -self.w_H * (probs * torch.log(probs + 0.0001)).sum(1).mean() # maximize the entropy

        # print('times: ', self.times[0])
        if self.times[0] < 100:
            self.times[0] = self.times[0] + 1
            self.momentum = 1 - 1/(self.times[0]) 
            return loss
        else:
            self.momentum = 0.99

        # P(E) = \sum_T (P(E,T))
        PE = self.PTE.sum(0).detach()

        # P(E,T) in this batch
        MI_task_gate = torch.zeros(1, self.num_experts).cuda()
        MI_task_gate[task_bh] = MI_task_gate[task_bh] + probs.mean(0) * PT

        # P(E) in this batch
        P_EI = probs.mean(0) * PT

        # get the MI loss
        # MI_loss = -((MI_task_gate * (1 + torch.log(self.PTE.detach() + 0.0001)) ).sum() - P_EI * (1 + torch.log(PE + 0.0001))).sum()
        MI_loss = -((MI_task_gate * (1 + torch.log(self.PTE.detach() + 0.0001)) ).sum() - (P_EI * (1 + torch.log(PE + 0.0001))).sum())

        finetune_MI_loss = -((MI_task_gate * (1 + torch.log(self.PTE.detach() + 0.0001)) ).sum())

        loss = loss + self.w_MI * MI_loss + self.w_finetune_MI * finetune_MI_loss 
        
        return loss


    def get_topk_loss_and_clear(self):
        top_k_probs, top_k_indices = self.topk_acc_probs.topk(self.limit_k, dim=0)
        zeros = torch.zeros_like(self.topk_acc_probs)
        gates = zeros.scatter(0, top_k_indices, top_k_probs)
        topk_loss = ((self.topk_acc_probs - gates) * (self.topk_acc_probs - gates)).sum()

        self.topk_acc_probs = 0.
        return topk_loss * self.w_topk_loss # 0.004 * 12 * 2 = 0.09

    def top_k_gating(self, x, task_bh, skip_mask=None, sample_topk=0, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = self.f_gate[task_bh](x).squeeze()

        # if self.noisy_gating and self.training:
        if self.noisy_gating:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_stddev
            logits = noisy_logits
        elif self.noisy_gating:
            logits, _ = clean_logits.chunk(2, dim=-1)
        else:
            logits = clean_logits

        probs = torch.softmax(logits, dim=1) + 1e-4
        
        if skip_mask is not None:
            probs = torch.masked_fill(probs, skip_mask, 0)

        if self.training and (sample_topk > 0):
            # top_k_indices = torch.multinomial(probs + 1e-6, self.k)
            # top_k_gates = torch.gather(probs, 1, top_k_indices)
            assert sample_topk <= self.k

            _, top_km1_indices = probs.topk(self.k - sample_topk, dim=1)
            masked_probs = probs + 1e-6
            masked_probs[torch.arange(probs.size(0)).unsqueeze(
                1), top_km1_indices] = 0
            k_indices = torch.multinomial(masked_probs, sample_topk)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.k, dim=1)

       # top_k_indecis: [batch, K]
       
        
        top_k_gates = top_k_gates

        

        batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
            compute_gating(self.k, probs, top_k_gates, top_k_indices)
        self.expert_size = expert_size
        # print('here: ', expert_size)
        # # print('probs: ', probs)
        # # print('x: ', x)
        # exit()
        self.index_sorted_experts = index_sorted_experts
        self.batch_index = batch_index
        self.batch_gates = batch_gates

        

        return self.get_MIloss(logits, probs, gates, task_bh)

    def forward(self, gate_input, expert_input, task_bh=None, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        if gate_input.dim() == 2:
            gate_input = gate_input.unsqueeze(1)
        if expert_input.dim() == 2:
            expert_input = expert_input.unsqueeze(1)
        if not ((gate_input.size(0) == expert_input.size(0)) & (gate_input.size(1) == expert_input.size(1))):
            raise Exception('the first two dimension of gate input and expert input should be the same.')
        
        
        if self.debug:
            expert_input_dim = len(self.expert_input_size)
            if tuple(expert_input.size()[-expert_input_dim:]) != self.expert_input_size:
                raise Exception('expert_input_size does not match.')
        

            
        bsz, length, emb_size = gate_input.size()
        gate_input = gate_input.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
            
        loss= self.top_k_gating(gate_input, task_bh, skip_mask,  sample_topk=sample_topk)
        
        
        bsz, length = expert_input.size()[:2]
        expert_input = expert_input.reshape(-1, *self.expert_input_size)
        
        expert_outputs = self.experts(expert_input[self.batch_index], self.expert_size)
        
        if self.debug:
            expert_output_dim = len(self.expert_output_size)
            if tuple(expert_outputs.size()[-expert_output_dim:]) != self.expert_output_size:
                raise Exception('expert_input_size does not match.')
        
        
        if multiply_by_gates:
            gates = self.batch_gates
            for _ in range(len(self.expert_input_size)):
                gates = gates.unsqueeze(-1)
            expert_outputs = expert_outputs * gates



        zeros = torch.zeros((bsz * length,) + self.expert_output_size, 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        # print(self.batch_index, expert_outputs.size())
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, *self.expert_output_size)
        
        return y, loss


if __name__ == '__main__':
    batch_size = 64
    sequence_length = 10
    gate_input_size = 16
    expert_input_size = (128,)
    expert_output_size = (8,)


    module = nn.Sequential(nn.LayerNorm(expert_input_size), nn.Linear(expert_input_size[0], expert_output_size[0]))
    num_experts = 8
    k = 2
    model = MoE(gate_input_size, 
                expert_input_size,
                expert_output_size,
                module, 
                num_experts, 
                k, 
                w_MI=0, w_H=0, w_finetune_MI=0, limit_k=0, w_topk_loss=0.0, task_num=8, noisy_gating=False, gating_activation=None, task_id=None)

    model.eval()

    gate_input = torch.randn(size=(batch_size, sequence_length, gate_input_size))
    expert_input = torch.randn(size=(batch_size, sequence_length) + expert_input_size)
    task_bh = torch.tensor([2])

    out, loss = model(gate_input, expert_input, task_bh)
    print(out.size(), loss)





    batch_size = 64
    sequence_length = 10
    gate_input_size = 16
    expert_input_size = (3,80,80)
    expert_output_size = (1,80,80)


    module = nn.Conv2d(3, 1, 3, 1, 1)
    num_experts = 8
    k = 2
    model = MoE(gate_input_size, 
                expert_input_size,
                expert_output_size,
                module, 
                num_experts, 
                k, 
                w_MI=0, w_H=0, w_finetune_MI=0, limit_k=0, w_topk_loss=0.0, task_num=8, noisy_gating=False, gating_activation=None, task_id=None)

    model.eval()

    gate_input = torch.randn(size=(batch_size, sequence_length, gate_input_size))
    expert_input = torch.randn(size=(batch_size, sequence_length) + expert_input_size)
    task_bh = torch.tensor([2])

    out, loss = model(gate_input, expert_input, task_bh)
    print(out.size(), loss)
  
  
#   model = LinearMoE(input_size=input_size,output_size=output_size,num_experts_per_task=1,k=-1,module=None,activation=nn.Sequential(
#                         nn.GELU(),
#                     ),noisy_gating=False,fixed_task_num=3,acc_aux_loss=True)
  
#   print(model.experts.old_experts.training)
#   exit()
#   if type(input_size) is not int:
#       input_data = torch.randn(batch_size,  *input_size)
#   else:
#     input_data = torch.randn(batch_size, sequence_length, input_size)

#   # Specify the task or task batch you want to perform inference for.
#   task_batch_index = int(-1) # Replace with the appropriate task batch index.

#   # You can skip certain tokens during inference by providing a skip_mask. 
#   # Set to None if you don't want to skip any tokens.
#   skip_mask = None

#   # Perform inference (forward pass) using the TaskMoE model for the specified task.
#   output= model(input_data, task_batch_index, skip_mask=skip_mask)
# #   print(model)
#   print(output.shape)
#   for name, param in model.named_parameters():
#     print(name, param.shape, param.requires_grad)
      
#     #   print(name, param.shape, param.requires_grad)
#     # print(name, param.shape, param.requires_grad)