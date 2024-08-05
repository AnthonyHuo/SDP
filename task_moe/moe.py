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
    def __init__(self, fixed_num_experts, new_num_experts, expert_module:nn.Module) -> None:
        super().__init__()
        
        if fixed_num_experts > 0:
            old_experts = nn.ModuleList([clone_module(expert_module) for _ in range(fixed_num_experts)])
        else:
            old_experts = None
        new_experts = nn.ModuleList([clone_module(expert_module) for _ in range(new_num_experts)])
        if old_experts is not None:
            self.all_experts = old_experts + new_experts
        else:
            self.all_experts = new_experts
        # set old experts to not require gradients
        if old_experts is not None:
            for i in range(fixed_num_experts):
                for param in self.all_experts[i].parameters():
                    param.requires_grad = False

        self.fixed_num_experts = fixed_num_experts
        self.new_num_experts = new_num_experts
        

    # def extra_repr(self):
    #     return 'new_num_experts={}, {}'.format(
    #         self.new_num_experts, self.new_experts.extra_repr()
    #     )
    
    def forward(self, inputs, expert_size,task_bh=-1):
        # if self.old_experts is None:
        #     all_experts = self.new_experts
        # else:
        #     all_experts = self.old_experts + self.new_experts
        all_experts = self.all_experts
        expert_size = expert_size.tolist()

        # exit()
        input_list = inputs.split(expert_size, dim=0)
        output_list = []
        if not self.training:
            if task_bh==-1:
                task_bh = self.fixed_num_experts//self.new_num_experts
                assert (task_bh+1)*self.new_num_experts == len(all_experts)
            all_experts = all_experts[:(task_bh+1)*self.new_num_experts]
            # print('Activating expert: 0-{}'.format((task_bh+1)*self.new_num_experts))
        for i, expert in enumerate(all_experts):
            output_list.append(expert(input_list[i]))
        
        return torch.cat(output_list, dim=0)

class LinearExperts(nn.Module):
    def __init__(self, fixed_num_experts, new_num_experts, input_size, output_size) -> None:
        super().__init__()
        assert new_num_experts == 1 # only support one expert for now
        if fixed_num_experts > 0:
            self.old_experts = nn.ModuleList([nn.Linear(input_size, output_size) for _ in range(fixed_num_experts)])
        else:
            self.old_experts = None
        self.new_experts = nn.ModuleList([nn.Linear(input_size, output_size) for _ in range(new_num_experts)])

        # set old experts to not require gradients
        if self.old_experts is not None:
            for expert in self.old_experts:
                for param in expert.parameters():
                    param.requires_grad = False

        self.fixed_num_experts = fixed_num_experts
        self.new_num_experts = new_num_experts
        

    # def extra_repr(self):
    #     return 'new_num_experts={}, {}'.format(
    #         self.new_num_experts, self.new_experts.extra_repr()
    #     )
    
    def forward(self, inputs,task_bh=-1):
        if self.old_experts is None:
            all_experts = self.new_experts
        else:
            all_experts = self.old_experts + self.new_experts

        # exit()
        if not self.training:
            if task_bh==-1:
                task_bh = len(self.old_experts)//self.new_num_experts if self.old_experts is not None else 0
                assert (task_bh+1)*self.new_num_experts == len(all_experts)
        
        # for i, expert in enumerate(all_experts):
        #     output_list.append(expert(input_list[i]))
        # print("Activating expert: ", task_bh)
        return all_experts[task_bh](inputs)   #torch.cat(output_list, dim=0)
    

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
    def __init__(self, input_size,output_size, module:nn.Module , num_experts_per_task, k, w_MI=0, w_H=0, w_finetune_MI=0, limit_k=0, w_topk_loss=0.0, fixed_task_num=9, noisy_gating=True, gating_activation=None, task_id=None,**kwargs):
        super(MoE, self).__init__()
        self.fixed_task_num = fixed_task_num
        self.w_topk_loss = w_topk_loss
        self.w_MI = w_MI
        self.w_H = w_H
        self.w_finetune_MI = w_finetune_MI
        self.input_size = input_size
        self.output_size = output_size
        self.task_id = task_id
        self.use_conv_router=False
        self.input_size_len = 1

        if type(input_size) is not int and len(input_size)>1:
            assert len(input_size)==3
            self.use_conv_router=True
            self.input_size_len = len(input_size)
        self.input_size = input_size

        self.limit_k = max(k, limit_k)

        if fixed_task_num > 0:
            self.new_num_experts = num_experts_per_task
            self.experts = ParallelExperts(fixed_task_num * num_experts_per_task, self.new_num_experts, module)
        else:
            self.new_num_experts = num_experts_per_task
            self.experts = ParallelExperts(0, self.new_num_experts, module)

        
        self.num_experts = fixed_task_num * num_experts_per_task + self.new_num_experts
        self.bias = True
        self.k = min(k, self.num_experts)
        self.activation = kwargs.get('activation', None)
        self.noisy_gating = noisy_gating
        
        
        if gating_activation is None:
            gating_activation = nn.GELU()

        
        if w_finetune_MI < -100: ## hack
            w_finetune_MI = 0
            self.w_finetune_MI = 0
            
            
            if fixed_task_num>0:
                if not self.use_conv_router:
                    self.f_gate = nn.ModuleList([nn.Sequential(
                                                        nn.Linear(input_size,
                                                            2 * ((i+1)*num_experts_per_task) if noisy_gating else ((i+1)*num_experts_per_task),
                                                            bias=False)
                                                ) for i in range(fixed_task_num)])
                    self.f_gate.append(nn.Sequential(
                                                        nn.Linear(input_size,
                                                            2 * (self.num_experts) if noisy_gating else self.num_experts,
                                                            bias=False)
                                                ))
                else:
                    # in: C1, H, W; out: C2, 1, 1
                    self.f_gate = nn.ModuleList([nn.Sequential(
                                                    nn.Conv2d(input_size[0], 2 * ((i+1)*num_experts_per_task) if noisy_gating else ((i+1)*num_experts_per_task), kernel_size=(input_size[1], input_size[2]), bias=False)
                                              
                                                ) for i in range(fixed_task_num)])
                    self.f_gate.append(nn.Sequential(
                                                    nn.Conv2d(input_size[0], 2 * (self.num_experts) if noisy_gating else self.num_experts, kernel_size=(input_size[1], input_size[2]), bias=False)
                                            
                                                ))
            else:
                if not self.use_conv_router:
                    self.f_gate = nn.ModuleList([nn.Sequential(
                                                        nn.Linear(input_size,
                                                            2 * (self.num_experts) if noisy_gating else self.num_experts,
                                                            bias=False)
                                                )])
                else:
                    self.f_gate = nn.ModuleList([nn.Sequential(
                                                    nn.Conv2d(input_size[0], 2 * (self.num_experts) if noisy_gating else self.num_experts, kernel_size=(input_size[1], input_size[2]), bias=False)
                                                )])

        else:
            if fixed_task_num>0:
                if not self.use_conv_router:
                    self.f_gate = nn.ModuleList([nn.Sequential(
                                                    nn.Linear(input_size, input_size//4),
                                                    gating_activation,
                                                    nn.Linear(input_size//4,
                                                            2 * ((i+1)*num_experts_per_task) if noisy_gating else ((i+1)*num_experts_per_task),
                                                            bias=True)
                                                ) for i in range(fixed_task_num)])
                    self.f_gate.append(nn.Sequential(
                                                nn.Linear(input_size, input_size//4),
                                                gating_activation,
                                                nn.Linear(input_size//4,
                                                        2 * (self.num_experts) if noisy_gating else (self.num_experts),
                                                        bias=True)
                                            ))
                else:
                    self.f_gate = nn.ModuleList([nn.Sequential(
                                                    nn.Conv2d(input_size[0], input_size[0], 1),
                                                    gating_activation,
                                                    nn.Conv2d(input_size[0],
                                                            2 * ((i+1)*num_experts_per_task) if noisy_gating else ((i+1)*num_experts_per_task),
                                                            kernel_size=(input_size[1], input_size[2]),
                                                            bias=True),
                  
                                                ) for i in range(fixed_task_num)])
                    self.f_gate.append(nn.Sequential(
                                                nn.Conv2d(input_size[0], input_size[0], kernel_size=1),
                                                gating_activation,
                                                nn.Conv2d(input_size[0],
                                                        2 * (self.num_experts) if noisy_gating else self.num_experts,
                                                        kernel_size=(input_size[1], input_size[2]),
                                                        bias=True),
                                             
                                            ))
            else:
                if not self.use_conv_router:
                    self.f_gate = nn.ModuleList([nn.Sequential(
                                                nn.Linear(input_size, input_size//4),
                                                gating_activation,
                                                nn.Linear(input_size//4,
                                                        2 * (self.num_experts) if noisy_gating else (self.num_experts),
                                                        bias=True),
                                            
                                            )])
                else:
                    self.f_gate = nn.ModuleList([nn.Sequential(
                                                nn.Conv2d(input_size[0], input_size[0], 1),
                                                gating_activation,
                                                nn.Conv2d(input_size[0],
                                                        2 * (self.num_experts) if noisy_gating else self.num_experts,
                                                        kernel_size=(input_size[1], input_size[2]),
                                                        bias=True)),
                                              
                                            ])
        
        # for i in range(task_num):
        nn.init.zeros_(self.f_gate[-1][-1].weight)
        # frozen all the weights except the last layer
        if self.fixed_task_num > 0:
            for i in range(self.fixed_task_num):
                for param in self.f_gate[i].parameters():
                    param.requires_grad = False

        self.register_buffer('PTE', torch.zeros(1, self.num_experts))
        self.register_buffer('PE', torch.zeros(self.num_experts))
        self.momentum = 0.0
        self.register_buffer('times',torch.zeros(1))
        # self.times = 0

        self.task_gate_freq = [0] * 1
        self.topk_acc_probs = [0] * 1
        self.token_probs = [0] * 1

    # def eval(self):
    #     super().eval()
    #     self.training = False
    #     self.f_gate.eval()
    #     self.experts.eval()

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

    def forward(self, x, task_bh=None, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        if task_bh is None:
            task_bh = self.task_id
            if task_bh is None:
                raise ValueError('task_id is not given!')

        other_dims = x.size()[:len(x.size())-self.input_size_len]
        other_size=1
        for i in other_dims:
            other_size*=i
        # bsz, length, emb_size = x.size()[0], x.size()[1], x.size()[2:]
        emb_size=x.size()[-self.input_size_len:]
        if self.use_conv_router:
            x = x.reshape(-1, *emb_size)
        else:
            x = x.reshape(-1, emb_size[0])

        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)

        
        loss = self.top_k_gating(x, task_bh, skip_mask,  sample_topk=sample_topk)
        
        expert_inputs = x[self.batch_index]

        h = self.experts(expert_inputs, self.expert_size, task_bh)
        
        expert_outputs = h

        if multiply_by_gates:
            dim_len=len(expert_outputs.size())-1
            batch_gates = self.batch_gates.view(-1, *[1]*dim_len)
            expert_outputs = expert_outputs * batch_gates

        zeros = torch.zeros((other_size, self.output_size),
            dtype=expert_outputs.dtype, device=expert_outputs.device)

        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(*other_dims, self.output_size)
        # y = y.view(bsz,self.input_size,length)
        # assert torch.allclose(y, y_)
        
        return y


if __name__ == '__main__':
  batch_size = 64
  sequence_length = 10
#   input_size = 10
#   output_size = 5
#   module = nn.Linear(input_size, output_size)
  input_size = torch.Size([3,80,80])
  output_size = 10
  module = nn.Sequential(
      nn.Conv2d(3, 1, 3, 1, 1), # 3x80x80 -> 1x80x80
      nn.Flatten(), # 1x80x80 -> 6400
        nn.Linear(6400, 10) # 6400 -> 10
  )
  model = MoE(input_size=input_size,output_size=output_size,module = module ,num_experts_per_task=2,k=2
                    ,noisy_gating=False,fixed_task_num=1,acc_aux_loss=True)
#   model = LinearMoE(input_size=input_size,output_size=output_size,num_experts_per_task=1,k=-1,module=None,activation=nn.Sequential(
#                         nn.GELU(),
#                     ),noisy_gating=False,fixed_task_num=3,acc_aux_loss=True)
  model.eval()
#   print(model.experts.old_experts.training)
#   exit()
  if type(input_size) is not int:
      input_data = torch.randn(batch_size,  *input_size)
  else:
    input_data = torch.randn(batch_size, sequence_length, input_size)

  # Specify the task or task batch you want to perform inference for.
  task_batch_index = int(-1) # Replace with the appropriate task batch index.

  # You can skip certain tokens during inference by providing a skip_mask. 
  # Set to None if you don't want to skip any tokens.
  skip_mask = None

  # Perform inference (forward pass) using the TaskMoE model for the specified task.
  output= model(input_data, task_batch_index, skip_mask=skip_mask)
#   print(model)
  print(output.shape)
  for name, param in model.named_parameters():
    print(name, param.shape, param.requires_grad)
      
    #   print(name, param.shape, param.requires_grad)
    # print(name, param.shape, param.requires_grad)