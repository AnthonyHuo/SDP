import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
from typing import Any, Dict, List, Optional
from torch import Tensor


class ParallelLinear(torch.autograd.Function):

    @staticmethod
    @custom_fwd #(cast_inputs=torch.float32)
    def forward(ctx, input, expert_size, weight, bias=None):
        output = ParallelLinear.forward_scriptable(input, expert_size, weight, bias)
        # assert torch.allclose(ParallelLinear._forward_scriptable(input, expert_size, weight, bias),  output)
        ctx.save_for_backward(input, expert_size, weight, bias)
        return output

    @staticmethod
    @torch.jit.script
    def forward_scriptable(input: Tensor, expert_size: Tensor,
                           weight: Tensor, bias: Optional[Tensor]):
        output_buf: Tensor = torch.empty((input.size(0), weight.size(2)),
                                         device=input.device, dtype=input.dtype)
        num_linears = weight.size(0)

        expert_size_list: List[int] = expert_size.tolist()
        # print('expert_size: ', expert_size)
        input_list = input.split(expert_size_list, dim=0)
        output_buf_list = output_buf.split(expert_size_list)

        for i in range(num_linears):
            torch.mm(input_list[i], weight[i], out=output_buf_list[i])

        if bias is not None:
            for i in range(num_linears):
                output_buf_list[i].add_(bias[i])

        output = output_buf
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        input, expert_size, weight, bias = ctx.saved_tensors
        return ParallelLinear.backward_scriptable(
            grad_out, input, expert_size,
            weight, bias
        )

    @staticmethod
    @torch.jit.script
    def backward_scriptable(grad_out: Tensor,
                 input: Tensor, expert_size: Tensor,
                 weight: Tensor, bias: Optional[Tensor]):
        num_linears = weight.size(0)
        expert_size_list: List[int] = expert_size.tolist()
        input_list = input.t().split(expert_size_list, dim=1)
        grad_list = grad_out.split(expert_size_list, dim=0)

        d_input_buf = torch.empty_like(input)
        d_input_buf_list = d_input_buf.split(expert_size_list, dim=0)
        d_weight_buf = torch.empty_like(weight)

        weight_t = weight.permute(0, 2, 1)

        for i in range(num_linears):
            torch.mm(grad_list[i], weight_t[i], out=d_input_buf_list[i])
            torch.mm(input_list[i], grad_list[i], out=d_weight_buf[i])

        d_input = d_input_buf
        d_weight = d_weight_buf

        if bias is not None:
            d_bias_buf = torch.empty_like(bias)
            for i in range(num_linears):
                torch.sum(grad_list[i], dim=0, keepdim=False, out=d_bias_buf[i])
            d_bias = d_bias_buf
        else:
            d_bias = None

        return d_input, None, d_weight, d_bias


class ParallelExperts(nn.Module):
    def __init__(self, num_experts, input_size, output_size, bias=False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, input_size, output_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_experts, output_size))
        else:
            self.bias = None
        self.reset_parameters()

    def extra_repr(self):
        return 'num_experts={}, input_size={}, output_size={}'.format(
            self.weight.size(0), self.weight.size(1), self.weight.size(2))

    def reset_parameters(self) -> None:
        # std = math.sqrt(2.0 / float(self.weight.size(1) + self.weight.size(2)))
        # a = math.sqrt(3.0) * std
        nn.init.uniform_(self.weight, -1. / self.weight.size(1), 1. / self.weight.size(1))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs, expert_size):
        results = ParallelLinear.apply(inputs, expert_size, self.weight, self.bias)
        return results

# import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.cuda.amp import custom_fwd, custom_bwd


# class ParallelLinear(torch.autograd.Function):
#     @staticmethod
#     @custom_fwd
#     def forward(ctx, input, expert_size, weight, bias=None):
#         output_list = []
#         expert_size_list = expert_size.tolist()
#         input_list = input.split(expert_size_list, dim=0)
#         for i in range(weight.size(0)):
#             if bias is not None:
#                 o_i = torch.mm(input_list[i], weight[i]) + bias[i]
#             else:
#                 o_i = torch.mm(input_list[i], weight[i])
#             output_list.append(o_i)
#         output = torch.cat(output_list, dim=0)
#         variables = (input, expert_size, weight, bias)
#         ctx.save_for_backward(*variables)
#         return output

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, grad_out):
#         input, expert_size, weight, bias = ctx.saved_tensors
#         num_linears = weight.size(0)

#         expert_size_list = expert_size.tolist()
#         input_list = input.split(expert_size_list, dim=0)
#         grad_list = grad_out.split(expert_size_list, dim=0)

#         d_input_list = []
#         for i in range(num_linears):
#             d_input_list.append(torch.einsum('bi,ji->bj', grad_list[i], weight[i]))
#         d_input = torch.cat(d_input_list, dim=0)

#         d_weight_list = []
#         for i in range(num_linears):
#             d_weight_list.append(torch.einsum('bi,bj->ij', input_list[i], grad_list[i]))
#         d_weight = torch.stack(d_weight_list, dim=0)

#         if bias is not None:
#             d_bias_list = []
#             for i in range(num_linears):
#                 d_bias_list.append(grad_list[i].sum(0))
#             d_bias = torch.stack(d_bias_list, dim=0)
#         else:
#             d_bias = None
#         return d_input, None, d_weight, d_bias


# class ParallelExperts(nn.Module):
#     def __init__(self, num_experts, input_size, output_size, bias=False) -> None:
#         super().__init__()
#         self.w = nn.Parameter(torch.empty(num_experts, input_size, output_size))
#         if bias:
#             self.b = nn.Parameter(torch.zeros(num_experts, output_size))
#         else:
#             self.b = None

#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         std = math.sqrt(2.0 / float(self.w.size(1) + self.w.size(2)))
#         a = math.sqrt(3.0) * std
#         nn.init.uniform_(self.w, -a, a)

#     def forward(self, inputs, expert_size):
#         results = ParallelLinear.apply(inputs, expert_size, self.w, self.b)
#         return results

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.cuda.amp import custom_fwd, custom_bwd
# from typing import Any, Dict, List, Optional
# from torch import Tensor


# class ParallelLinear(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float16)
#     def forward(ctx, input, expert_size, batch_index, weight, bias=None):
#         expert_size_list: List[int] = expert_size.tolist()
#         output = ParallelLinear.forward_scriptable(input, expert_size_list, batch_index, weight, bias)
#         # assert torch.allclose(ParallelLinear._forward_scriptable(input, expert_size, weight, bias),  output)
#         ctx.save_for_backward(input, batch_index, weight, bias)
#         ctx.expert_size_list = expert_size_list
#         return output

#     @staticmethod
#     @torch.jit.script
#     def forward_scriptable(input: Tensor, expert_size_list: List[int], batch_index: Optional[Tensor],
#                            weight: Tensor, bias: Optional[Tensor]):
#         if batch_index is not None:
#             input = input[batch_index]
#         output_buf: Tensor = torch.empty((input.size(0), weight.size(2)),
#                                          device=input.device, dtype=input.dtype)
#         num_linears = weight.size(0)

#         input_list = input.split(expert_size_list, dim=0)
#         output_buf_list = output_buf.split(expert_size_list)

#         for i in range(num_linears):
#             torch.mm(input_list[i], weight[i], out=output_buf_list[i])

#         if bias is not None:
#             for i in range(num_linears):
#                 output_buf_list[i].add_(bias[i])

#         output = output_buf
#         return output

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, grad_out):
#         input, batch_index, weight, bias = ctx.saved_tensors
#         expert_size_list = ctx.expert_size_list
#         return ParallelLinear.backward_scriptable(
#             grad_out, input, expert_size_list, batch_index,
#             weight, bias
#         )

#     @staticmethod
#     @torch.jit.script
#     def backward_scriptable(grad_out: Tensor,
#                  input: Tensor, expert_size_list: List[int], batch_index: Optional[Tensor],
#                  weight: Tensor, bias: Optional[Tensor]):
#         num_linears = weight.size(0)
#         d_input = torch.zeros_like(input)
#         if batch_index is not None:
#             input = input[batch_index]
#         input_list = input.t().split(expert_size_list, dim=1)
#         grad_list = grad_out.split(expert_size_list, dim=0)

#         d_input_buf = torch.empty_like(input)
#         d_input_buf_list = d_input_buf.split(expert_size_list, dim=0)
#         d_weight_buf = torch.empty_like(weight)

#         weight_t = weight.permute(0, 2, 1)

#         for i in range(num_linears):
#             torch.mm(grad_list[i], weight_t[i], out=d_input_buf_list[i])
#             torch.mm(input_list[i], grad_list[i], out=d_weight_buf[i])

#         # d_input = d_input_buf
#         d_weight = d_weight_buf

#         if bias is not None:
#             d_bias_buf = torch.empty_like(bias)
#             for i in range(num_linears):
#                 torch.sum(grad_list[i], dim=0, keepdim=False, out=d_bias_buf[i])
#             d_bias = d_bias_buf
#         else:
#             d_bias = None

#         if batch_index is not None:
#             d_input.index_add_(0, batch_index, d_input_buf)
#         else:
#             d_input = d_input + d_input_buf

#         return d_input, None, None, d_weight, d_bias


# class ParallelExperts(nn.Module):
#     def __init__(self, num_experts, input_size, output_size, bias=False) -> None:
#         super().__init__()
#         # self.input_experts = nn.ModuleList(
#         #     [nn.Linear(input_size, output_size, bias=bias) for _ in range(num_experts)]
#         # )
#         self.weight = nn.Parameter(torch.empty(num_experts, input_size, output_size))
#         if bias:
#             self.bias = nn.Parameter(torch.zeros(num_experts, output_size))
#         else:
#             self.bias = None
#         self.reset_parameters()
#         self.num_experts = num_experts
#         self.input_size = input_size
#         self.output_size = output_size

#     def extra_repr(self):
#         return 'num_experts={}, input_size={}, output_size={}'.format(
#             self.num_experts, self.input_size, self.output_size)

#     def reset_parameters(self) -> None:
#         # std = math.sqrt(2.0 / float(self.weight.size(1) + self.weight.size(2)))
#         # a = math.sqrt(3.0) * std
#         nn.init.uniform_(self.weight, -1. / self.weight.size(1), 1. / self.weight.size(1))
#         if self.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
#             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             nn.init.uniform_(self.bias, -bound, bound)

#     def forward(self, inputs, expert_size, batch_index=None):
#         # if batch_index is None:
#         #     batch_index = torch.LongTensor(inputs.shape[0])
#         #     assert False
#         results = ParallelLinear.apply(inputs, expert_size, batch_index, self.weight, self.bias)
#         # expert_size_list: List[int] = expert_size.tolist()
#         # input_list = inputs.split(expert_size_list, dim=0)
#         # output_list = []
#         # for i in range(self.num_experts):
#         #     output_list.append(self.input_experts[i](input_list[i]))
#         # results = torch.cat(output_list, dim=0)
#         return results

# # import math
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.cuda.amp import custom_fwd, custom_bwd
# # from typing import Any, Dict, List, Optional
# # from torch import Tensor


# # class ParallelLinear(torch.autograd.Function):

# #     @staticmethod
# #     @custom_fwd(cast_inputs=torch.float16)
# #     def forward(ctx, input, expert_size, batch_index, weight, bias=None):
# #         # def forward(ctx, input, expert_size, weight, bias=None):
# #         # output = ParallelLinear.forward_scriptable(input, expert_size, weight, bias)
# #         output = ParallelLinear.forward_scriptable(input, expert_size_list, batch_index, weight, bias)
# #         # assert torch.allclose(ParallelLinear._forward_scriptable(input, expert_size, weight, bias),  output)
# #         # ctx.save_for_backward(input, expert_size, weight, bias)
# #         ctx.save_for_backward(input, batch_index, weight, bias)
# #         return output

# #     @staticmethod
# #     @torch.jit.script
# #     def forward_scriptable(input: Tensor, expert_size_list: List[int], batch_index: Tensor,
# #                            weight: Tensor, bias: Optional[Tensor]):
# #         # def forward_scriptable(input: Tensor, expert_size: Tensor,
# #         #                        weight: Tensor, bias: Optional[Tensor]):
# #         input = input[batch_index]
# #         output_buf: Tensor = torch.empty((input.size(0), weight.size(2)),
# #                                          device=input.device, dtype=input.dtype)
# #         num_linears = weight.size(0)

# #         # expert_size_list: List[int] = expert_size.tolist()
# #         input_list = input.split(expert_size_list, dim=0)
# #         output_buf_list = output_buf.split(expert_size_list)

# #         for i in range(num_linears):
# #             torch.mm(input_list[i], weight[i], out=output_buf_list[i])

# #         if bias is not None:
# #             for i in range(num_linears):
# #                 output_buf_list[i].add_(bias[i])

# #         output = output_buf
# #         return output

# #     @staticmethod
# #     @custom_bwd
# #     def backward(ctx, grad_out):
# #         # input, expert_size, weight, bias = ctx.saved_tensors
# #         input, batch_index, weight, bias = ctx.saved_tensors
# #         return ParallelLinear.backward_scriptable(
# #             grad_out, input, expert_size, batch_index,
# #             weight, bias
# #         )

# #     @staticmethod
# #     @torch.jit.script
# #     def backward_scriptable(grad_out: Tensor,
# #                  input: Tensor, expert_size: Tensor, batch_index: Tensor,
# #                  weight: Tensor, bias: Optional[Tensor]):
# #         num_linears = weight.size(0)
# #         d_input = torch.zeros_like(input)
# #         input = input[batch_index]
# #         expert_size_list: List[int] = expert_size.tolist()
# #         input_list = input.t().split(expert_size_list, dim=1)
# #         grad_list = grad_out.split(expert_size_list, dim=0)

# #         d_input_buf = torch.empty_like(input)
# #         d_input_buf_list = d_input_buf.split(expert_size_list, dim=0)
# #         d_weight_buf = torch.empty_like(weight)

# #         weight_t = weight.permute(0, 2, 1)

# #         for i in range(num_linears):
# #             torch.mm(grad_list[i], weight_t[i], out=d_input_buf_list[i])
# #             torch.mm(input_list[i], grad_list[i], out=d_weight_buf[i])

# #         # d_input = d_input_buf
# #         d_weight = d_weight_buf

# #         if bias is not None:
# #             d_bias_buf = torch.empty_like(bias)
# #             for i in range(num_linears):
# #                 torch.sum(grad_list[i], dim=0, keepdim=False, out=d_bias_buf[i])
# #             d_bias = d_bias_buf
# #         else:
# #             d_bias = None

# #         d_input.index_add_(0, batch_index, d_input_buf)
# #         return d_input, None, d_weight, d_bias


# # class ParallelExperts(nn.Module):
# #     def __init__(self, num_experts, input_size, output_size, bias=False) -> None:
# #         super().__init__()
# #         self.w = nn.Parameter(torch.empty(num_experts, input_size, output_size))
# #         if bias:
# #             self.b = nn.Parameter(torch.zeros(num_experts, output_size))
# #         else:
# #             self.b = None
# #         self.reset_parameters()

# #     def extra_repr(self):
# #         return 'num_experts={}, input_size={}, output_size={}'.format(
# #             self.w.size(0), self.w.size(1), self.w.size(2))

# #     def reset_parameters(self) -> None:
# #         # std = math.sqrt(2.0 / float(self.w.size(1) + self.w.size(2)))
# #         # a = math.sqrt(3.0) * std
# #         nn.init.uniform_(self.w, -1. / self.w.size(1), 1. / self.w.size(1))
# #         if self.b is not None:
# #             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w[0])
# #             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
# #             nn.init.uniform_(self.b, -bound, bound)

# #     # def forward(self, inputs, expert_size):
# #     def forward(self, inputs, expert_size, batch_index):
# #         results = ParallelLinear.apply(inputs, expert_size, batch_index, self.weight, self.bias)
# #         # results = ParallelLinear.apply(inputs, expert_size, self.w, self.b)
# #         return results

# # import math

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.cuda.amp import custom_fwd, custom_bwd


# # class ParallelLinear(torch.autograd.Function):
# #     @staticmethod
# #     @custom_fwd
# #     def forward(ctx, input, expert_size, weight, bias=None):
# #         output_list = []
# #         expert_size_list = expert_size.tolist()
# #         input_list = input.split(expert_size_list, dim=0)
# #         for i in range(weight.size(0)):
# #             if bias is not None:
# #                 o_i = torch.mm(input_list[i], weight[i]) + bias[i]
# #             else:
# #                 o_i = torch.mm(input_list[i], weight[i])
# #             output_list.append(o_i)
# #         output = torch.cat(output_list, dim=0)
# #         variables = (input, expert_size, weight, bias)
# #         ctx.save_for_backward(*variables)
# #         return output

# #     @staticmethod
# #     @custom_bwd
# #     def backward(ctx, grad_out):
# #         input, expert_size, weight, bias = ctx.saved_tensors
# #         num_linears = weight.size(0)

# #         expert_size_list = expert_size.tolist()
# #         input_list = input.split(expert_size_list, dim=0)
# #         grad_list = grad_out.split(expert_size_list, dim=0)

# #         d_input_list = []
# #         for i in range(num_linears):
# #             d_input_list.append(torch.einsum('bi,ji->bj', grad_list[i], weight[i]))
# #         d_input = torch.cat(d_input_list, dim=0)

# #         d_weight_list = []
# #         for i in range(num_linears):
# #             d_weight_list.append(torch.einsum('bi,bj->ij', input_list[i], grad_list[i]))
# #         d_weight = torch.stack(d_weight_list, dim=0)

# #         if bias is not None:
# #             d_bias_list = []
# #             for i in range(num_linears):
# #                 d_bias_list.append(grad_list[i].sum(0))
# #             d_bias = torch.stack(d_bias_list, dim=0)
# #         else:
# #             d_bias = None
# #         return d_input, None, d_weight, d_bias


# # class ParallelExperts(nn.Module):
# #     def __init__(self, num_experts, input_size, output_size, bias=False) -> None:
# #         super().__init__()
# #         self.w = nn.Parameter(torch.empty(num_experts, input_size, output_size))
# #         if bias:
# #             self.b = nn.Parameter(torch.zeros(num_experts, output_size))
# #         else:
# #             self.b = None

# #         self.reset_parameters()

# #     def reset_parameters(self) -> None:
# #         std = math.sqrt(2.0 / float(self.w.size(1) + self.w.size(2)))
# #         a = math.sqrt(3.0) * std
# #         nn.init.uniform_(self.w, -a, a)

# #     def forward(self, inputs, expert_size):
# #         results = ParallelLinear.apply(inputs, expert_size, self.w, self.b)
# #         return results
