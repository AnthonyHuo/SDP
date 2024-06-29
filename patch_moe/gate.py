import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Gate(nn.Module):
    
    def __init__(self, k, gating_kernel_size, strides=1, padding=0, 
                 gating_activation=None, gating_kernel_initializer=None):
        super(Gate, self).__init__()
        
        self.k = k
        self.gating_kernel_size = gating_kernel_size
        self.strides = strides
        self.padding = padding
        self.gating_activation = gating_activation

        self.gating_kernel = nn.Parameter(torch.empty(1, 3, 4, 4))

        # Initialize with normal distribution
        init.normal_(self.gating_kernel, mean=0.0, std=0.0001)
      


    def forward(self, inputs):
        
        # Convolution
        #(b,3,76,76)
        gating_outputs = F.conv2d(inputs, self.gating_kernel, stride=self.strides, padding=self.padding)
        #(b,1,19,19)
        # Apply activation function if specified
        if self.gating_activation is not None:
            gating_outputs = self.gating_activation(gating_outputs)

        # Flatten and apply top-k
        b, c, h, w = gating_outputs.shape
        gating_outputs = gating_outputs.view(b, c, -1)
        #(b,1,361)
        values, indices = torch.topk(gating_outputs, self.k, dim=2, sorted=False)
        #(b,1,2)
        # Scatter values to original positions
        out_shape = (b, c, h * w)
        ret_flat = torch.zeros(b * c * h * w, device=inputs.device)
        #[[20,40][34,56]]
        indices_flat = indices.view(b*c,-1) + torch.arange(b * c, device=inputs.device).unsqueeze(-1) * h * w
        indices_flat = indices_flat.view(-1)
        ret_flat.scatter_add_(0, indices_flat, values.view(-1))
        #[b,1,361]
        # Reshape and reorder
        new_gating_outputs = ret_flat.view(b, c, h, w)
        #[b,1,19,19]
        # Repeat and reshape the gating outputs
        new_gating_outputs = new_gating_outputs.repeat_interleave(self.gating_kernel_size[0], dim=2)
        new_gating_outputs = new_gating_outputs.repeat_interleave(self.gating_kernel_size[1], dim=3)
        new_gating_outputs = new_gating_outputs.repeat_interleave(self.gating_kernel.size(1), dim=1)
        #[b,48,19,19]
        # new_gating_outputs = new_gating_outputs.view(b, h, self.gating_kernel_size[0], w, self.gating_kernel_size[1],-1)
        # new_gating_outputs = new_gating_outputs.view(b, h * self.gating_kernel_size[0], w * self.gating_kernel_size[1],-1)
        # # new_gating_outputs = new_gating_outputs.permute(0, 3, 1, 2).contiguous()
        # repeat_factor = self.gating_kernel[0] * self.gating_kernel[1] * 3
        # new_gating_outputs = new_gating_outputs.repeat(1, 1, 1, 48)

        # # Step 2: Reshape new_gating_outputs
        # new_shape = (new_gating_outputs.size(0), new_gating_outputs.size(1), new_gating_outputs.size(2), 
        #             self.gating_kernel[0], self.gating_kernel[1], 3)
        # new_gating_outputs = new_gating_outputs.view(new_shape)

        # # Step 3: Transpose new_gating_outputs
        # new_gating_outputs = new_gating_outputs.permute(0, 1, 3, 2, 4, 5)

        # # Step 4: Final reshape
        # final_shape = (new_gating_outputs.size(0), new_gating_outputs.size(1) * new_gating_outputs.size(2), 
        #             new_gating_outputs.size(3) * new_gating_outputs.size(4), new_gating_outputs.size(5))
        # new_gating_outputs = new_gating_outputs.view(final_shape)
        # Element-wise multiplication
        outputs = inputs * new_gating_outputs

        return outputs
def test_gate_layer():
    # Parameters for the gate layer
    k = 2
    gating_kernel_size = (4, 4)  # Example kernel size
    strides = 4
    padding = 0

    # Initialize the Gate layer
    gate_layer = Gate(k, gating_kernel_size, strides, padding, gating_activation=torch.relu)

    # Create a random input tensor
    batch_size = 2
    in_channels = 3
    height, width = 16, 16  # Example dimensions
    input_tensor = torch.randn(batch_size, in_channels, height, width)

    # Forward pass through the Gate layer
    output = gate_layer(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    test_gate_layer()