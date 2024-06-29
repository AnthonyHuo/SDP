import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
class Gate(nn.Module):
    
    def __init__(self, k, patch_size, in_channels, gating_activation=None, gating_kernel_initializer=None):
        super(Gate, self).__init__()
        
        self.k = k
        self.patch_size = patch_size
        self.strides = patch_size
        self.padding = 0
        self.gating_activation = gating_activation
        self.in_channels = in_channels
        self.gating_kernel_1 = nn.Parameter(torch.empty(in_channels // 2, in_channels, patch_size, patch_size))
        self.gating_kernel_2 = nn.Parameter(torch.empty(1, in_channels // 2, 1, 1))
        
        
        init.normal_(self.gating_kernel_1, mean=0.0, std=0.0001)
        init.normal_(self.gating_kernel_2, mean=0.0, std=0.0001)
      


    def forward(self, inputs):
        
        # Convolution and divides into patches
        gating_outputs = F.conv2d(inputs, self.gating_kernel_1, stride=self.strides, padding=0)
        gating_outputs = F.conv2d(gating_outputs, self.gating_kernel_2, stride=1, padding=0)
        
        # Apply activation function if specified
        if self.gating_activation is not None:
            gating_outputs = self.gating_activation(gating_outputs)

        # Flatten and apply top-k
        b, c, h, w = gating_outputs.shape
        gating_outputs = gating_outputs.view(b, c, -1)
        values, indices = torch.topk(gating_outputs, self.k, dim=2, sorted=False)
        
        
        # Scatter values to original positions
        ret_flat = torch.zeros(b * c * h * w, device=inputs.device)
        indices_flat = indices.view(b*c,-1) + torch.arange(b * c, device=inputs.device).unsqueeze(-1) * h * w
        indices_flat = indices_flat.view(-1)
        ret_flat.scatter_add_(0, indices_flat, values.view(-1))
        
        # Reshape and reorder
        new_gating_outputs = ret_flat.view(b, c, h, w)

        # scale to original size (x patch size)
        new_gating_outputs = new_gating_outputs.repeat_interleave(self.patch_size, dim=2)
        new_gating_outputs = new_gating_outputs.repeat_interleave(self.patch_size, dim=3)
        new_gating_outputs = new_gating_outputs.repeat_interleave(self.in_channels, dim=1)
        
        outputs = inputs * new_gating_outputs

        return outputs
    

class MoEEncoder(nn.Module):
    def __init__(self):
        super(MoEEncoder, self).__init__()

        self.batch_norm = nn.BatchNorm2d(num_features=3)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pre_conv_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        
        
        self.gates = nn.ModuleList([Gate(k = 64, patch_size=4,in_channels=128, gating_activation=F.softmax) for _ in range(8)])
        self.batch_norm_gates = nn.ModuleList([nn.GroupNorm(num_groups=8, num_channels=128) for _ in range(8)])
        self.relu_gates = nn.ModuleList([nn.ReLU() for _ in range(8)])
        self.conv_gates = nn.ModuleList([nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1) for _ in range(8)])
        
        self.batch_norm_gates_1 = nn.ModuleList([nn.GroupNorm(num_groups=8, num_channels=128) for _ in range(8)])
        self.relu_gates_1 = nn.ModuleList([nn.ReLU() for _ in range(8)])
        self.conv_gates_1 = nn.ModuleList([nn.Conv2d(in_channels=128, out_channels=4, kernel_size=3, stride=1, padding=1) for _ in range(8)])
        
        self.resblocks = nn.ModuleList([ResidualBlock(in_channels = 32, out_channels=32) for _ in range(4)])
        
        self.out_cov = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)        
        
        self.avgpool = nn.AvgPool2d(kernel_size = 2, stride=2)
        
    def forward(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        
        x = x + self.pre_conv_1(x)
        
        gates_output = [gate(x) for gate in self.gates]
        # Batch Normalization
        gates_output = [batch_norm(gate_output) for batch_norm, gate_output in zip(self.batch_norm_gates, gates_output)]

        # ReLU
        gates_output = [relu(gate_output) for relu, gate_output in zip(self.relu_gates, gates_output)]

        # Convolution
        gates_output = [conv(gate_output) for conv, gate_output in zip(self.conv_gates, gates_output)]
        
        
        # Batch Normalization
        gates_output = [batch_norm(gate_output) for batch_norm, gate_output in zip(self.batch_norm_gates_1, gates_output)]

        # ReLU
        gates_output = [relu(gate_output) for relu, gate_output in zip(self.relu_gates_1, gates_output)]

        # Convolution
        gates_output = [conv(gate_output) for conv, gate_output in zip(self.conv_gates_1, gates_output)]
        
        # Concatenate
        x = torch.cat(gates_output, dim=1)
        
        
        for block in self.resblocks:
            x = block(x)
            
        x = self.out_cov(x)
        x = self.avgpool(x)
        
        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


    

if __name__ == "__main__":
    # Create an instance of the model
    model = MoEEncoder()

    # Assuming input is a PyTorch tensor with the appropriate shape
    input_tensor = torch.randn(64, 3, 64, 64)

    # Forward pass
    output_tensor = model(input_tensor)
    print("Output Shape:", output_tensor.shape)