import torch
import torch.nn as nn

from torchvision import models
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

class BottleNeck(nn.Module):
    # Scale factor of the number of output channels
    expansion = 4

    def __init__(self, in_channels, out_channels, 
                 stride=1, is_first_block=False):
        """
        Args: 
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride using in (a) 3x3 convolution and 
                    (b) 1x1 convolution used for downsampling for skip connection
            is_first_block: whether it is the first residual block of the layer
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels*self.expansion,
                               kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.relu = nn.ReLU()

        # Skip connection goes through 1x1 convolution with stride=2 for 
        # the first blocks of conv3_x, conv4_x, and conv5_x layers for matching
        # spatial dimension of feature maps and number of channels in order to 
        # perform the add operations.
        self.downsample = None
        if is_first_block:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                      out_channels=out_channels*self.expansion,
                                                      kernel_size=1,
                                                      stride=stride,
                                                      padding=0),
                                            nn.BatchNorm2d(out_channels*self.expansion))
            

    def forward(self, x):
        """
        Args:
            x: input
        Returns:
            Residual block output
        """
        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class BasicBlock(nn.Module):
    # Scale factor of the number of output channels
    expansion = 1

    def __init__(self, in_channels, out_channels,
                 stride=1, is_first_block=False):
        """
        Args: 
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride using in (a) the first 3x3 convolution and 
                    (b) 1x1 convolution used for downsampling for skip connection
            is_first_block: whether it is the first residual block of the layer
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        # Skip connection goes through 1x1 convolution with stride=2 for 
        # the first blocks of conv3_x, conv4_x, and conv5_x layers for matching
        # spatial dimension of feature maps and number of channels in order to 
        # perform the add operations.
        self.downsample = None
        if is_first_block and stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                                      out_channels=out_channels,
                                                      kernel_size=1,
                                                      stride=stride,
                                                      padding=0),
                                            nn.BatchNorm2d(out_channels))


    def forward(self, x):
        """
        Args:
            x: input
        Returns:
            Residual block ouput
        """
        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsample:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)

        return x
    
    
    
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
    
    
    
    
class PatchMoeBasicBlock(nn.Module):
    # Scale factor of the number of output channels
    expansion = 1

    def __init__(self, k, exp, patch_size, in_channels, out_channels,
                 stride=1, is_first_block=False):
        """
        Args: 
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride using in (a) the first 3x3 convolution and 
                    (b) 1x1 convolution used for downsampling for skip connection
            is_first_block: whether it is the first residual block of the layer
        """
        super().__init__()
        self.patch_size = patch_size
        self.exp = exp
        self.k = k
        
        self.gate = nn.ModuleList([Gate(k = self.k, 
                                        patch_size=self.patch_size,
                                        in_channels=in_channels, 
                                        gating_activation=F.softmax) for _ in range(self.exp)])
        self.conv1 = nn.ModuleList([nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1) for _ in range(self.exp)])
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(self.exp)])
        self.conv2 = nn.ModuleList([nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1) for _ in range(self.exp)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(self.exp)])

        self.relu = nn.ModuleList([nn.ReLU() for _ in range(self.exp)])

        # Skip connection goes through 1x1 convolution with stride=2 for 
        # the first blocks of conv3_x, conv4_x, and conv5_x layers for matching
        # spatial dimension of feature maps and number of channels in order to 
        # perform the add operations.
        self.downsample = None
        if is_first_block and stride != 1:
            self.downsample = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                                      out_channels=out_channels,
                                                      kernel_size=1,
                                                      stride=stride,
                                                      padding=0),
                                            nn.BatchNorm2d(out_channels)) for _ in range(self.exp)])


    def forward(self, x):
        """
        Args:
            x: input
        Returns:
            Residual block ouput
        """
        patch_valid = (self.k <= (x.size(2) / self.patch_size)**2)
        
        
        identity = x.clone()
        
        if patch_valid:
            gate_outputs = [gate(x) for gate in self.gate]
            exp = self.exp
        else:
            gate_outputs = [x]
            exp = 1
        
        print('gate_outputs',gate_outputs[0].size())
        
        # exp_outputs = []
        exp_outputs = 0
        for i in range(exp):
            conv1 = self.conv1[i]
            conv2 = self.conv2[i]
            relu = self.relu[i]
            bn1 = self.bn1[i]
            bn2 = self.bn2[i]
            if self.downsample:
                downsample = self.downsample[i]
            
            y = relu(bn1(conv1(gate_outputs[i])))
            y = bn2(conv2(y))

            if self.downsample:
                initial = downsample(identity)
            else:
                initial = identity
            y += initial
            y = relu(y)
            # exp_outputs.append(y)
            exp_outputs += y
            
        # exp_outputs

        print('exp outputs',exp_outputs.size())
        
        return exp_outputs


class ResNet(nn.Module):
    def __init__(self, resblock = 'basic', n_blocks_list=[3, 4, 6, 3],
                 out_channels_list=[64, 128, 256, 512], num_channels=3):
        """
        Args:
            ResBlock: residual block type, BasicBlock for ResNet-18, 34 or 
                      BottleNeck for ResNet-50, 101, 152
            n_class: number of classes for image classifcation (used in classfication head)
            n_block_lists: number of residual blocks for each conv layer (conv2_x - conv5_x)
            out_channels_list: list of the output channel numbers for conv2_x - conv5_x
            num_channels: the number of channels of input image
        """
        super().__init__()

        if resblock == 'basic':
            ResBlock = BasicBlock
        elif resblock == 'bottle':
            ResBlock = BottleNeck
        
        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=num_channels, 
                                             out_channels=64, kernel_size=7,
                                             stride=2, padding=3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3,
                                                stride=2, padding=1))

        # Create four convoluiontal layers
        in_channels = 64
        # For the first block of the second layer, do not downsample and use stride=1.
        self.conv2_x = self.CreateLayer(ResBlock, n_blocks_list[0], 
                                        in_channels, out_channels_list[0], stride=1)
        
        # For the first blocks of conv3_x - conv5_x layers, perform downsampling using stride=2.
        # By default, ResBlock.expansion = 4 for ResNet-50, 101, 152, 
        # ResBlock.expansion = 1 for ResNet-18, 34.
        self.conv3_x = self.CreateLayer(ResBlock, n_blocks_list[1], 
                                        out_channels_list[0]*ResBlock.expansion,
                                        out_channels_list[1], stride=2)
        self.conv4_x = self.CreateLayer(ResBlock, n_blocks_list[2],
                                        out_channels_list[1]*ResBlock.expansion,
                                        out_channels_list[2], stride=2)
        self.conv5_x = self.CreateLayer(ResBlock, n_blocks_list[3], 
                                        out_channels_list[2]*ResBlock.expansion,
                                        out_channels_list[3], stride=2)


    def forward(self, x):
        """
        Args: 
            x: input image
        Returns:
            C2: feature maps after conv2_x
            C3: feature maps after conv3_x
            C4: feature maps after conv4_x
            C5: feature maps after conv5_x
            y: output class
        """
        x = self.conv1(x)

        print(x.size())
        # Feature maps
        C2 = self.conv2_x(x)
        print(C2.size())
        C3 = self.conv3_x(C2)
        print(C3.size())
        C4 = self.conv4_x(C3)
        print(C4.size())
        C5 = self.conv5_x(C4)
        print(C5.size())

        

        return C5


    def CreateLayer(self, ResBlock, n_blocks, in_channels, out_channels, stride=1):
        """
        Create a layer with specified type and number of residual blocks.
        Args: 
            ResBlock: residual block type, BasicBlock for ResNet-18, 34 or 
                      BottleNeck for ResNet-50, 101, 152
            n_blocks: number of residual blocks
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride used in the first 3x3 convolution of the first resdiual block
            of the layer and 1x1 convolution for skip connection in that block
        Returns: 
            Convolutional layer
        """
        layer = []
        for i in range(n_blocks):
            if i == 0:
                # Downsample the feature map using input stride for the first block of the layer.
                layer.append(ResBlock(in_channels, out_channels, 
                             stride=stride, is_first_block=True))
            else:
                # Keep the feature map size same for the rest three blocks of the layer.
                # by setting stride=1 and is_first_block=False.
                # By default, ResBlock.expansion = 4 for ResNet-50, 101, 152, 
                # ResBlock.expansion = 1 for ResNet-18, 34.
                layer.append(ResBlock(out_channels*ResBlock.expansion, out_channels))

        return nn.Sequential(*layer)


class PatchMoeResNet(nn.Module):
    def __init__(self, k, exp, patch_size, n_blocks_list=[3, 4, 6, 3],
                 out_channels_list=[64, 128, 256, 512], num_channels=3):
        """
        Args:
            ResBlock: residual block type, BasicBlock for ResNet-18, 34 or 
                      BottleNeck for ResNet-50, 101, 152
            n_class: number of classes for image classifcation (used in classfication head)
            n_block_lists: number of residual blocks for each conv layer (conv2_x - conv5_x)
            out_channels_list: list of the output channel numbers for conv2_x - conv5_x
            num_channels: the number of channels of input image
        """
        super().__init__()
        
        self.k = k
        self.exp = exp
        self.patch_size = patch_size

        ResBlock = PatchMoeBasicBlock
        
        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=num_channels, 
                                             out_channels=64, kernel_size=7,
                                             stride=2, padding=3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3,
                                                stride=2, padding=1))

        # Create four convoluiontal layers
        in_channels = 64
        # For the first block of the second layer, do not downsample and use stride=1.
        self.conv2_x = self.CreateLayer(0, ResBlock, n_blocks_list[0], 
                                        in_channels, out_channels_list[0], stride=1)
        
        # For the first blocks of conv3_x - conv5_x layers, perform downsampling using stride=2.
        # By default, ResBlock.expansion = 4 for ResNet-50, 101, 152, 
        # ResBlock.expansion = 1 for ResNet-18, 34.
        self.conv3_x = self.CreateLayer(1, ResBlock, n_blocks_list[1], 
                                        out_channels_list[0]*ResBlock.expansion,
                                        out_channels_list[1], stride=2)
        self.conv4_x = self.CreateLayer(2, ResBlock, n_blocks_list[2],
                                        out_channels_list[1]*ResBlock.expansion,
                                        out_channels_list[2], stride=2)
        self.conv5_x = self.CreateLayer(3, ResBlock, n_blocks_list[3], 
                                        out_channels_list[2]*ResBlock.expansion,
                                        out_channels_list[3], stride=2)


    def forward(self, x):
        """
        Args: 
            x: input image
        Returns:
            C2: feature maps after conv2_x
            C3: feature maps after conv3_x
            C4: feature maps after conv4_x
            C5: feature maps after conv5_x
            y: output class
        """
        x = self.conv1(x)

        # Feature maps
        C2 = self.conv2_x(x)
        C3 = self.conv3_x(C2)
        C4 = self.conv4_x(C3)
        C5 = self.conv5_x(C4)

        return C5


    def CreateLayer(self, idx, ResBlock, n_blocks, in_channels, out_channels, stride=1):
        """
        Create a layer with specified type and number of residual blocks.
        Args: 
            ResBlock: residual block type, BasicBlock for ResNet-18, 34 or 
                      BottleNeck for ResNet-50, 101, 152
            n_blocks: number of residual blocks
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride used in the first 3x3 convolution of the first resdiual block
            of the layer and 1x1 convolution for skip connection in that block
        Returns: 
            Convolutional layer
        """
        layer = []
        for i in range(n_blocks):
            if i == 0:
                # Downsample the feature map using input stride for the first block of the layer.
                layer.append(ResBlock(k = self.k[idx], 
                                      exp = self.exp[idx], 
                                      patch_size = self.patch_size[idx],
                                      in_channels = in_channels, 
                                      out_channels = out_channels, 
                                      stride=stride, is_first_block=True))
            else:
                # Keep the feature map size same for the rest three blocks of the layer.
                # by setting stride=1 and is_first_block=False.
                # By default, ResBlock.expansion = 4 for ResNet-50, 101, 152, 
                # ResBlock.expansion = 1 for ResNet-18, 34.
                layer.append(ResBlock(k = self.k[idx], 
                                      exp = self.exp[idx], 
                                      patch_size = self.patch_size[idx],
                                      in_channels = out_channels*ResBlock.expansion, 
                                      out_channels = out_channels))

        return nn.Sequential(*layer)



if __name__ == "__main__":
    ### Customed version ###
    # Resnet18
    net = ResNet(resblock='basic', n_blocks_list=[2, 2, 2, 2])
    # Resnet34
    #net = ResNet(BasicBlock, 1000)
    # Resnet50
    # net = ResNet(BottleNeck, 1000)
    # Resnet101
    #net = ResNet(BottleNeck, 1000, n_blocks_list=[3, 4, 23, 3])
    # Resnet152
    #net = ResNet(BottleNeck, 1000, n_blocks_list=[3, 8, 36, 3])
    x = torch.randn((1, 3, 64, 64), dtype=torch.float32)
    out = net(x)
    
    k = [4,4,2,2]
    exp = [8,8,4,4]
    patch_size = [2,2,2,2]
    
    net1 = PatchMoeResNet(k = k, exp = exp, patch_size=patch_size ,n_blocks_list=[2, 2, 2, 2])
    out1 = net1(x)
    ### torchvision version ###
    net_tv = models.resnet18(pretrained=False)
    
    
    print(out.size(), out1.size(),net_tv(x).size())


# from torch.nn import functional as F

# class CausalSelfAttention(nn.Module):
#     """
#     mainly copied from CausalSelfAttention
#     """

#     def __init__(self, n_embd, n_head, attn_pdrop = 0.1, resid_pdrop = 0.1):
#         super().__init__()
#         assert n_embd % n_head == 0
#         # key, query, value projections for all heads
#         self.key = nn.Linear(n_embd, n_embd)
#         self.query = nn.Linear(n_embd, n_embd)
#         self.value = nn.Linear(n_embd, n_embd)
#         # regularization
#         self.attn_drop = nn.Dropout(attn_pdrop)
#         self.resid_drop = nn.Dropout(resid_pdrop)
#         # output projection
#         self.proj = nn.Linear(n_embd, n_embd)
#         self.n_head = n_head

#     def forward(self, x):
#         (
#             B,
#             T,
#             C,
#         ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

#         # calculate query, key, values for all heads in batch and move head forward to be the batch dim
#         k = (
#             self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
#         )  # (B, nh, T, hs)
#         q = (
#             self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
#         )  # (B, nh, T, hs)
#         v = (
#             self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
#         )  # (B, nh, T, hs)

#         # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
#         att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(k.size(-1)))
#         att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
#         att = F.softmax(att, dim=-1)
#         att = self.attn_drop(att)
#         y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
#         y = (
#             y.transpose(1, 2).contiguous().view(B, T, C)
#         )  # re-assemble all head outputs side by side

#         # output projection
#         y = self.resid_drop(self.proj(y))
#         return y

# def GetFeatureMapsFromResnet(net, x):
#     """
#     Args:
#         net: network input from torchvision.
#         x: input image
#     Returns:
#         C2: feature maps after conv2_x
#         C3: feature maps after conv3_x
#         C4: feature maps after conv4_x
#         C5: feature maps after conv5_x
#     """
#     x = net.conv1(x)
#     x = net.bn1(x)
#     x = net.relu(x)
#     x = net.maxpool(x)
#     C2 = net.layer1(x)
#     C3 = net.layer2(C2)
#     C4 = net.layer3(C3)
#     C5 = net.layer4(C4)
#     return C2, C3, C4, C5


# if __name__ == "__main__":
#     ### Customed version ###
#     # Resnet18
#     net = ResNet(BasicBlock, 1000, n_blocks_list=[2, 2, 2, 2])
#     # Resnet34
#     #net = ResNet(BasicBlock, 1000)
#     # Resnet50
#     # net = ResNet(BottleNeck, 1000)
#     # Resnet101
#     #net = ResNet(BottleNeck, 1000, n_blocks_list=[3, 4, 23, 3])
#     # Resnet152
#     #net = ResNet(BottleNeck, 1000, n_blocks_list=[3, 8, 36, 3])
#     x = torch.randn((1, 3, 512, 512), dtype=torch.float32)
#     C2, C3, C4, C5, out = net(x)

#     ### torchvision version ###
#     net_tv = models.resnet18(pretrained=False)
#     #net_tv = models.resnet34(pretrained=False)
#     # net_tv = models.resnet50(pretrained=False)
#     #net_tv = models.resnet101(pretrained=False)
#     #net_tv = models.resnet152(pretrained=False)
#     C2_tv, C3_tv, C4_tv, C5_tv = GetFeatureMapsFromResnet(net_tv, x)

#     print("Verifying the feature map shapes of customed ResNet and ResNet from torchvision")
#     print(f"C2.shape of customed ResNet: {C2.shape}")
#     print(f"C2.shape of torchvision ResNet: {C2_tv.shape}")
#     print(f"C3.shape of customed ResNet: {C3.shape}")
#     print(f"C3.shape of torchvision ResNet: {C3_tv.shape}")
#     print(f"C4.shape of customed ResNet: {C4.shape}")
#     print(f"C4.shape of torchvision ResNet: {C4_tv.shape}")
#     print(f"C5.shape of customed ResNet: {C5.shape}")
#     print(f"C5.shape of torchvision ResNet: {C5_tv.shape}")
    
#     print("Done!")
    
#     print(out.size(), net_tv(x).size())