import os
import datetime

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import torchsummary
import torchinfo



## ****************************************************************************
## HELPER CLASSES                                                             *
## ****************************************************************************
class FireModule(nn.Module):
    def __init__(self, in_channels, s_channels, e1_channels, e2_channels):
        super(FireModule, self).__init__()
        """
        The Fire Module is described with the following parameters: Number of 
        input channels of module C, the number of output channels of the 1x1 
        squeeze layer S, the number of output channels of 1x1 expand layer E1,
        and the number of output channels of the 3x3 expand layer E2, resulting
        in (C, S, E1, E2)
        """
        self.squeeze = nn.Conv2d(in_channels, s_channels, kernel_size=1)
        self.expand1 = nn.Conv2d(s_channels, e1_channels, kernel_size=1)
        self.expand2 = nn.Conv2d(s_channels, e2_channels, kernel_size=3,
        padding=1)

    def forward(self, x):
        x = F.relu(self.squeeze(x))
        return torch.cat([F.relu(self.expand1(x)), F.relu(self.expand2(x))], 1)



class FireModuleBN(nn.Module):
    def __init__(self, in_channels, s_channels, e1_channels, e2_channels):
        super(FireModuleBN, self).__init__()
        """
        The Fire Module is described with the following parameters: Number of 
        input channels of module C, the number of output channels of the 1x1 
        squeeze layer S, the number of output channels of 1x1 expand layer E1,
        and the number of output channels of the 3x3 expand layer E2, resulting
        in (C, S, E1, E2)
        """
        self.squeeze = nn.Conv2d(in_channels, s_channels, kernel_size=1)
        self.bn_squeeze = nn.BatchNorm2d(s_channels)
        self.expand1 = nn.Conv2d(s_channels, e1_channels, kernel_size=1)
        self.bn_expand1 = nn.BatchNorm2d(e1_channels)
        self.expand2 = nn.Conv2d(s_channels, e2_channels, kernel_size=3, padding=1)
        self.bn_expand2 = nn.BatchNorm2d(e2_channels)

    def forward(self, x):
        x = F.relu(self.bn_squeeze(self.squeeze(x)))
        return torch.cat([F.relu(self.bn_expand1(self.expand1(x))), F.relu(self.bn_expand2(self.expand2(x)))], 1)
    


class DepthSepConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthSepConv, self).__init__()
        """
        Depthwise Separable Convolution consists of a depthwise convolution, 
        which applies a single filter per input channel, and a pointwise 
        convolution, which applies a 1x1 convolution to combine the outputs 
        from the depthwise convolution.

        Parameters:
        - in_channels: Number of input channels
        - out_channels: Number of output channels
        """
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                   padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   bias=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = F.relu(x)
        x = self.pointwise(x)
        x = F.relu(x)
        return x
    


class DepthSepConvBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthSepConvBN, self).__init__()
        """
        Depthwise Separable Convolution consists of a depthwise convolution, 
        which applies a single filter per input channel, and a pointwise 
        convolution, which applies a 1x1 convolution to combine the outputs 
        from the depthwise convolution.

        Parameters:
        - in_channels: Number of input channels
        - out_channels: Number of output channels
        """
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                   padding=1, groups=in_channels, bias=False)
        self.bn_depth = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   bias=True)
        self.bn_point = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_depth(x)
        x = F.relu(x)
        x = self.pointwise(x)
        x = self.bn_point(x)
        x = F.relu(x)
        return x



class ResidualBlock(nn.Module):
    """
    A simple residual block with two linear layers and a skip connection.
    """
    def __init__(self, in_features, out_features, activation=nn.LeakyReLU(0.01)):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.activation = activation
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        # Adapting input if necessary
        self.adapt = nn.Linear(in_features, out_features) if in_features != out_features else None
        
        # Apply He initialization
        torch.nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='leaky_relu')
        if self.adapt:
            torch.nn.init.kaiming_normal_(self.adapt.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.bn2(out)
        
        if self.adapt:
            identity = self.adapt(identity)
        
        out += identity
        out = self.activation(out)
        return out
    


## ****************************************************************************
## CNN1                                                                       *
## ****************************************************************************
class CNN1(nn.Module):
    def __init__(self, selected_channels: list, input_size, num_classes=2):
        super(CNN1, self).__init__()

        self.selected_channels = selected_channels
        self.num_classes = num_classes

        num_in_channels = len(self.selected_channels)

        self.conv1 = nn.Conv2d(num_in_channels, 32, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)

        # Use dummy input to pass through the conv layers to determine output size
        with torch.no_grad():
            d_input = torch.zeros(1, num_in_channels, 50, 50)
            d_output = self.conv4(self.conv3(self.conv2(self.conv1(d_input))))
            self.flattened_size = d_output.data.view(1, -1).size(1)
        
        # Layer 5 (Dense)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, self.num_classes)

    def forward(self, x):        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Flatten the output for the dense layer
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer
        return x




class CNN1BN(nn.Module):
    """ CNN1 with Batch Normalization. """
    def __init__(self, selected_channels: list, input_size, num_classes=2):
        super(CNN1BN, self).__init__()

        self.selected_channels = selected_channels
        self.num_classes = num_classes

        num_in_channels = len(self.selected_channels)

        self.conv1 = nn.Conv2d(num_in_channels, 32, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(64)

        # Use dummy input to pass through the conv layers to determine output size
        with torch.no_grad():
            d_input = torch.zeros(1, num_in_channels, input_size, input_size)
            d_output = self.conv4(self.conv3(self.conv2(self.conv1(d_input))))
            self.flattened_size = d_output.data.view(1, -1).size(1)
        
        # Layer 5 (Dense)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, self.num_classes)

    def forward(self, x):        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Flatten the output for the dense layer
        x = self.flatten(x)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.fc2(x)  # Output layer
        return x


## ****************************************************************************
## CNN2                                                                       *
## ****************************************************************************
class CNN2(nn.Module):
    def __init__(self, selected_channels: list, input_size, num_classes=2):
        super(CNN2, self).__init__()

        self.selected_channels = selected_channels
        self.num_classes = num_classes

        num_in_channels = len(self.selected_channels)

        self.conv1 = nn.Conv2d(num_in_channels, 32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)

        # Max pooling layer 1 has to have kernel_size=3 to match TensorFlow model.
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # Use dummy input to pass through conv layers to determine output size.
        with torch.no_grad():
            d_input = torch.zeros(1, num_in_channels, input_size, input_size)

            x = self.maxpool1(self.conv1(d_input))
            x = self.maxpool2(self.conv2(x))
            x = self.maxpool3(self.conv3(x))
            x = self.maxpool4(self.conv4(x))

            self.flattened_size = x.data.view(1, -1).size(1)


        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, self.num_classes)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)
        x = F.relu(self.conv4(x))
        x = self.maxpool4(x)

        # Flatten output for dense layer.
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




class CNN2BN(nn.Module):
    def __init__(self, selected_channels: list, input_size, num_classes=2):
        super(CNN2BN, self).__init__()

        self.selected_channels = selected_channels
        self.num_classes = num_classes

        num_in_channels = len(self.selected_channels)

        self.conv1 = nn.Conv2d(num_in_channels, 32, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(64)

        # Max pooling layer 1 has to have kernel_size=3 to match TensorFlow model.
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # Use dummy input to pass through conv layers to determine output size.
        with torch.no_grad():
            d_input = torch.zeros(1, num_in_channels, input_size, input_size)

            x = self.maxpool1(self.bn1(self.conv1(d_input)))
            x = self.maxpool2(self.bn2(self.conv2(x)))
            x = self.maxpool3(self.bn3(self.conv3(x)))
            x = self.maxpool4(self.bn4(self.conv4(x)))

            self.flattened_size = x.data.view(1, -1).size(1)


        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, self.num_classes)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxpool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.maxpool4(x)

        # Flatten output for dense layer.
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

## ****************************************************************************
## CNN3                                                                     *
## ****************************************************************************
class CNN3(nn.Module):
    def __init__(self, selected_channels: list, input_size, num_classes=2):
        super(CNN3, self).__init__()

        self.selected_channels = selected_channels
        self.num_classes = num_classes

        # Number of input channels is the length of selected_channels
        num_in_channels = len(self.selected_channels)

        # Depthwise + Pointwise Convolution 1
        # NOTE: To match the TensorFlow model, bias has to be set to False in
        # the depthwise layers! This is not the default in PyTorch.
        self.depthwise_conv1 = nn.Conv2d(num_in_channels, num_in_channels, kernel_size=3, stride=1, padding=1, groups=num_in_channels, bias=False)
        self.pointwise_conv1 = nn.Conv2d(num_in_channels, 32, kernel_size=1, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Depthwise + Pointwise Convolution 2
        self.depthwise_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False)
        self.pointwise_conv2 = nn.Conv2d(32, 48, kernel_size=1, bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # Depthwise + Pointwise Convolution 3
        self.depthwise_conv3 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, groups=48, bias=False)
        self.pointwise_conv3 = nn.Conv2d(48, 64, kernel_size=1, bias=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # Depthwise + Pointwise Convolution 4
        self.depthwise_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
        self.pointwise_conv4 = nn.Conv2d(64, 64, kernel_size=1, bias=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # Use dummy input to pass through conv layers to determine output size.
        with torch.no_grad():
            d_input = torch.zeros(1, num_in_channels, 50, 50)

            # Pass through layers.
            x = self.maxpool1(self.pointwise_conv1(self.depthwise_conv1(d_input)))
            x = self.maxpool2(self.pointwise_conv2(self.depthwise_conv2(x)))
            x = self.maxpool3(self.pointwise_conv3(self.depthwise_conv3(x)))
            x = self.maxpool4(self.pointwise_conv4(self.depthwise_conv4(x)))

            self.flattened_size = x.data.view(1, -1).size(1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, num_classes)


    def forward(self, x):
        x = F.relu(self.pointwise_conv1(self.depthwise_conv1(x)))
        x = self.maxpool1(x)
        x = F.relu(self.pointwise_conv2(self.depthwise_conv2(x)))
        x = self.maxpool2(x)
        x = F.relu(self.pointwise_conv3(self.depthwise_conv3(x)))
        x = self.maxpool3(x)
        x = F.relu(self.pointwise_conv4(self.depthwise_conv4(x)))
        x = self.maxpool4(x)
        # Flatten output for dense layer.
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x



class CNN3BN(nn.Module):
    def __init__(self, selected_channels: list, input_size, num_classes=2):
        super(CNN3BN, self).__init__()

        self.selected_channels = selected_channels
        self.num_classes = num_classes

        num_in_channels = len(self.selected_channels)

        # Initialize Depthwise Separable Convolution blocks
        self.depth_sep_conv1 = DepthSepConvBN(num_in_channels, 32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.depth_sep_conv2 = DepthSepConvBN(32, 48)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.depth_sep_conv3 = DepthSepConvBN(48, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.depth_sep_conv4 = DepthSepConvBN(64, 64)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # Use dummy input to pass through conv blocks to determine output size
        with torch.no_grad():
            d_input = torch.zeros(1, num_in_channels, 50, 50)
            x = self.depth_sep_conv1(d_input)
            x = self.maxpool1(x)
            x = self.depth_sep_conv2(x)
            x = self.maxpool2(x)
            x = self.depth_sep_conv3(x)
            x = self.maxpool3(x)
            x = self.depth_sep_conv4(x)
            x = self.maxpool4(x)
            self.flattened_size = x.data.view(1, -1).size(1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.depth_sep_conv1(x)
        x = self.maxpool1(x)
        x = self.depth_sep_conv2(x)
        x = self.maxpool2(x)
        x = self.depth_sep_conv3(x)
        x = self.maxpool3(x)
        x = self.depth_sep_conv4(x)
        x = self.maxpool4(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
## ****************************************************************************
## CNN4                                                                       *
## ****************************************************************************
class CNN4(nn.Module):
    def __init__(self, selected_channels: list, input_size, num_classes=2):
        super(CNN4, self).__init__()

        self.selected_channels = selected_channels
        self.num_classes = num_classes

        # Number of input channels is the length of selected_channels
        num_in_channels = len(self.selected_channels)

        # Depthwise + Pointwise Convolution
        # NOTE: To match the TensorFlow model, bias needs to be set to False in
        # the depthwise layers! This is not the default in PyTorch for Conv2d.
        self.depthwise_conv1 = nn.Conv2d(num_in_channels, num_in_channels, kernel_size=3, stride=1, padding=1, groups=num_in_channels, bias=False)
        self.pointwise_conv1 = nn.Conv2d(num_in_channels, 32, kernel_size=1, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.depthwise_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False)
        self.pointwise_conv2 = nn.Conv2d(32, 48, kernel_size=1, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.depthwise_conv3 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, groups=48, bias=False)
        self.pointwise_conv3 = nn.Conv2d(48, 64, kernel_size=1, bias=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.depthwise_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
        self.pointwise_conv4 = nn.Conv2d(64, 64, kernel_size=1, bias=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.depthwise_conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
        self.pointwise_conv5 = nn.Conv2d(64, num_classes, kernel_size=1, bias=True)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Flatten layer.
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = F.relu(self.pointwise_conv1(self.depthwise_conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.pointwise_conv2(self.depthwise_conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.pointwise_conv3(self.depthwise_conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.pointwise_conv4(self.depthwise_conv4(x)))
        x = self.pool4(x)
        x = F.relu(self.pointwise_conv5(self.depthwise_conv5(x)))
        x = self.global_avg_pool(x)
        x = self.flatten(x)

        return x


class CNN4BN(nn.Module):
    def __init__(self, selected_channels: list, input_size, num_classes=2):
        super(CNN4BN, self).__init__()

        self.selected_channels = selected_channels
        self.num_classes = num_classes

        num_in_channels = len(self.selected_channels)

        # Initialize Depthwise Separable Convolution blocks
        self.depth_sep_conv1 = DepthSepConvBN(num_in_channels, 32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.depth_sep_conv2 = DepthSepConvBN(32, 48)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.depth_sep_conv3 = DepthSepConvBN(48, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.depth_sep_conv4 = DepthSepConvBN(64, 64)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.depth_sep_conv5 = DepthSepConvBN(64, self.num_classes)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Flatten layer.
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = self.depth_sep_conv1(x)
        x = self.maxpool1(x)
        x = self.depth_sep_conv2(x)
        x = self.maxpool2(x)
        x = self.depth_sep_conv3(x)
        x = self.maxpool3(x)
        x = self.depth_sep_conv4(x)
        x = self.maxpool4(x)
        x = self.depth_sep_conv5(x)
        
        x = self.global_avg_pool(x)
        x = self.flatten(x)

        return x

## ****************************************************************************
## SpectrumNet                                                                *
## ****************************************************************************
class SpectrumNet(nn.Module):
    def __init__(self, selected_channels: list, input_size, num_classes=2):
        super(SpectrumNet, self).__init__()

        self.selected_channels = selected_channels
        self.num_classes = num_classes
        num_in_channels = len(self.selected_channels)

        # Layer 1 - Initial Convolution
        self.conv1 = nn.Conv2d(num_in_channels, 96, kernel_size=2, stride=1, padding=1)

        # Layer 2 to 9 - Fire Modules
        self.fire2 = FireModule(96, 16, 96, 32)
        self.fire3 = FireModule(128, 16, 96, 32)
        self.fire4 = FireModule(128, 32, 192, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fire5 = FireModule(256, 32, 192, 64)
        self.fire6 = FireModule(256, 48, 288, 96)
        self.fire7 = FireModule(384, 48, 288, 96)
        self.fire8 = FireModule(384, 64, 384, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fire9 = FireModule(512, 64, 384, 128)

        # Layer 12 - Final Convolution
        self.conv2 = nn.Conv2d(512, num_classes, kernel_size=1)

        # Layer 13 - Global Average Pooling (GAP)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool1(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool2(x)
        x = self.fire9(x)
        x = self.conv2(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        return x



class SpectrumNetBN(nn.Module):
    def __init__(self, selected_channels: list, input_size, num_classes=2):
        super(SpectrumNetBN, self).__init__()

        self.selected_channels = selected_channels
        self.num_classes = num_classes
        num_in_channels = len(self.selected_channels)

        # Layer 1 - Initial Convolution
        self.conv1 = nn.Conv2d(num_in_channels, 96, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(96)

        # Layer 2 to 9 - Fire Modules
        self.fire2 = FireModuleBN(96, 16, 96, 32)
        self.fire3 = FireModuleBN(128, 16, 96, 32)
        self.fire4 = FireModuleBN(128, 32, 192, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fire5 = FireModuleBN(256, 32, 192, 64)
        self.fire6 = FireModuleBN(256, 48, 288, 96)
        self.fire7 = FireModuleBN(384, 48, 288, 96)
        self.fire8 = FireModuleBN(384, 64, 384, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fire9 = FireModuleBN(512, 64, 384, 128)

        # Layer 12 - Final Convolution
        self.conv2 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(num_classes)

        # Layer 13 - Global Average Pooling (GAP)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool1(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool2(x)
        x = self.fire9(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        return x

## ****************************************************************************
## SpectrumNetDS                                                              *
## ****************************************************************************
class SpectrumNetDS(nn.Module):
    """
    SpectrumNetDS introduces depthwise separable convolutions to SpectrumNet,
    which replace the two basic convolutions.
    """
    def __init__(self, selected_channels: list, input_size, num_classes=2):
        super(SpectrumNetDS, self).__init__()

        self.selected_channels = selected_channels
        self.num_classes = num_classes

        num_in_channels = len(self.selected_channels)

        # Layer 1 - Initial Convolution
        self.depth_sep_conv1 = DepthSepConv(num_in_channels, 96)

        # Layer 2 to 9 - Fire Modules
        self.fire2 = FireModule(96, 16, 96, 32)
        self.fire3 = FireModule(128, 16, 96, 32)
        self.fire4 = FireModule(128, 32, 192, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fire5 = FireModule(256, 32, 192, 64)
        self.fire6 = FireModule(256, 48, 288, 96)
        self.fire7 = FireModule(384, 48, 288, 96)
        self.fire8 = FireModule(384, 64, 384, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fire9 = FireModule(512, 64, 384, 128)

        # Layer 12 - Final Convolution
        self.depth_sep_conv2 = DepthSepConv(512, num_classes)
        
        # Layer 13 - Global Average Pooling (GAP)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.depth_sep_conv1(x))
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool1(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool2(x)
        x = self.fire9(x)
        x = F.relu(self.depth_sep_conv2(x))
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        return x
    

class SpectrumNetDSBN(nn.Module):
    """
    SpectrumNetDS introduces depthwise separable convolutions to SpectrumNet,
    which replace the two basic convolutions.
    """
    def __init__(self, selected_channels: list, input_size, num_classes=2):
        super(SpectrumNetDSBN, self).__init__()

        self.selected_channels = selected_channels
        self.num_classes = num_classes

        num_in_channels = len(self.selected_channels)

        # Layer 1 - Initial Convolution
        self.depth_sep_conv1 = DepthSepConvBN(num_in_channels, 96)

        # Layer 2 to 9 - Fire Modules
        self.fire2 = FireModuleBN(96, 16, 96, 32)
        self.fire3 = FireModuleBN(128, 16, 96, 32)
        self.fire4 = FireModuleBN(128, 32, 192, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fire5 = FireModuleBN(256, 32, 192, 64)
        self.fire6 = FireModuleBN(256, 48, 288, 96)
        self.fire7 = FireModuleBN(384, 48, 288, 96)
        self.fire8 = FireModuleBN(384, 64, 384, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fire9 = FireModuleBN(512, 64, 384, 128)

        # Layer 12 - Final Convolution
        self.depth_sep_conv2 = DepthSepConvBN(512, num_classes)
        
        # Layer 13 - Global Average Pooling (GAP)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.depth_sep_conv1(x))
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool1(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool2(x)
        x = self.fire9(x)
        x = F.relu(self.depth_sep_conv2(x))
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        return x
    


## ****************************************************************
## Pretrained Models                                              *
## ****************************************************************
class EfficientNetB3(nn.Module):
    def __init__(self, selected_channels: list, input_size, num_classes=2):
        super(EfficientNetB3, self).__init__()

        self.num_classes = num_classes

        # Load EfficientNet-B3 model with pre-trained weights from ImageNet
        weights = models.EfficientNet_B3_Weights.DEFAULT
        self.model = models.efficientnet_b3(weights=weights)

        # Freeze all parameters in the network
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(1536, 128), 
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3, inplace=False), 
            nn.Linear(128, self.num_classes) 
        )

        self._initialize_weights()
        
    def forward(self, x):
        x = self.model(x)
        return x
    
    def _initialize_weights(self):
        for m in self.model.classifier:
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    

class EfficientNetB3_Simple(nn.Module):
    def __init__(self, selected_channels: list, input_size, num_classes=2):
        super(EfficientNetB3_Simple, self).__init__()

        self.num_classes = num_classes

        # Load EfficientNet-B3 model with pre-trained weights from ImageNet
        weights = models.EfficientNet_B3_Weights.DEFAULT
        self.model = models.efficientnet_b3(weights=weights)

        # Freeze all parameters in the network
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(1536, self.num_classes)
        )

        self._initialize_weights()

        
    def forward(self, x):
        x = self.model(x)
        return x
    
    def _initialize_weights(self):
        for m in self.model.classifier:
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)




class EfficientNetB3_Complex(nn.Module):
    def __init__(self, selected_channels: list, input_size, num_classes=2):
        super(EfficientNetB3_Complex, self).__init__()
        self.num_classes = num_classes

        # Load EfficientNet-B3 model with pre-trained weights from ImageNet
        weights = models.EfficientNet_B3_Weights.DEFAULT
        self.model = models.efficientnet_b3(weights=weights)

        # Freeze all parameters in the network
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(1536, 512),
            nn.LeakyReLU(0.01, inplace=True),
            nn.BatchNorm1d(512),
            ResidualBlock(512, 256),
            nn.Dropout(0.3, inplace=False),
            ResidualBlock(256, 128),
            nn.Dropout(0.3, inplace=False),
            nn.Linear(128, self.num_classes)
        )
        
        self._initialize_weights()

    def forward(self, x):
        x = self.model(x)
        return x

    def _initialize_weights(self):
        for m in self.model.classifier:
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, ResidualBlock):
                pass


class EfficientNetB3_Blank(nn.Module):
    def __init__(self):
        super(EfficientNetB3_Blank, self).__init__()

        # Load EfficientNet-B3 model with pre-trained weights from ImageNet
        weights = models.EfficientNet_B3_Weights.DEFAULT
        self.model = models.efficientnet_b3(weights=weights)

        # Remove the classifier
        self.model.classifier = nn.Identity()

    def forward(self, x):
        x = self.model(x)
        return x
    


def check_frozen_layers(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"NOT frozen: {name}")
        else:
            print(f"frozen: {name}")




if __name__ == '__main__':
    """ Create a model with the selected channels and print. """

    selected_channels = range(0, 717)

    # cnn = EfficientNetB3_Complex(selected_channels, input_size=300)

    # unfreeze_prefixes = (
    #     # "model.features.0",
    #     # "model.features.1",
    #     # "model.features.2",
    #     # "model.features.3",
    #     "model.features.4",
    #     "model.features.5",
    #     "model.features.6",
    #     "model.features.7",
    #     "model.features.8",
    #     "model.classifier"
    # )

    # # Iterate through each parameter and explicitly set requires_grad
    # for name, parameter in cnn.named_parameters():
    #     if any(name.startswith(prefix) for prefix in unfreeze_prefixes):
    #         parameter.requires_grad = True
    #     else:
    #         # Explicitly freeze layers that don't match the unfreeze prefixes
    #         parameter.requires_grad = False

    
    cnn = CNN4BN(list(selected_channels), 50)
    input_size = (1, len(selected_channels), 50, 50)
    torchinfo.summary(cnn, input_size)

    #####################
    # Count parameters in model:
    def count_parameters_adjusted(model):
        trainable_params = 0
        non_trainable_params = 0

        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                # Count all parameters in BatchNorm2d layers as non-trainable
                non_trainable_params += sum(p.numel() for p in module.parameters())
            elif isinstance(module, nn.BatchNorm1d):
                # Count all parameters in BatchNorm1d layers as non-trainable
                non_trainable_params += sum(p.numel() for p in module.parameters())
            else:
                # For other layers, classify parameters based on requires_grad
                for param in module.parameters(recurse=False):
                    if param.requires_grad:
                        trainable_params += param.numel()
                    else:
                        non_trainable_params += param.numel()
                        
        return {"trainable": trainable_params, "non-trainable": non_trainable_params}

    print(count_parameters_adjusted(cnn))