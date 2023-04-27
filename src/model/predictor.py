import torch
import torch.nn as nn

class CNNPredictor(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNNPredictor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(output_channels, 512, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = x.view(1, 4, 1, 512)  # Reshape the input tensor to be compatible with the convolutional layer
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return x



