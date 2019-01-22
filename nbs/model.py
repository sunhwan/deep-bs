import torch
from torch import nn

class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv3d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv3d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv3d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class SqueezeNet(nn.Module):
    def __init__(self, input_nc):
        super().__init__()
        output_nc = 64
        features = [nn.Conv3d(input_nc, output_nc, kernel_size=7, stride=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(64, 16, 64, 64),
                    Fire(128, 16, 64, 64),
                    nn.MaxPool3d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(128, 32, 128, 128),
                    Fire(256, 32, 128, 128),
                    nn.MaxPool3d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(256, 48, 192, 192),
                    Fire(384, 48, 192, 192),
                    Fire(384, 64, 256, 256),
                    Fire(512, 64, 256, 256)]
        
        head = [Flatten(),
                nn.Dropout(p=0.5),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(128),
                nn.Dropout(p=0.5),
                nn.Linear(128, 1)
                ]
        
        self.features = nn.Sequential(*features)
        self.head = nn.Sequential(*head)
    
    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x
