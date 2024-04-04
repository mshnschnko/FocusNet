from torch import nn
import torch
from torchvision.models.mobilenetv3 import mobilenet_v3_small

class FocusNet(nn.Module):
    def __init__(self):
        super(FocusNet, self).__init__()
        self.model = mobilenet_v3_small(prretrained=True)
        self.model.classifier = nn.Identity()
        
        self.regressor = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        
    def forward(self, x):
        x = self.model(x)
        return self.regressor(x)
        
    

if __name__ == '__main__':
    model = FocusNet()
    input_tensor = torch.randn(1, 3, 672, 672)
    print(model(input_tensor))