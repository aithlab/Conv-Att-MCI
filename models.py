import torchvision.models as models
from torch import nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.build_network()

    def build_network(self):

        self.network = nn.Sequential(
            models.vgg16(pretrained=True).features,
            nn.AdaptiveAvgPool2d([1,1]),
            nn.Flatten(1),
            nn.Linear(512,2)
        )
    
    def forward(self, x):
        y = self.network(x)
        return y