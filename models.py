import torch
import torchvision.models as models
from torch import nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.build_network()

    def build_network(self):
        self.vgg16_model = models.vgg16(pretrained=True).features
        self.vgg16_model.eval()

        self.network = nn.Sequential(
            nn.AdaptiveAvgPool2d([1,1]),
            nn.Flatten(1),
            nn.Linear(512,2)
        )
    
    def forward(self, x):
        x = self.vgg16_model(x)
        y = self.network(x)
        return y
    
class VGG16GradCAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.build_network()
        
    def build_network(self):
        # VGG16 모델 불러오기 (사전 훈련된)
        self.vgg16_model = models.vgg16(pretrained=True).features
        self.vgg16_model.eval()
        
        self.network = nn.Sequential(
            nn.AdaptiveAvgPool2d([1,1]),
            nn.Flatten(1),
            nn.Linear(512,2)
        )
        self.gradients = None
    
    # 훅(hook) 등록을 위한 함수
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.vgg16_model(x)

        # 특성 맵에 대한 그래디언트를 저장하기 위한 훅 등록
        h = x.register_hook(self.activations_hook)

        y = self.network(x)
        return y

    # Grad-CAM 처리 함수
    def generate_cam(self, input_image, class_idx):
        # 모델의 예측값과 그래디언트를 얻기 위한 forward pass
        model_output = self.forward(input_image)
        model_output[:, class_idx].backward()

        gradients = self.gradients.data
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = self.vgg16_model(input_image).data

        activations *= pooled_gradients[None, :, None, None]

        heatmap = torch.mean(activations, dim=1)
        heatmap = torch.clamp(heatmap, min=0)
        heatmap /= torch.max(heatmap)
        return heatmap
