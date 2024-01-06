import torch
import torchvision.models as models
from torch import nn

class BaseModel(nn.Module):
    def __init__(self, img_type, vgg16_freezing=True):
        super().__init__()
        self.build_network(img_type, vgg16_freezing)

    def build_network(self, img_type, vgg16_freezing):
        n_img_type = len(img_type)
        self.vgg16_models = nn.ModuleList()
        for _ in range(n_img_type):
            self.vgg16_models.append(models.vgg16(pretrained=True).features)
            if vgg16_freezing:
                self.vgg16_models[-1].eval()

        self.network = nn.Sequential(
            nn.AdaptiveAvgPool2d([1,1]),
            nn.Flatten(1),
            nn.Linear(512,2)
        )
    
    def forward(self, xs):
        x_cat = []
        for i, x in enumerate(xs):
            x = self.vgg16_models[i](x) # (B, C, H, W)
            x_cat.append(x)
        x = torch.cat(x_cat, -1)
        y = self.network(x)
        return y
    
class VGG16GradCAM(nn.Module):
    def __init__(self, img_type, vgg16_freezing=True):
        super().__init__()
        self.build_network(img_type, vgg16_freezing)
        
    def build_network(self, img_type, vgg16_freezing):
        n_img_type = len(img_type)
        self.vgg16_models = nn.ModuleList()
        for _ in range(n_img_type):
            self.vgg16_models.append(models.vgg16(pretrained=True).features)
            if vgg16_freezing:
                self.vgg16_models[-1].eval()
        
        self.network = nn.Sequential(
            nn.AdaptiveAvgPool2d([1,1]),
            nn.Flatten(1),
            nn.Linear(512,2)
        )
        self.gradients = None
    
    # 훅(hook) 등록을 위한 함수
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, xs):
        x_cat = []
        for i, x in enumerate(xs):
            x = self.vgg16_models[i](x) # (B, C, H, W)
            
            # 특성 맵에 대한 그래디언트를 저장하기 위한 훅 등록
            h = x.register_hook(self.activations_hook)
            x_cat.append(x)
        x = torch.cat(x_cat, -1)
        y = self.network(x)
        return y

    # Grad-CAM 처리 함수
    def generate_cam(self, input_image, class_idx):
        # 모델의 예측값과 그래디언트를 얻기 위한 forward pass
        model_output = self.forward(input_image)
        model_output[:, class_idx].backward()

        gradients = self.gradients.data
        pooled_gradients = torch.mean(gradients, dim=[2, 3])
        activations = self.vgg16_model(input_image).data

        activations *= pooled_gradients[:, :, None, None]

        heatmap = torch.mean(activations, dim=1)
        heatmap = torch.clamp(heatmap, min=0)
        heatmap /= torch.max(heatmap)
        return heatmap

class ConvAttnModel(nn.Module):
    def __init__(self, img_type, h_dim_attn, n_heads, h_dim_fc, n_layers, vgg16_freezing=True):
        super().__init__()
        self.h_dim_attn = h_dim_attn
        self.build_network(img_type, n_heads, h_dim_fc, n_layers, vgg16_freezing)
        
    def build_network(self, img_type, n_heads, h_dim_fc, n_layers, vgg16_freezing):
        n_img_type = len(img_type)
        
        self.vgg16_models, self.conv_1x1s, self.attns = nn.ModuleList(),nn.ModuleList(),nn.ModuleList()
        
        for _ in range(n_img_type):
            self.vgg16_models.append(models.vgg16(pretrained=True).features)
            if vgg16_freezing:
                self.vgg16_models[-1].eval()
            vgg_out_channels = self.vgg16_models[-1][-3].out_channels
            self.conv_1x1s.append(nn.Conv2d(vgg_out_channels, self.h_dim_attn, 1))

            enc_layer = nn.TransformerEncoderLayer(self.h_dim_attn, n_heads, h_dim_fc, batch_first=True)    
            self.attns.append(nn.TransformerEncoder(enc_layer, n_layers))

        self.network = nn.Sequential(
            nn.Linear(self.h_dim_attn*n_img_type, 2)
        )
    
    def forward(self, xs):
        x_cat = []
        for i,x in enumerate(xs):
            x_tilde = self.vgg16_models[i](x) # (B, C, H, W)
            x_tilde = self.conv_1x1s[i](x_tilde) # (B, D, H, W)

            _x_tilde = x_tilde.flatten(2).permute(0,2,1) #(B, HW, D)
            cls_rand_token = torch.randn(_x_tilde.shape[0], 1, self.h_dim_attn).to(_x_tilde) #(B, 1, D)
            # cls_rand_token = torch.zeros(_x_tilde.shape[0], 1, self.h_dim_attn).to(_x_tilde) #(B, 1, D)
            cls_x_tilde = torch.cat([cls_rand_token, _x_tilde], 1) # (B, HW+1, D)
            x_attn = self.attns[i](cls_x_tilde) # (B, HW+1, D)

            x = x_attn[:,0] #(B, D)
            x_cat.append(x)
        x = torch.cat(x_cat, -1)
        y = self.network(x)
        return y
