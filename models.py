import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def _get_img_type(self, img_type):
        self.img_type = img_type
    
    def _build_backbone_vgg16(self, n_img_type, vgg16_freezing):
        self.vgg16_models = nn.ModuleList()
        for _ in range(n_img_type):
            backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
            if vgg16_freezing:
                for p in backbone.parameters():
                    p.requires_grad = False
            self.vgg16_models.append(backbone)

    def _get_features(self, xs):
        x_cat = []
        for i, x in enumerate(xs):
            x = self.vgg16_models[i](x) # (B, C, H, W)
            x = F.adaptive_avg_pool2d(x, [1,1])
            x = x.squeeze()
            assert len(x.shape) == 2
            x_cat.append(x)
        x = torch.cat(x_cat, -1)
        return x

class BaseModel(BaseModule):
    def __init__(self, img_type, vgg16_freezing=True):
        super().__init__()
        self.build_network(img_type, vgg16_freezing)
       
    def build_network(self, img_type, vgg16_freezing):
        self._get_img_type(img_type)
        n_img_type = len(self.img_type)
        self._build_backbone_vgg16(n_img_type, vgg16_freezing)

        self.network = nn.Sequential(
            nn.Linear(512*n_img_type,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,2)
        )
    
    def forward(self, xs):
        x = self._get_features(xs)
        y = self.network(x)
        return y
    
class VGG16GradCAM(BaseModule):
    def __init__(self, img_type, vgg16_freezing=True):
        super().__init__()
        self.build_network(img_type, vgg16_freezing)
        
    def build_network(self, img_type, vgg16_freezing):
        self._get_img_type(img_type)
        n_img_type = len(self.img_type)
        self._build_backbone_vgg16(n_img_type, vgg16_freezing)
        
        self.network = nn.Sequential(
            nn.Linear(512*n_img_type,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,2)
        )
        
    # 훅(hook) 등록을 위한 함수
    def activations_hook(self, grad):
        self.gradients.append(grad)

    def forward(self, xs):
        x = self._get_features(xs)
        y = self.network(x)
        return y

    # Grad-CAM 처리 함수
    def generate_cam(self, xs, class_idx):
        # 모델의 예측값과 그래디언트를 얻기 위한 forward pass
        # model_output = self.forward(xs)
        x_cat = []
        self.gradients = []
        for i, x in enumerate(xs):
            x = self.vgg16_models[i](x) # (B, C, H, W)
            
            # 특성 맵에 대한 그래디언트를 저장하기 위한 훅 등록
            h = x.register_hook(self.activations_hook)
            x_cat.append(x)
        x = torch.cat(x_cat, -1)
        y = self.network(x)
        y[:, class_idx].backward()

        heatmaps = []
        for i, x in enumerate(xs):
            gradients = self.gradients[i].data
            pooled_gradients = torch.mean(gradients, dim=[2, 3])
            activations = self.vgg16_models[i](x).data

            activations *= pooled_gradients[:, :, None, None]

            heatmap = torch.mean(activations, dim=1)
            heatmap = torch.clamp(heatmap, min=0)
            heatmap /= torch.max(heatmap)
            heatmaps.append(heatmap)
        return heatmaps

class ConvAttnModel(BaseModule):
    def __init__(self, img_type, h_dim_attn, n_heads, h_dim_fc, n_layers, vgg16_freezing=True):
        super().__init__()
        self.h_dim_attn = h_dim_attn
        self.build_network(img_type, n_heads, h_dim_fc, n_layers, vgg16_freezing)
        
    def build_network(self, img_type, n_heads, h_dim_fc, n_layers, vgg16_freezing):
        self._get_img_type(img_type)
        n_img_type = len(self.img_type)
        self._build_backbone_vgg16(n_img_type, vgg16_freezing)
        vgg_out_channels = self.vgg16_models[-1][-3].out_channels
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.h_dim_attn))
        
        self.conv_1x1s, self.attns = nn.ModuleList(), nn.ModuleList()
        for _ in range(n_img_type):
            self.conv_1x1s.append(nn.Conv2d(vgg_out_channels, self.h_dim_attn, 1))

            enc_layer = nn.TransformerEncoderLayer(self.h_dim_attn, n_heads, h_dim_fc, batch_first=True)    
            self.attns.append(nn.TransformerEncoder(enc_layer, n_layers))
        
        self.network = nn.Sequential(
            nn.Linear(self.h_dim_attn*n_img_type, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
    
    def forward(self, xs):
        x_cat = []
        for i,x in enumerate(xs):
            # cls_token = torch.randn(x.shape[0], 1, self.h_dim_attn).to(x) #(B, 1, D)
            # cls_token = torch.zeros(x.shape[0], 1, self.h_dim_attn).to(x) #(B, 1, D)
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            
            x_tilde = self.vgg16_models[i](x) # (B, C, H, W)
            x_tilde = self.conv_1x1s[i](x_tilde) # (B, D, H, W)
            
            x_tilde_reshaped = x_tilde.flatten(2).permute(0,2,1) #(B, HW, D)
            
            cls_x_tilde = torch.cat([cls_token, x_tilde_reshaped], 1) # (B, HW+1, D)
            x_attn = self.attns[i](cls_x_tilde) # (B, HW+1, D)

            x = x_attn[:,0] #(B, D)
            x_cat.append(x)
        x = torch.cat(x_cat, -1)
        y = self.network(x)
        return y
