import torch
import copy

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
IMG_HEIGHT = 64
def replace_bn_with_gn(module, num_groups=32):
    """
    Recursively replaces all BatchNorm2d layers with GroupNorm.
    Essential for stabilizing training on low batch sizes on home GPUs.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            gn = nn.GroupNorm(
                num_groups=min(num_groups, num_channels), 
                num_channels=num_channels
            )
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups)
    return module
class CRNN(nn.Module):
    def __init__(self, num_classes, img_height=IMG_HEIGHT, in_channels=1, freeze_cnn=False):
        super(CRNN, self).__init__()
        self.normalize = transforms.Normalize(mean=[0.5] * in_channels, std=[0.5] * in_channels)
        self.img_height = img_height
        self.in_channels = in_channels
        self.num_classes = num_classes

        resnet = models.resnet50(pretrained=True)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        
        if freeze_cnn:
            for name, param in self.cnn.named_parameters():
                if "layer4" not in name:
                    param.requires_grad = False

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_height, 100)
            dummy_output = self.cnn(dummy)
            _, c, h, w = dummy_output.size()
            self.feature_dim = c  

        self.lstm1 = nn.LSTM(self.feature_dim, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.normalize(x)
        x = self.cnn(x)
        x = F.adaptive_avg_pool2d(x, (1, None))       
        x = x.squeeze(2).permute(0, 2, 1) 
  
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)
        return x
# Your model weight-averaging trick lives safely inside the model tools now
def merge_best_task_layers(global_model, best_dicts, alphas=None, merge_cnn5=False):
    """
    Blends recurrent and linear head weights from separate dataset training runs.
    """
    if alphas is None:
        alphas = [1.0 / len(best_dicts)] * len(best_dicts)

    with torch.no_grad():  
        for name, param in global_model.named_parameters():
            if ("lstm" in name) or ("fc" in name) or (merge_cnn5 and "cnn.5" in name):
                merged = sum(alpha * bd[name] for alpha, bd in zip(alphas, best_dicts))
                param.copy_(merged)   
    return global_model

def merge_best_states(global_model, best_dicts, alphas=None):
    """
    global_model: the model to load merged weights into
    best_dicts: list of state_dicts from different dataloaders [dict1, dict2, ...]
    alphas: list of weights for each state_dict (must sum to 1)
    """
    if alphas is None:
        alphas = [1.0 / len(best_dicts)] * len(best_dicts)  # equal weighting

    # start from global_model state_dict
    merged_state = copy.deepcopy(global_model.state_dict())

    # merge each parameter
    for key in merged_state.keys():
        merged_state[key] = sum(alpha * bd[key] for alpha, bd in zip(alphas, best_dicts))

    # load merged weights into global_model
    global_model.load_state_dict(merged_state)
    return global_model





