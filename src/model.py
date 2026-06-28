import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
IMG_HEIGHT = 64

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
