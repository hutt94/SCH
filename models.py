import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ImgNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(ImgNet, self).__init__()
        self.vgg19 = torchvision.models.vgg19(pretrained=True)
        self.vgg19.classifier = nn.Sequential(*list(self.vgg19.classifier.children())[:6])
        self.fc_encode = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        for layer in self.vgg19.features:
            x = layer(x)

        x = x.view(x.size(0), -1)
        feat = self.vgg19.classifier(x)
        hid = self.fc_encode(feat)
        code = torch.tanh(self.alpha * hid)
        return (x, feat), hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len, image_size=4096):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc2 = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        feat1 = self.fc1(x)
        feat = F.relu(feat1)
        hid = self.fc2(feat)

        code = torch.tanh(self.alpha * hid)
        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)
