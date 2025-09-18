import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class VGG16Decoder(nn.Module):
    def __init__(self, pretrained=True, use_bilinear=False):
        super().__init__()
        vgg = models.vgg16(pretrained=pretrained)
        features = list(vgg.features.children())
        # exclude the final maxpool (the last MaxPool2d in features)
        # vgg.features has 30 layers typically; we will keep all features
        self.features = nn.Sequential(*features)  # we'll use the full features and then decode
        self.use_bilinear = use_bilinear
        # Simple decoder using convtranspose
        # reduce channels gradually: VGG final conv has 512 channels
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
    def forward(self, x):
        h = self.features(x)  # downsampled features
        if self.use_bilinear:
            # simple bilinear upsample from 512 -> HxW (assumes downscale factor of 32)
            out = F.interpolate(h, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            out = nn.Conv2d(h.shape[1], 1, kernel_size=1)(out)  # 1x1 conv on the fly
            return out
        else:
            out = self.decoder(h)
            # ensure size match
            if out.shape[2:] != x.shape[2:]:
                out = F.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            return out
