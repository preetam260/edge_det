import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class HEDLike(nn.Module):
    """
    Extract side outputs before each maxpool in VGG and produce side-output maps.
    Then fuse them using learnable weights.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        vgg = models.vgg16(pretrained=pretrained)
        features = list(vgg.features.children())

        # indices of layers where MaxPool occurs in vgg16.features: 4, 9, 16, 23, 30 (0-index)
        pool_indices = [4, 9, 16, 23, 30]
        self.stages = nn.ModuleList()
        prev = 0
        for idx in pool_indices:
            stage = nn.Sequential(*features[prev:idx+1])  # include the pool layer
            self.stages.append(stage)
            prev = idx+1
        # After last pool, there may be remaining convs; include them as final stage if any
        if prev < len(features):
            self.stages.append(nn.Sequential(*features[prev:]))

        # For each side output, map feature channels to 1
        self.side_convs = nn.ModuleList([
            nn.Conv2d(self._out_channels(i), 1, kernel_size=1) for i in range(len(self.stages))
        ])
        # fusion weights (learnable scalar per side)
        self.fuse_weights = nn.Parameter(torch.ones(len(self.side_convs), dtype=torch.float32), requires_grad=True)

    def _out_channels(self, stage_idx):
        t = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            for i in range(stage_idx + 1):
                t = self.stages[i](t)
        return t.shape[1]

    def forward(self, x):
        side_outs = []
        h = x
        for i, s in enumerate(self.stages):
            h = s(h)
            so = self.side_convs[i](h)  # (N,1,h,w)
            # upsample to input size
            so_up = F.interpolate(so, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            side_outs.append(so_up)
        # fuse using learned weights (normalize weights)
        w = torch.softmax(self.fuse_weights, dim=0)
        fused = 0
        for i, so in enumerate(side_outs):
            fused = fused + w[i] * so
        return side_outs, fused
