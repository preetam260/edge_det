import torch
import torch.nn.functional as F

def class_balanced_sigmoid_cross_entropy(pred, gt, eps=1e-6):
    """
    pred: (N,1,H,W) logits
    gt: (N,1,H,W) float in [0,1] (averaged GT)
    Implements class-balanced sigmoid cross-entropy per HED (sec 2.2) approximation.
    """
    assert pred.shape == gt.shape, f"{pred.shape} vs {gt.shape}"
    b = gt.size(0)
    losses = []
    for i in range(b):
        gt_i = gt[i,0]
        pred_i = pred[i,0]
        pos = (gt_i > 0.5).float().sum()
        neg = (gt_i <= 0.5).float().sum()
        # avoid zero
        N = pos + neg + eps
        beta = neg / N
        # per-pixel weights
        weight_pos = beta
        weight_neg = 1 - beta
        # stable BCE with logits
        loss_pos = weight_pos * F.binary_cross_entropy_with_logits(pred_i, torch.ones_like(pred_i), reduction='none')
        loss_neg = weight_neg * F.binary_cross_entropy_with_logits(pred_i, torch.zeros_like(pred_i), reduction='none')
        mask = (gt_i > 0.5).float()
        loss = (mask * loss_pos + (1-mask) * loss_neg).mean()
        losses.append(loss)
    return torch.stack(losses).mean()
