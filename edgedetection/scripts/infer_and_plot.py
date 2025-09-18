import argparse, os
from pathlib import Path
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from src.utils_bsds import load_bsds_gt_mat
from src.models.cnn_simple import SimpleEdgeCNN
from src.models.vgg_decoder import VGG16Decoder
from src.models.hed_like import HEDLike

def imread_rgb(path):
    return np.array(Image.open(path).convert("RGB"))

def show_results(img_path, gt_mat_path, model_checkpoint, model_type, out_path, thresh=0.5, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    img = imread_rgb(img_path)
    gt = load_bsds_gt_mat(gt_mat_path) if gt_mat_path is not None else None
    # prepare tensor
    import torch
    t = TF.to_tensor(Image.fromarray(img)).unsqueeze(0).to(device)
    if model_type=="cnn":
        model = SimpleEdgeCNN().to(device)
    elif model_type=="vgg":
        model = VGG16Decoder(pretrained=False).to(device)
    elif model_type=="hed":
        model = HEDLike(pretrained=False).to(device)
    else:
        raise ValueError(model_type)
    ck = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    with torch.no_grad():
        out = model(t)
        if isinstance(out, tuple) or isinstance(out, list):
            # hed returns (side_outs_list, fused)
            if len(out)==2 and isinstance(out[0], list):
                side, fused = out
                pred = torch.sigmoid(fused)[0,0].cpu().numpy()
            else:
                pred = torch.sigmoid(out[0])[0,0].cpu().numpy()
        else:
            pred = torch.sigmoid(out)[0,0].cpu().numpy()
    bin_pred = (pred >= thresh).astype(np.uint8)*255
    # plotting
    cols=3 if gt is not None else 2
    plt.figure(figsize=(12,4))
    ax=plt.subplot(1,cols,1); ax.imshow(img); ax.set_title("Image"); ax.axis("off")
    if gt is not None:
        ax=plt.subplot(1,cols,2); ax.imshow(gt, cmap="gray"); ax.set_title("GT (avg)"); ax.axis("off")
        ax=plt.subplot(1,cols,3); ax.imshow(bin_pred, cmap="gray"); ax.set_title(f"Pred >= {thresh}"); ax.axis("off")
    else:
        ax=plt.subplot(1,cols,2); ax.imshow(bin_pred, cmap="gray"); ax.set_title(f"Pred >= {thresh}"); ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200); plt.close()

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", choices=["cnn","vgg","hed"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()
    show_results(args.img, args.gt, args.ckpt, args.model, args.out, args.thresh)
