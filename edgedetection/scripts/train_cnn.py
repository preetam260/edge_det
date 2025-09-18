import argparse, os, time
from pathlib import Path
import torch, torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import BSDSDataset
from src.models.cnn_simple import SimpleEdgeCNN
from src.losses import class_balanced_sigmoid_cross_entropy
from src.train_helper import save_checkpoint, plot_losses
import torchvision.transforms.functional as TF
import numpy as np

def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0; running_loss = 0.0
    for imgs, gts, _ in loader:
        imgs = imgs.to(device); gts = gts.to(device)
        opt.zero_grad()
        preds = model(imgs)  # logits (N,1,H,W)
        loss = class_balanced_sigmoid_cross_entropy(preds, gts)
        loss.backward(); opt.step()
        running_loss += float(loss.item()); total += 1
    return running_loss / max(1, total)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = 0; running_loss = 0.0
    for imgs, gts, _ in loader:
        imgs = imgs.to(device); gts = gts.to(device)
        preds = model(imgs)
        loss = class_balanced_sigmoid_cross_entropy(preds, gts)
        running_loss += float(loss.item()); total += 1
    return running_loss / max(1, total)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_images", type=int, default=-1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_train = BSDSDataset(os.path.join(args.data_dir, "images", "train"), os.path.join(args.data_dir, "groundTruth", "train"), max_images=args.max_images)
    ds_val = BSDSDataset(os.path.join(args.data_dir, "images", "val"), os.path.join(args.data_dir, "groundTruth", "val"), max_images=args.max_images)
    loader_tr = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = SimpleEdgeCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    best_val = 1e9
    train_losses=[]; val_losses=[]
    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss = train_one_epoch(model, loader_tr, opt, device)
        val_loss = eval_epoch(model, loader_val, device)
        train_losses.append(tr_loss); val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{args.epochs} tr={tr_loss:.4f} val={val_loss:.4f} time={time.time()-t0:.1f}s")
        # save checkpoint
        save_checkpoint({"epoch": epoch+1, "model_state": model.state_dict(), "opt_state": opt.state_dict()}, os.path.join(args.out_dir, f"cnn_epoch{epoch+1}.pth"))
        plot_losses(train_losses, val_losses, os.path.join(args.out_dir, "loss_curve.png"))
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint({"epoch": epoch+1, "model_state": model.state_dict(), "opt_state": opt.state_dict()}, os.path.join(args.out_dir, "best_cnn.pth"))

if __name__ == "__main__":
    main()
