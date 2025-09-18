import os, time
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def save_checkpoint(state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def plot_losses(losses, val_losses, out_path):
    plt.figure()
    plt.plot(losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(out_path, dpi=150)
    plt.close()
