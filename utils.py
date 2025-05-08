import torch
import os

def save_checkpoint(state, ckpt_dir, filename="checkpoint.pth"):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, filename)
    torch.save(state, path)

def load_checkpoint(model_G, model_F1, model_F2, optimizer_G, optimizer_F, ckpt_path):
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model_G.load_state_dict(checkpoint['model_G'])
        model_F1.load_state_dict(checkpoint['model_F1'])
        model_F2.load_state_dict(checkpoint['model_F2'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_F.load_state_dict(checkpoint['optimizer_F'])
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"No checkpoint found at {ckpt_path}")
