import torch
import os
import torchvision.utils as vutils
import matplotlib.pyplot as plt

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


def save_sample_images(loader_src, loader_tgt, output_dir="debug_images"):
    os.makedirs(output_dir, exist_ok=True)

    src_batch = next(iter(loader_src))
    tgt_batch = next(iter(loader_tgt))

    src_imgs, src_labels = src_batch
    tgt_imgs, _ = tgt_batch

    # Save a grid of 8 source images
    vutils.save_image(src_imgs[:8], os.path.join(output_dir, "source_batch.png"), nrow=4, normalize=True)
    vutils.save_image(tgt_imgs[:8], os.path.join(output_dir, "target_batch.png"), nrow=4, normalize=True)

    print(f"Saved source and target batches to: {output_dir}")

    # Optional: show inline (for Jupyter or Colab)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(plt.imread(os.path.join(output_dir, "source_batch.png")))
    plt.title("Source Samples")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(plt.imread(os.path.join(output_dir, "target_batch.png")))
    plt.title("Target Samples")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
