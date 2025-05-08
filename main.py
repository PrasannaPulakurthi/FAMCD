import torch
from models import FeatureGenerator, Classifier
from datasets import get_data_loaders
from trainer import train_epoch_step1, train_epoch_step2, train_epoch_step3, evaluate
from utils import save_checkpoint, load_checkpoint, save_sample_images
import torch.optim as optim
from tqdm import trange


def run(src='mnist', tgt='usps', epochs=20, ckpt_dir='./checkpoints'):
    domain = src + '_' + tgt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    loaders = get_data_loaders(batch_size=512, image_size=32)
    if domain == 'mnist_usps':
        loader_src, loader_src_test = loaders['mnist']
        loader_tgt, loader_tgt_test = loaders['usps']
    elif domain == 'usps_mnist':
        loader_src, loader_src_test = loaders['usps']
        loader_tgt, loader_tgt_test = loaders['mnist']
    elif domain == 'svhn_mnist':
        loader_src, loader_src_test = loaders['svhn']
        loader_tgt, loader_tgt_test = loaders['mnist']
    else:
        raise ValueError("Invalid domain argument")

    # save_sample_images(loader_src, loader_tgt)

    # Models
    model_G = FeatureGenerator().to(device)
    model_F1 = Classifier().to(device)
    model_F2 = Classifier().to(device)

    # Optimizers
    optimizer_G = optim.Adam(model_G.parameters(), lr=0.001, weight_decay=0.0005)
    optimizer_F = optim.Adam(list(model_F1.parameters()) + list(model_F2.parameters()), lr=0.0002, weight_decay=0.0005)

    # Load checkpoint if available
    # load_checkpoint(model_G, model_F1, model_F2, optimizer_G, optimizer_F, f"{ckpt_dir}/checkpoint.pth")

    for epoch in trange(epochs, desc="Training Epochs"):
        train_epoch_step1(model_G, model_F1, model_F2, optimizer_G, optimizer_F, loader_src, loader_tgt, device)
        train_epoch_step2(model_G, model_F1, model_F2, optimizer_F, loader_src, loader_tgt, device)
        train_epoch_step3(model_G, model_F1, model_F2, optimizer_G, loader_tgt, device)

        acc_src = evaluate(model_G, model_F1, model_F2, loader_src_test, device)
        acc_tgt = evaluate(model_G, model_F1, model_F2, loader_tgt_test, device)
        print(f"Source Test Accuracy: {acc_src:.2f}%, Target Test Accuracy: {acc_tgt:.2f}%")

        save_checkpoint({
            'model_G': model_G.state_dict(),
            'model_F1': model_F1.state_dict(),
            'model_F2': model_F2.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_F': optimizer_F.state_dict(),
        }, ckpt_dir)


if __name__ == '__main__':
    run(src='mnist', tgt='usps', epochs=20)
