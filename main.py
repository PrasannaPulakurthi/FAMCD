import torch
from models import FeatureGenerator, Classifier
from datasets import get_data_loaders
from trainer import train_epoch_step1, train_epoch_step2, train_epoch_step3
from utils import save_checkpoint, load_checkpoint
import torch.optim as optim
import torch.nn.functional as F


def evaluate(model_G, model_F1, model_F2, loader, device):
    model_G.eval()
    model_F1.eval()
    model_F2.eval()
    correct = total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            feats = model_G(x)
            preds1 = model_F1(feats)
            preds2 = model_F2(feats)
            avg_preds = (F.softmax(preds1, dim=1) + F.softmax(preds2, dim=1)) / 2
            pred_labels = avg_preds.argmax(dim=1)
            correct += (pred_labels == y).sum().item()
            total += y.size(0)

    acc = 100. * correct / total
    return acc


def run(domain='mnist_usps', epochs=20, ckpt_dir='./checkpoints'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    loaders = get_data_loaders(batch_size=32, image_size=32)
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

    # Models
    model_G = FeatureGenerator().to(device)
    model_F1 = Classifier().to(device)
    model_F2 = Classifier().to(device)

    # Optimizers
    optimizer_G = optim.Adam(model_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_F = optim.Adam(list(model_F1.parameters()) + list(model_F2.parameters()), lr=0.0002, betas=(0.5, 0.999))

    # Load checkpoint if available
    load_checkpoint(model_G, model_F1, model_F2, optimizer_G, optimizer_F, f"{ckpt_dir}/checkpoint.pth")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_epoch_step1(model_G, model_F1, model_F2, optimizer_G, optimizer_F, loader_src, loader_tgt, device)
        train_epoch_step2(model_G, model_F1, model_F2, optimizer_F, loader_src, loader_tgt, device)
        train_epoch_step3(model_G, model_F1, model_F2, optimizer_G, loader_tgt, device)

        acc = evaluate(model_G, model_F1, model_F2, loader_src_test, device)
        acc = evaluate(model_G, model_F1, model_F2, loader_tgt_test, device)
        print(f"Source Test Accuracy: {acc:.2f}%, Target Test Accuracy: {acc:.2f}%")

        save_checkpoint({
            'model_G': model_G.state_dict(),
            'model_F1': model_F1.state_dict(),
            'model_F2': model_F2.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_F': optimizer_F.state_dict(),
        }, ckpt_dir)


if __name__ == '__main__':
    run(domain='mnist_usps', epochs=20)
