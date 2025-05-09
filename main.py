import torch
from models import FeatureGenerator, Classifier
from datasets import get_data_loaders
from trainer import train_epoch_step1, train_epoch_step2, train_epoch_step3, evaluate
from utils import save_checkpoint, load_checkpoint, save_sample_images
import torch.optim as optim
from tqdm import trange
import argparse

NUM_CLASSES = {'mnist':10, 'usps':10, 'svhn':10, 'gtsrb':43, 'synsig':43}

def run(src='mnist', tgt='usps', image_size=32, epochs=20):
    domain = src + '_' + tgt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir = './checkpoints/' + domain
    # Load data
    loaders = get_data_loaders(batch_size=128, image_size=image_size)
    loader_src, loader_src_test = loaders[src]
    loader_tgt, loader_tgt_test = loaders[tgt]

    print(len(loader_src), len(loader_tgt))

    save_sample_images(loader_src, loader_tgt)

    # Models
    model_G = FeatureGenerator().to(device)
    model_F1 = Classifier(num_classes=NUM_CLASSES[src]).to(device)
    model_F2 = Classifier(num_classes=NUM_CLASSES[src]).to(device)

    # Optimizers
    optimizer_G = optim.Adam(model_G.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0005)
    optimizer_F = optim.Adam(list(model_F1.parameters()) + list(model_F2.parameters()), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0005)

    # Load checkpoint if available
    # load_checkpoint(model_G, model_F1, model_F2, optimizer_G, optimizer_F, f"{ckpt_dir}/checkpoint.pth")

    for epoch in trange(epochs, desc="Training Epochs"):
        train_epoch_step1(model_G, model_F1, model_F2, optimizer_G, optimizer_F, loader_src, loader_tgt, device)
        train_epoch_step2(model_G, model_F1, model_F2, optimizer_F, loader_src, loader_tgt, device)
        train_epoch_step3(model_G, model_F1, model_F2, optimizer_G, loader_tgt, device)

        acc_src1, acc_src2 = evaluate(model_G, model_F1, model_F2, loader_src_test, device)
        acc_tgt1, acc_tgt2 = evaluate(model_G, model_F1, model_F2, loader_tgt_test, device)
        print(f"Source Test Accuracy: [{acc_src1:.2f}%, {acc_src2:.2f}%], Target Test Accuracy: [{acc_tgt1:.2f}%, {acc_tgt2:.2f}%]")

        save_checkpoint({
            'model_G': model_G.state_dict(),
            'model_F1': model_F1.state_dict(),
            'model_F2': model_F2.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_F': optimizer_F.state_dict(),
        }, ckpt_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Domain Adaptation Model")
    parser.add_argument('--src', type=str, default='mnist', help='Source domain')
    parser.add_argument('--tgt', type=str, default='usps', help='Target domain')
    parser.add_argument('--image_size', type=int, default=32, help='Input image size')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    args = parser.parse_args()
    
    run(src=args.src, tgt=args.tgt, image_size=args.image_size, epochs=args.epochs)

