from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=512, image_size=32):
    tf_common = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    tf_mnist = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    tf_visda = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Digits Datasets
    mnist_train = datasets.MNIST(root="./data", train=True, transform=tf_mnist, download=True)
    mnist_test = datasets.MNIST(root="./data", train=False, transform=tf_mnist, download=True)

    usps_train = datasets.USPS(root="./data/USPS", train=True, transform=tf_mnist, download=True)
    usps_test = datasets.USPS(root="./data/USPS", train=False, transform=tf_mnist, download=True)

    svhn_train = datasets.SVHN(root="./data/SVHN", split="train", transform=tf_common, download=True)
    svhn_test = datasets.SVHN(root="./data/SVHN", split="test", transform=tf_common, download=True)

    # Traffic Signs Datasets from Folder
    gtsrb_train = datasets.ImageFolder(root="./data/GTSRB/train", transform=tf_common)
    gtsrb_test = datasets.ImageFolder(root="./data/GTSRB/test", transform=tf_common)

    synsig_train = datasets.ImageFolder(root="./data/SYNSIG/train", transform=tf_common)
    synsig_test = datasets.ImageFolder(root="./data/SYNSIG/test", transform=tf_common)

    # Syn2Real
    syn_train = datasets.ImageFolder(root="./data/SYN2REAL/train", transform=tf_visda)
    syn_test = datasets.ImageFolder(root="./data/SYN2REAL/train", transform=tf_common)

    real_train = datasets.ImageFolder(root="./data/SYN2REAL/validation", transform=tf_visda)
    real_test = datasets.ImageFolder(root="./data/SYN2REAL/validation", transform=tf_common)

    class_to_idx = syn_train.class_to_idx
    classes = syn_train.classes
    syn_test.class_to_idx, syn_test.classes = class_to_idx, classes
    real_train.class_to_idx, real_train.classes = class_to_idx, classes
    real_test.class_to_idx, real_test.classes = class_to_idx, classes

    return {
        'mnist': (DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True),
                  DataLoader(mnist_test, batch_size=batch_size)),
        'usps':  (DataLoader(usps_train, batch_size=batch_size, shuffle=True, drop_last=True),
                  DataLoader(usps_test, batch_size=batch_size)),
        'svhn':  (DataLoader(svhn_train, batch_size=batch_size, shuffle=True, drop_last=True),
                  DataLoader(svhn_test, batch_size=batch_size)),
        'gtsrb': (DataLoader(gtsrb_train, batch_size=batch_size, shuffle=True, drop_last=True),
                  DataLoader(gtsrb_test, batch_size=batch_size)),
        'synsig':(DataLoader(synsig_train, batch_size=batch_size, shuffle=True, drop_last=True),
                  DataLoader(synsig_test, batch_size=batch_size)),
        'syn':   (DataLoader(syn_train, batch_size=batch_size, shuffle=True, drop_last=True),
                  DataLoader(syn_test, batch_size=batch_size)),
        'real':  (DataLoader(real_train, batch_size=batch_size, shuffle=True, drop_last=True),
                  DataLoader(real_test, batch_size=batch_size)),
    }
