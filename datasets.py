from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=512, image_size=32):
    tf_mnist = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    tf_svhn = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


    mnist_train = datasets.MNIST(root="./data", train=True, transform=tf_mnist, download=True)
    mnist_test = datasets.MNIST(root="./data", train=False, transform=tf_mnist, download=True)

    usps_train = datasets.USPS(root="./data", train=True, transform=tf_mnist, download=True)
    usps_test = datasets.USPS(root="./data", train=False, transform=tf_mnist, download=True)

    svhn_train = datasets.SVHN(root="./data", split="train", transform=tf_svhn, download=True)
    svhn_test = datasets.SVHN(root="./data", split="test", transform=tf_svhn, download=True)

    return {
        'mnist': (DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True),
                  DataLoader(mnist_test, batch_size=batch_size)),
        'usps':  (DataLoader(usps_train, batch_size=batch_size, shuffle=True, drop_last=True),
                  DataLoader(usps_test, batch_size=batch_size)),
        'svhn':  (DataLoader(svhn_train, batch_size=batch_size, shuffle=True, drop_last=True),
                  DataLoader(svhn_test, batch_size=batch_size))
    }
