import torch
from torch.utils.data import DataLoader
import argparse
from torchvision import datasets, transforms
from model import CVAE
import matplotlib.pyplot as plt
from torchvision.transforms import v2
import torch.nn.functional as F


def train():
    transform_v2 = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform_v2
    )

    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform_v2
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    batch_size = 64

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(input_dim=28, latent_dim=20, num_classes=10).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 10

    def loss_function(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE, KLD

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        bce_loss = 0
        kld_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            labels = F.one_hot(labels, num_classes=10).float()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data, labels)
            BCE, KLD = loss_function(recon_batch, data, mu, logvar)
            loss = BCE + KLD
            loss.backward()
            train_loss += loss.item()
            bce_loss += BCE.item()
            kld_loss += KLD.item()
            optimizer.step()

        print(
            f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset):.4f}, BCE: {bce_loss / len(train_loader.dataset):.4f}, KLD: {kld_loss / len(train_loader.dataset):.4f}"
        )

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            labels = F.one_hot(labels, num_classes=10).float()
            recon_batch, mu, logvar = model(data, labels)
            BCE, KLD = loss_function(recon_batch, data, mu, logvar)
            loss = BCE + KLD
            test_loss += loss.item()
    print(f"Test Loss: {test_loss / len(test_loader.dataset):.4f}")

    test_labels = next(iter(test_loader))[1][:16].to(device)
    input_labels = F.one_hot(test_labels, num_classes=10).float()
    with torch.no_grad():
        latent = torch.randn(16, 20).to(device)
        latent = torch.cat([latent, input_labels], dim=1)
        generated_data = model.decoder(latent)
        generated_data = generated_data * 255.0
        generated_data = generated_data.cpu().numpy().astype("uint8")
        fig, axes = plt.subplots(2, 8, figsize=(12, 4))
        for i in range(16):
            ax = axes[i // 8, i % 8]
            ax.imshow(generated_data[i][0], cmap="gray")
            ax.axis("off")
            ax.set_title(f"Label: {test_labels[i].item()}")
        plt.show()
        plt.savefig("cvae_reconstructions.png")

    torch.save(model.state_dict(), "cvae_mnist.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--test_png_path",
        type=str,
        default="cvae_generated_samples.png",
        help="Path to save generated samples PNG",
    )
    args = parser.parse_args()

    if args.train:
        train()
    test_data = torch.arange(0, 10)
    model = CVAE(input_dim=28, latent_dim=20, num_classes=10)
    model.load_state_dict(torch.load("cvae_mnist.pth"))
    model.eval()
    with torch.no_grad():
        latent = torch.randn(10, 20)
        input_labels = F.one_hot(test_data, num_classes=10).float()
        latent = torch.cat([latent, input_labels], dim=1)
        generated_data = model.decoder(latent)
        generated_data = generated_data * 255.0
        generated_data = generated_data.numpy().astype("uint8")
        fig, axes = plt.subplots(1, 10, figsize=(15, 2))
        for i in range(10):
            ax = axes[i]
            ax.imshow(generated_data[i][0], cmap="gray")
            ax.axis("off")
            ax.set_title(f"Label: {i}")
        plt.show()
        plt.savefig(args.test_png_path)
