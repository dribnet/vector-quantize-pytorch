# FashionMnist VQ experiment with various settings, using FSQ.
# From https://github.com/minyoungg/vqtorch/blob/main/examples/autoencoder.py

from tqdm.auto import trange

import math
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from vector_quantize_pytorch import FSQ


lr = 3e-4
train_iter = 1000
levels = [8, 6, 5] # target size 2^8, actual size 240
num_codes = math.prod(levels)
seed = 1234
device = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleFSQAutoEncoder(nn.Module):
    def __init__(self, levels: list[int]):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(8, 8, kernel_size=6, stride=3, padding=0),
                FSQ(levels),
                nn.ConvTranspose2d(8, 8, kernel_size=6, stride=3, padding=0),
                nn.Conv2d(8, 16, kernel_size=4, stride=1, padding=2),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=2),
            ]
        )
        return

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, FSQ):
                x, indices = layer(x)
            else:
                x = layer(x)

        return x.clamp(-1, 1), indices


def train(model, train_loader, train_iterations=1000):
    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x, y = next(data_iter)
            yield x.to(device), y.to(device)

    for _ in (pbar := trange(train_iterations)):
        opt.zero_grad()
        x, _ = next(iterate_dataset(train_loader))
        out, indices = model(x)
        rec_loss = (out - x).abs().mean()
        rec_loss.backward()

        opt.step()
        pbar.set_description(
            f"rec loss: {rec_loss.item():.3f} | "
            + f"active %: {indices.unique().numel() / num_codes * 100:.3f}"
        )
    return


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_dataset = DataLoader(
    datasets.MNIST(
        root="~/data/fashion_mnist", train=True, download=True, transform=transform
    ),
    batch_size=256,
    shuffle=True,
)

print("baseline")
torch.random.manual_seed(seed)
model = SimpleFSQAutoEncoder(levels).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr)
train(model, train_dataset, train_iterations=train_iter)

# ---- 8< -----

batch = next(iter(train_dataset))
img, _ = batch
img = img.to(device)
rec_x2 = model(img)

# Extracting recorded information
temp = rec_x2[0].cpu().detach().numpy()

import matplotlib.pyplot as plt

# Initializing subplot counter
counter = 1

# Plotting first five images of the last batch
for idx in range(5):
    plt.subplot(2, 5, counter)
    plt.title(f"index {idx}")
    plt.imshow(temp[idx].reshape(28,28), cmap= 'gray')
    plt.axis('off')

    # Incrementing the subplot counter
    counter+=1

# Iterating over first five
# images of the last batch

# Obtaining image from the dictionary
val = img.cpu()

for idx in range(5):
    # Plotting image
    plt.subplot(2,5,counter)
    plt.imshow(val[idx].reshape(28, 28), cmap = 'gray')
    plt.title("Original Image")
    plt.axis('off')

    # Incrementing subplot counter
    counter+=1

plt.tight_layout()
plt.savefig('figgy2.png')
