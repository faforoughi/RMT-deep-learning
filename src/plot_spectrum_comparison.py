
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn

from src.model import MLP
from src.experiment import data
from src.rmt_utils import gram_eigenvalues, mp_density


SEED = 42
WIDTH = 512
EPOCHS = 20


np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

train_loader, _ = data(seed=SEED)

model = MLP(width=WIDTH).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.CrossEntropyLoss()


# -----------------------
# Spectrum at initialization
# -----------------------

W0 = model.net[2].weight.detach().cpu().numpy().copy()
eig_0 = gram_eigenvalues(W0)


# -----------------------
# Training
# -----------------------

for epoch in range(1, EPOCHS + 1):

    model.train()

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        loss = criterion(model(x), y)

        loss.backward()
        optimizer.step()


# -----------------------
# Spectrum after training
# -----------------------

W20 = model.net[2].weight.detach().cpu().numpy().copy()
eig_20 = gram_eigenvalues(W20)


# -----------------------
# Marchenko-Pastur
# -----------------------

x_mp = np.linspace(0.02, 4, 1000)
mp = mp_density(x_mp, 1.0)


# -----------------------
# Figure
# -----------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)


# Bulk spectrum

axes[0].hist(
    eig_0,
    bins=60,
    density=True,
    alpha=0.45,
    label="Initialization"
)

axes[0].hist(
    eig_20,
    bins=60,
    density=True,
    alpha=0.45,
    label="Epoch 20"
)

axes[0].plot(
    x_mp,
    mp,
    linewidth=2,
    label="Marchenko-Pastur"
)

axes[0].axvline(
    4,
    linestyle="--",
    linewidth=1.5,
    label="MP upper edge"
)

axes[0].set_xlim(0, 4.5)
axes[0].set_ylim(0, 3)

axes[0].set_xlabel("Eigenvalue")
axes[0].set_ylabel("Density")

axes[0].set_title(
    "A. Bulk Spectrum"
)

axes[0].legend(fontsize=9)


# Full spectrum

axes[1].scatter(
    np.arange(len(eig_0)),
    np.sort(eig_0),
    s=10,
    alpha=0.7,
    label="Initialization"
)

axes[1].scatter(
    np.arange(len(eig_20)),
    np.sort(eig_20),
    s=10,
    alpha=0.7,
    label="Epoch 20"
)

axes[1].axhline(
    4,
    linestyle="--",
    linewidth=1.5,
    label="MP upper edge"
)

axes[1].set_xlabel(
    "Ordered Eigenvalue Index"
)

axes[1].set_ylabel(
    "Eigenvalue"
)

axes[1].set_title(
    "B. Full Spectrum and Spectral Outliers"
)

axes[1].legend(fontsize=9)


fig.suptitle(
    "Spectral Evolution of Neural Network Weights (Width = 512)",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    "results/figures/spectrum_before_after_training.png",
    dpi=250,
    bbox_inches="tight"
)

print(
    "Epoch 0 max eigenvalue:",
    eig_0.max()
)

print(
    "Epoch 20 max eigenvalue:",
    eig_20.max()
)
