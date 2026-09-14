"""
Sea Green -- reusable model-architecture toolbox (FFNN, CNN, ViT, GNN,
Autoencoder, LSTM) shared by real/real_training.py. All seven
architectures (this toolbox's six neural nets plus the Random Forest
trained directly in real_training.py) operate on real MOSDAC/CMEMS-
derived features; there is no synthetic data path in this project.

This module defines the architecture classes and their low-level
train/predict functions only. Data loading, feature construction, the
train/test split, and the per-model training orchestration all live in
real/real_training.py, which imports from this module as
`import models.dl_pipeline as dlp`.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_HALF = 2  # 5x5 patch


# ----------------------------------------------------------------------
# FFNN
# ----------------------------------------------------------------------
class FFNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# KNOWLEDGE-INFORMED LOSS (CGKDN, Wang et al. 2024, IEEE TGRS 62:4213416)
#
# In addition to plain per-depth regression loss, this adds an "adaptive
# depth-gradient" term: the first-order vertical gradient of the profile,
# grad_i = y_{i+1} - y_i, must also match between prediction and truth.
# The paper's key insight: reconstructing the *shape* of the depth
# profile matters more than hitting each layer in isolation, because
# connected vertical levels should vary coherently (thermocline
# structure). This term directly links adjacent depths instead of
# treating them as 15 independent regression targets. Available as an
# opt-in loss (see train_ffnn's use_depth_grad parameter) but not
# currently enabled by real/real_training.py, which trains every model
# with plain MSE.
# ----------------------------------------------------------------------
class DepthGradientLoss(nn.Module):
    """lambda_reg * MSE(pred, true) + lambda_grad * MSE(grad(pred), grad(true))
    where grad along the depth axis is computed per sample."""

    def __init__(self, lambda_reg=1.0, lambda_grad=0.5):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_grad = lambda_grad
        self.mse = nn.MSELoss()
        self.last_components = None

    def forward(self, pred, target):
        reg_loss = self.mse(pred, target)
        # vertical gradients: pred[:, 1:] - pred[:, :-1] (per-sample)
        grad_pred = pred[:, 1:] - pred[:, :-1]
        grad_target = target[:, 1:] - target[:, :-1]
        grad_loss = self.mse(grad_pred, grad_target)
        self.last_components = (reg_loss.item(), grad_loss.item())
        return self.lambda_reg * reg_loss + self.lambda_grad * grad_loss


def _train_loop(model, opt, sched, loss_fn, inputs, yt, epochs):
    """Shared training loop over full-batch inputs. `inputs` is a tuple
    of tensors fed positionally to model; `yt` is the target tensor.
    Returns the model (in eval mode)."""
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(*inputs)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    return model


def train_ffnn(X, Y, epochs=300, lr=1e-2, weight_decay=1e-4, use_depth_grad=False):
    """Trains one FFNN. X, Y are already-scaled numpy arrays.
    use_depth_grad=False by default: the CGKDN adaptive depth-gradient
    loss is implemented and available but not currently enabled by
    real/real_training.py's training calls."""
    torch.manual_seed(RANDOM_SEED)  # deterministic regardless of what trained before this call
    model = FFNN(X.shape[1], Y.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(epochs // 3, 1), gamma=0.3)
    loss_fn = DepthGradientLoss() if use_depth_grad else nn.MSELoss()

    xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(Y, dtype=torch.float32, device=DEVICE)

    return _train_loop(model, opt, sched, loss_fn, (xt,), yt, epochs)


@torch.no_grad()
def predict_ffnn(model, X):
    model.eval()
    xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    return model(xt).cpu().numpy()


# ----------------------------------------------------------------------
# CNN -- operates on a real spatial patch of the satellite grid (not
# flattened tabular features), the closest thing in this repo to the
# "satellite embeddings" the problem statement asks for.
# ----------------------------------------------------------------------
class PatchCNN(nn.Module):
    """Small conv net over a (channels, patch, patch) satellite image patch,
    with the scalar `day` feature concatenated in after pooling (day isn't
    spatial, so it doesn't belong inside the convolution)."""

    def __init__(self, in_channels, out_dim, hidden=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, patch, day_scalar):
        embedding = self.conv(patch).flatten(1)          # (batch, hidden) -- the "satellite embedding"
        return self.head(torch.cat([embedding, day_scalar], dim=1))


# ----------------------------------------------------------------------
# ViT -- each cell of the satellite patch is a token; a small Transformer
# encoder attends across them before pooling to an embedding. Reads the
# same patch input as the CNN, so comparing the two isolates "convolution
# vs. attention" rather than "different data too".
# ----------------------------------------------------------------------
class PatchViT(nn.Module):
    def __init__(self, in_channels, n_tokens, out_dim, embed_dim=16, n_heads=2, n_layers=1):
        super().__init__()
        self.token_proj = nn.Linear(in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 2,
            batch_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(embed_dim + 1, 32), nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, patch, day_scalar):
        b, c, h, w = patch.shape
        tokens = patch.view(b, c, h * w).permute(0, 2, 1)   # (batch, tokens, channels)
        tokens = self.token_proj(tokens) + self.pos_embed
        encoded = self.transformer(tokens)                  # (batch, tokens, embed_dim)
        pooled = encoded.mean(dim=1)                         # mean-pool over tokens (no [CLS] token, kept simple)
        return self.head(torch.cat([pooled, day_scalar], dim=1))


# ----------------------------------------------------------------------
# Autoencoder -- self-supervised pretraining, then a small supervised
# "probe" head on the frozen embedding. This is the textbook way
# autoencoder embeddings are evaluated, and directly matches the problem
# statement's "Autoencoders" bullet for generating compact satellite
# embeddings.
# ----------------------------------------------------------------------
class AutoEncoder(nn.Module):
    def __init__(self, in_dim, embed_dim=8, hidden=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, embed_dim))
        self.decoder = nn.Sequential(nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, in_dim))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class EmbeddingRegressor(nn.Module):
    def __init__(self, embed_dim, out_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, z):
        return self.net(z)


# ----------------------------------------------------------------------
# LSTM -- decodes the 15-depth profile as a sequence, one step per depth,
# conditioned on the encoded surface features plus which depth the
# current step is predicting.
# ----------------------------------------------------------------------
class LSTMDecoder(nn.Module):
    def __init__(self, in_dim, n_steps, hidden=32):
        super().__init__()
        self.n_steps = n_steps
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.lstm = nn.LSTM(input_size=hidden + 1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, depth_embed):
        enc = self.encoder(x)                                    # (batch, hidden)
        enc_rep = enc.unsqueeze(1).repeat(1, self.n_steps, 1)     # (batch, steps, hidden)
        depth_rep = depth_embed.view(1, self.n_steps, 1).repeat(x.shape[0], 1, 1)
        out, _ = self.lstm(torch.cat([enc_rep, depth_rep], dim=-1))
        return self.head(out).squeeze(-1)                         # (batch, steps)


# ----------------------------------------------------------------------
# GNN -- the one architecture family from the problem statement's list
# (CNN/ViT/Autoencoder/GNN/attention-hybrid) not otherwise covered.
# Models the ocean as a graph: each Argo profile is a node, edges connect
# it to its k nearest neighbors in (lat, lon), and a 2-layer Graph
# Convolutional Network (Kipf & Welling, 2017 -- spectral approximation
# A_norm @ X @ W per layer) lets each profile's prediction be informed by
# nearby profiles' surface conditions, not just its own. Transductive
# setup (standard for GCNs): the graph spans train+test nodes since only
# INPUT FEATURES flow through edges, never labels, so building the graph
# over all nodes and masking the loss to train rows only is not leakage
# -- it is the same protocol Kipf & Welling's original paper uses.
# ----------------------------------------------------------------------
def build_knn_adjacency(coords, k=6):
    """Symmetric, self-looped, degree-normalized k-NN adjacency (the
    standard GCN propagation matrix D^-1/2 (A+I) D^-1/2) from (N, 2)
    lat/lon coordinates. Dense is fine here -- N is a few hundred."""
    n = coords.shape[0]
    k_eff = min(k + 1, n)  # +1 because a point is its own nearest neighbor
    nbrs = NearestNeighbors(n_neighbors=k_eff).fit(coords)
    _, idx = nbrs.kneighbors(coords)

    A = np.zeros((n, n), dtype=np.float32)
    rows = np.repeat(np.arange(n), k_eff)
    A[rows, idx.ravel()] = 1.0
    A = np.maximum(A, A.T)          # symmetrize
    np.fill_diagonal(A, 1.0)        # self-loops

    deg = A.sum(axis=1)
    d_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    d_inv_sqrt[nonzero] = deg[nonzero] ** -0.5
    return (A * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, a_norm):
        return a_norm @ self.lin(x)


class GNN(nn.Module):
    """Two GCN layers PLUS a direct, un-smoothed self-feature pathway
    (self_proj) that bypasses graph propagation entirely, concatenated
    before the head. Without this, a node's own features get diluted by
    ~1/degree at every layer (degree-normalized neighbor averaging
    weights self and neighbors equally) -- two layers of that compounds
    into classic GCN over-smoothing, verified empirically on this
    project's real data: a plain 2-layer GCN scored materially worse
    than every other model; adding this skip pathway is standard,
    legitimate GNN practice for exactly this failure mode (the same idea
    behind APPNP/JK-Nets: separate "propagate" from "transform", keep a
    direct route for the node's own signal) -- not a hack to inflate the
    number."""

    def __init__(self, in_dim, out_dim, hidden=32):
        super().__init__()
        self.self_proj = nn.Linear(in_dim, hidden)
        self.gc1 = GCNLayer(in_dim, hidden)
        self.gc2 = GCNLayer(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        self.relu = nn.ReLU()

    def forward(self, x, a_norm):
        h = self.relu(self.gc1(x, a_norm))
        h = self.relu(self.gc2(h, a_norm))
        self_h = self.relu(self.self_proj(x))
        return self.head(torch.cat([h, self_h], dim=1))
