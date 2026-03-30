"""
============================================================
  PINN for 1D Heat Equation
  u_t = alpha * u_xx,  x in [0,1], t in [0,1]
  BC: u(0,t) = 0,  u(1,t) = 0
  IC: u(x,0) = sin(pi*x)
  Exact: u(x,t) = exp(-alpha*pi^2*t) * sin(pi*x)
============================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

#Reproducibility
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

#Thermal diffusivity 
ALPHA = 0.1

class PINN(nn.Module):
    """
    Fully-connected network that maps (x, t) -> u
    """
    def __init__(self, layers):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 1):
            self.net.add_module(f"linear_{i}", nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.net.add_module(f"act_{i}", nn.Tanh())

        # Xavier initialisation – better for tanh
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)


#Loss Components
def pde_residual(model, x, t, alpha):
    """Physics loss: u_t - alpha * u_xx = 0"""
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)

    u = model(x, t)

    u_t  = torch.autograd.grad(u, t,  grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x  = torch.autograd.grad(u, x,  grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    residual = u_t - alpha * u_xx
    return residual


def compute_loss(model, alpha,
                 x_ic, t_ic, u_ic,
                 x_bc0, t_bc0,
                 x_bc1, t_bc1,
                 x_col, t_col):
    """
    Total loss = w_ic * L_ic + w_bc * L_bc + w_pde * L_pde
    """
    #IC loss
    u_pred_ic = model(x_ic, t_ic)
    loss_ic = torch.mean((u_pred_ic - u_ic) ** 2)

    #BC loss (Dirichlet = 0 at x=0 and x=1)
    u_pred_bc0 = model(x_bc0, t_bc0)
    u_pred_bc1 = model(x_bc1, t_bc1)
    loss_bc = torch.mean(u_pred_bc0 ** 2) + torch.mean(u_pred_bc1 ** 2)

    #PDE residual loss
    res = pde_residual(model, x_col, t_col, alpha)
    loss_pde = torch.mean(res ** 2)

    #Weights
    w_ic, w_bc, w_pde = 10.0, 10.0, 1.0
    total = w_ic * loss_ic + w_bc * loss_bc + w_pde * loss_pde
    return total, loss_ic, loss_bc, loss_pde


def exact_solution(x, t, alpha):
    return np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)


#Training Data Preparation
def prepare_training_data(n_ic=200, n_bc=200, n_col=5000):
    """Sample IC, BC and collocation points."""

    def T(arr):
        return torch.tensor(arr, dtype=torch.float32, device=DEVICE).reshape(-1, 1)

    # Initial condition: t = 0, x ~ U[0,1]
    x_ic_np = np.random.uniform(0, 1, n_ic)
    t_ic_np = np.zeros(n_ic)
    u_ic_np = np.sin(np.pi * x_ic_np)           # IC: u(x,0) = sin(πx)
    x_ic, t_ic, u_ic = T(x_ic_np), T(t_ic_np), T(u_ic_np)

    # Boundary conditions
    t_bc_np = np.random.uniform(0, 1, n_bc)
    x_bc0 = T(np.zeros(n_bc));  t_bc0 = T(t_bc_np)   # x = 0
    x_bc1 = T(np.ones(n_bc));   t_bc1 = T(t_bc_np)   # x = 1

    # Collocation points (Latin Hypercube-like)
    x_col = T(np.random.uniform(0, 1, n_col))
    t_col = T(np.random.uniform(0, 1, n_col))

    return x_ic, t_ic, u_ic, x_bc0, t_bc0, x_bc1, t_bc1, x_col, t_col


#Training Loop
def train(model, optimizer, scheduler, data, epochs=10000, log_every=500):
    history = {"total": [], "ic": [], "bc": [], "pde": []}

    x_ic, t_ic, u_ic, x_bc0, t_bc0, x_bc1, t_bc1, x_col, t_col = data

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        total, l_ic, l_bc, l_pde = compute_loss(
            model, ALPHA,
            x_ic, t_ic, u_ic,
            x_bc0, t_bc0, x_bc1, t_bc1,
            x_col, t_col
        )

        total.backward()
        optimizer.step()
        scheduler.step()

        history["total"].append(total.item())
        history["ic"].append(l_ic.item())
        history["bc"].append(l_bc.item())
        history["pde"].append(l_pde.item())

        if epoch % log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:6d} | Total: {total.item():.4e} | "
                  f"IC: {l_ic.item():.4e} | BC: {l_bc.item():.4e} | "
                  f"PDE: {l_pde.item():.4e}")

    return history


#Evaluation & Plotting
def evaluate_and_plot(model, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    Nx, Nt = 200, 200
    x_np = np.linspace(0, 1, Nx)
    t_np = np.linspace(0, 1, Nt)
    X, T_ = np.meshgrid(x_np, t_np)

    def T(arr):
        return torch.tensor(arr.flatten(), dtype=torch.float32, device=DEVICE).reshape(-1, 1)

    with torch.no_grad():
        u_pred = model(T(X), T(T_)).cpu().numpy().reshape(Nt, Nx)

    u_exact = exact_solution(X, T_, ALPHA)
    error   = np.abs(u_pred - u_exact)

    #Fig 1: PINN vs Exact vs Error
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("PINN Solution – 1D Heat Equation  ($\\alpha=0.1$)", fontsize=14, fontweight='bold')

    im0 = axes[0].contourf(X, T_, u_pred,  levels=50, cmap='hot')
    axes[0].set_title("PINN Prediction"); axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(X, T_, u_exact, levels=50, cmap='hot')
    axes[1].set_title("Exact Solution");  axes[1].set_xlabel("x")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].contourf(X, T_, error,   levels=50, cmap='viridis')
    axes[2].set_title("Absolute Error");  axes[2].set_xlabel("x")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(f"{save_dir}/solution_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {save_dir}/solution_comparison.png")

    #Fig 2: Time slices
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, 5))
    for idx, t_val in enumerate([0.0, 0.1, 0.25, 0.5, 1.0]):
        t_idx = np.argmin(np.abs(t_np - t_val))
        ax.plot(x_np, u_exact[t_idx],  '--',  color=colors[idx], linewidth=1.8, label=f"Exact  t={t_val}")
        ax.plot(x_np, u_pred[t_idx],   'o',   color=colors[idx], markersize=3,  label=f"PINN   t={t_val}")
    ax.set_xlabel("x", fontsize=12); ax.set_ylabel("u(x,t)", fontsize=12)
    ax.set_title("Temperature Profile at Various Time Snapshots", fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/time_slices.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {save_dir}/time_slices.png")

    #Error stats
    l2_err  = np.sqrt(np.mean(error**2))
    max_err = error.max()
    print(f"\n{'='*45}")
    print(f"  L2  Error : {l2_err:.4e}")
    print(f"  Max Error : {max_err:.4e}")
    print(f"{'='*45}\n")
    return l2_err, max_err


def plot_loss_history(history, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Training Loss History", fontsize=13, fontweight='bold')

    epochs = range(1, len(history["total"]) + 1)
    axes[0].semilogy(epochs, history["total"], color='black',   label='Total',  linewidth=2)
    axes[0].semilogy(epochs, history["ic"],    color='#e74c3c', label='IC',     linewidth=1.5)
    axes[0].semilogy(epochs, history["bc"],    color='#2980b9', label='BC',     linewidth=1.5)
    axes[0].semilogy(epochs, history["pde"],   color='#27ae60', label='PDE',    linewidth=1.5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (log scale)")
    axes[0].set_title("All Losses"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(epochs, history["total"], color='black', linewidth=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Total Loss (log scale)")
    axes[1].set_title("Total Loss Convergence"); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/loss_history.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {save_dir}/loss_history.png")


if __name__ == "__main__":
    # Network: input(2) → 4 hidden layers × 64 neurons → output(1)
    LAYERS = [2, 64, 64, 64, 64, 1]
    EPOCHS = 10000

    model     = PINN(LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)

    print(f"\n{'='*50}")
    print(f"  PINN – 1D Heat Equation Solver")
    print(f"  alpha = {ALPHA}, Epochs = {EPOCHS}")
    print(f"  Architecture: {LAYERS}")
    print(f"{'='*50}\n")

    data    = prepare_training_data(n_ic=200, n_bc=200, n_col=5000)
    history = train(model, optimizer, scheduler, data, epochs=EPOCHS)

    # Save model
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/pinn_heat_model.pth")
    print("\nModel saved → results/pinn_heat_model.pth")

    plot_loss_history(history)
    evaluate_and_plot(model)