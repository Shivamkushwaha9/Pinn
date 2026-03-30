import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from pinn_model import PINN, exact_solution, ALPHA, DEVICE
import os

LAYERS = [2, 64, 64, 64, 64, 1]


def load_model(path="results/pinn_heat_model.pth"):
    model = PINN(LAYERS).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def make_grid(Nx=200, Nt=200):
    x_np = np.linspace(0, 1, Nx)
    t_np = np.linspace(0, 1, Nt)
    X, T_ = np.meshgrid(x_np, t_np)
    return x_np, t_np, X, T_


def predict(model, X, T_):
    def tens(a):
        return torch.tensor(a.flatten(), dtype=torch.float32, device=DEVICE).reshape(-1,1)
    with torch.no_grad():
        u = model(tens(X), tens(T_)).cpu().numpy().reshape(X.shape)
    return u


#3-D Surface Plot
def plot_3d_surface(model, save_dir="results"):
    x_np, t_np, X, T_ = make_grid(100, 100)
    u_pred  = predict(model, X, T_)
    u_exact = exact_solution(X, T_, ALPHA)

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("3D Surface – 1D Heat Equation", fontsize=14, fontweight='bold')

    for i, (data, title) in enumerate([(u_pred, "PINN Prediction"), (u_exact, "Exact Solution")]):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        ax.plot_surface(X, T_, data, cmap='inferno', alpha=0.92, linewidth=0)
        ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_zlabel("u(x,t)")
        ax.set_title(title, fontsize=12)
        ax.view_init(elev=30, azim=-60)

    plt.tight_layout()
    path = f"{save_dir}/3d_surface.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {path}")


#Animated GIF
def plot_animation(model, save_dir="results"):
    x_np, t_np, X, T_ = make_grid(200, 100)
    u_pred  = predict(model, X, T_)
    u_exact = exact_solution(X, T_, ALPHA)

    fig, ax = plt.subplots(figsize=(8, 4))
    line_pred,  = ax.plot([], [], 'r-',  linewidth=2, label='PINN')
    line_exact, = ax.plot([], [], 'b--', linewidth=2, label='Exact')
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("x"); ax.set_ylabel("u(x,t)")
    ax.legend(loc='upper right')
    time_text = ax.text(0.02, 0.92, '', transform=ax.transAxes, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title("Heat Diffusion Animation", fontweight='bold')

    def init():
        line_pred.set_data([], [])
        line_exact.set_data([], [])
        time_text.set_text('')
        return line_pred, line_exact, time_text

    def update(frame):
        line_pred.set_data(x_np, u_pred[frame])
        line_exact.set_data(x_np, u_exact[frame])
        time_text.set_text(f"t = {t_np[frame]:.3f}")
        return line_pred, line_exact, time_text

    ani = animation.FuncAnimation(fig, update, frames=len(t_np),
                                  init_func=init, blit=True, interval=40)
    path = f"{save_dir}/heat_animation.gif"
    ani.save(path, writer='pillow', fps=25)
    plt.close()
    print(f"Saved → {path}")


#Error Heatmap
def plot_error_analysis(model, save_dir="results"):
    x_np, t_np, X, T_ = make_grid()
    u_pred  = predict(model, X, T_)
    u_exact = exact_solution(X, T_, ALPHA)
    error   = np.abs(u_pred - u_exact)
    rel_err = error / (np.abs(u_exact) + 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Error Analysis", fontsize=13, fontweight='bold')

    im0 = axes[0].contourf(X, T_, error, levels=50, cmap='Reds')
    axes[0].set_title("Absolute Error |u_PINN − u_exact|")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(X, T_, rel_err, levels=50, cmap='YlOrRd')
    axes[1].set_title("Relative Error")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("t")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    path = f"{save_dir}/error_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved → {path}")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    print("Loading trained model …")
    model = load_model()

    print("Generating 3D surface …")
    plot_3d_surface(model)

    print("Generating error analysis …")
    plot_error_analysis(model)

    print("Generating animation (may take ~30 s) …")
    plot_animation(model)

    print("\nAll visualizations saved to ./results/")