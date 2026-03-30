# PINN — 1D Heat Equation Solver
### Mini Project | Physics-Informed Neural Networks | PyTorch

---

## Problem Statement

solving the **1D Heat (Diffusion) Equation**:

```
∂u/∂t = α · ∂²u/∂x²,    x ∈ [0, 1],  t ∈ [0, 1]
```

### Conditions

| Type | Expression |
|------|-----------|
| Initial Condition | u(x, 0) = sin(πx) |
| Boundary (left)  | u(0, t) = 0       |
| Boundary (right) | u(1, t) = 0       |
| Exact Solution   | u(x,t) = e^(−α π² t) · sin(πx) |

Thermal diffusivity **α = 0.1** (configurable in `pinn_model.py`).

---

## What is a PINN?

A **Physics-Informed Neural Network (PINN)** is a neural network trained to satisfy both:
1. **Data constraints** — Initial and boundary conditions.
2. **Physics constraints** — The governing PDE residual, computed via automatic differentiation.

The total loss is:

```
L = w_ic · L_IC  +  w_bc · L_BC  +  w_pde · L_PDE
```

where:
- **L_IC** — Mean squared error at initial condition points
- **L_BC** — Mean squared error at boundary points
- **L_PDE** — Mean squared PDE residual at collocation points

---

## Architecture

```
Input (x, t)
     ↓
Linear(2 → 64) + Tanh
     ↓
Linear(64 → 64) + Tanh    ×  4 hidden layers
     ↓
Linear(64 → 1)
     ↓
Output: u(x, t)
```

- **Activation**: Tanh (smooth for clean second-order derivatives)
- **Initialisation**: Xavier Normal
- **Optimizer**: Adam (lr = 1e-3) with StepLR scheduler
- **Epochs**: 10,000

---

## Project Structure

```
pinn_heat_equation/
│
├── pinn_model.py       ← Main: architecture, loss, training, evaluation
├── visualize.py        ← Advanced plots: 3D surface, animation, error maps
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
│
└── results/            ← Auto-created after training
    ├── pinn_heat_model.pth      ← Saved model weights
    ├── solution_comparison.png  ← PINN vs Exact vs Error
    ├── time_slices.png          ← Temperature at t=0, 0.1, 0.25, 0.5, 1.0
    ├── loss_history.png         ← Training convergence curves
    ├── 3d_surface.png           ← 3D surface comparison
    ├── error_analysis.png       ← Absolute & relative error maps
    └── heat_animation.gif       ← Animated heat diffusion
```

---

## How to Run

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> PyTorch GPU is optional; the code auto-detects CUDA. CPU training (~5 min for 10k epochs) is fine.

### Step 2 — Train the PINN

```bash
python pinn_model.py
```

This will:
- Train the network for 10,000 epochs
- Print loss every 500 epochs
- Save model weights to `results/`
- Generate & display comparison plots

### Step 3 — Advanced Visualizations (optional)

```bash
python visualize.py
```

Generates 3D surfaces, error maps, and an animated GIF.

---

## Expected Results

| Metric | Typical Value |
|--------|--------------|
| L2 Error | ~1e-3 to 1e-4 |
| Max Error | ~5e-3 |
| Training time (CPU) | ~3–6 minutes |

---

## Hyperparameters You Can Tune

In `pinn_model.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `ALPHA`   | 0.1 | Thermal diffusivity |
| `LAYERS`  | [2,64,64,64,64,1] | Network depth/width |
| `EPOCHS`  | 10000 | Training iterations |
| `n_col`   | 5000 | Collocation points |
| `w_ic / w_bc / w_pde` | 10/10/1 | Loss weights |

---

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686–707.
2. Karniadakis, G. E., et al. (2021). *Physics-informed machine learning.* Nature Reviews Physics.

---

## How to Present This

1. **Introduction**: Briefly explain the Heat Equation and its physical meaning (heat diffusion in a 1D rod).
2. **Method**: Explain how a PINN replaces traditional numerical solvers (FEM, FDM).
3. **Results**: Show the comparison plots, loss curves, and error metrics.
4. **Conclusion**: Highlight low error (~1e-3), no mesh required, and generalizability of PINNs.