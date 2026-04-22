import os
# Prevent PyTorch/OpenMP deadlocks on CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp

# Ensure single-threaded execution for stability
torch.set_num_threads(1)

from plant import Plant as QuadcopterSim

WEIGHTS_PATH = "ssml_weights.pt"

# Network Architecture
# Input: x_in = [vx, vy, vz, phi, theta]
# Output: additive acceleration mismatch [delta_ax, delta_ay, delta_az]
INPUT_DIM = 5
HIDDEN_DIM = 50
OUTPUT_DIM = 3


class SSMLNet(nn.Module):
    """
    SSML-AC neural network that estimates unmodeled acceleration disturbances.
    Takes velocity and attitude as input and outputs additive acceleration corrections.
    """
    def __init__(self):
        super(SSMLNet, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc4 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)


def spectral_normalization_clip(model, v_max=10.0):
    """Clips each parameter tensor's spectral norm to v_max for Lipschitz bounding."""
    for param in model.parameters():
        norm = torch.linalg.norm(param)
        if norm > v_max:
            param.data = param.data * (v_max / norm)


def collect_offline_data():
    """
    Simulates the quadcopter under varied controls and records
    (velocity/attitude input, true acceleration mismatch) pairs for SSML pre-training.
    """
    print("Collecting Offline Data...")
    sim = QuadcopterSim()
    dt = 0.02  # 50 Hz
    t_end = 60.0  # 3 loops

    times = np.arange(0, t_end, dt)
    state = np.zeros(8)  # [px, py, pz, vx, vy, vz, phi, theta]

    data_x = []
    data_y = []

    for t in times:
        # Varied controls to explore state space
        p_u = 0.5 * np.sin(t)
        q_u = 0.5 * np.cos(1.5 * t)
        T_u = 9.81 * sim.m + 2.0 * np.sin(0.5 * t)
        u = np.array([p_u, q_u, T_u])

        v = state[3:6]
        angles = state[6:8]
        d_true = sim.unmodeled_dynamics(t, state[0:3], v, angles)

        # Input: [vx, vy, vz, phi, theta]
        x_in = np.concatenate((state[3:6], state[6:8]))

        # Target: additive acceleration mismatch = d_true / m
        y_target = d_true / sim.m

        data_x.append(x_in)
        data_y.append(y_target)

        sol = solve_ivp(sim.dynamics, [t, t + dt], state, args=(u,), method="RK45")
        state = sol.y[:, -1]

    data_x = np.array(data_x)
    data_y = np.array(data_y)
    return data_x, data_y


def train_ssml(data_x, data_y):
    """Trains the SSMLNet using First-Order MAML (FOMAML) on offline data."""
    print("Training SSML...")
    model = SSMLNet()

    Ha = 25          # Adaptation horizon
    Ht = 25          # Training horizon
    alpha = 0.05     # Inner-loop step size
    beta = 0.01      # Outer-loop (meta) learning rate
    lambda_dir = 0.5 # Direct prediction loss weight
    lambda_norm = 0.05  # L2 weight regularization
    epochs = 500
    N = len(data_x)

    optimizer = optim.SGD(model.parameters(), lr=beta)

    X_tensor = torch.tensor(data_x, dtype=torch.float32)
    Y_tensor = torch.tensor(data_y, dtype=torch.float32)

    for epoch in range(epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        num_tasks = 10
        for _ in range(num_tasks):
            start_idx = np.random.randint(0, N - Ha - Ht - 1)

            X_a = X_tensor[start_idx : start_idx + Ha]
            Y_a = Y_tensor[start_idx : start_idx + Ha]
            X_t = X_tensor[start_idx + Ha : start_idx + Ha + Ht]
            Y_t = Y_tensor[start_idx + Ha : start_idx + Ha + Ht]

            # Inner loop: FOMAML step
            pred_a = model(X_a)
            loss_a = torch.mean((pred_a - Y_a) ** 2)
            inner_grads = torch.autograd.grad(loss_a, model.parameters(), create_graph=False)
            fast_weights = [p - alpha * g.detach() for p, g in zip(model.parameters(), inner_grads)]

            def inner_forward(x, weights):
                h1 = torch.nn.functional.relu(torch.nn.functional.linear(x, weights[0], weights[1]))
                h2 = torch.nn.functional.relu(torch.nn.functional.linear(h1, weights[2], weights[3]))
                h3 = torch.nn.functional.relu(torch.nn.functional.linear(h2, weights[4], weights[5]))
                return torch.nn.functional.linear(h3, weights[6], weights[7])

            # Outer loop: evaluate adapted model on training set
            pred_t = inner_forward(X_t, fast_weights)
            loss_t_adapt = torch.mean((pred_t - Y_t) ** 2)

            # Direct prediction loss
            pred_t_dir = model(X_t)
            loss_t_dir = torch.mean((pred_t_dir - Y_t) ** 2)

            meta_loss = (loss_t_adapt + lambda_dir * loss_t_dir) / num_tasks
            meta_loss.backward()
            epoch_loss += meta_loss.item()

        # L2 weight regularization
        for p in model.parameters():
            if p.grad is not None:
                p.grad += 2 * lambda_norm * p.data

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        spectral_normalization_clip(model)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")

    print("Pre-training Complete.")
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Weights saved to {WEIGHTS_PATH}")
    return model


def get_or_train_model():
    """Loads pre-trained weights if they exist, otherwise collects data and trains."""
    model = SSMLNet()
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))
        print(f"Loaded pre-trained weights from {WEIGHTS_PATH}")
    else:
        data_x, data_y = collect_offline_data()
        model = train_ssml(data_x, data_y)
    return model


def flatten_params(model):
    """Returns all model parameters as a single flat tensor."""
    return torch.cat([p.flatten() for p in model.parameters()])


def assign_params(model, flat_params):
    """Loads a flat parameter vector back into the model."""
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[idx : idx + numel].view_as(p))
        idx += numel


def compute_jacobian(model, x_in):
    """
    Computes J = d(f_nn) / d(theta): Jacobian of network output w.r.t. all parameters.

    Args:
        x_in: numpy array of shape (INPUT_DIM,)

    Returns:
        Tensor of shape (OUTPUT_DIM, num_params)
    """
    model.eval()
    x_tensor = torch.tensor(x_in, dtype=torch.float32).unsqueeze(0)

    J_list = []
    for i in range(OUTPUT_DIM):
        model.zero_grad()
        out = model(x_tensor)[0, i]
        out.backward(retain_graph=True)
        grad_flat = torch.cat([p.grad.flatten() for p in model.parameters()])
        J_list.append(grad_flat)

    return torch.stack(J_list)
