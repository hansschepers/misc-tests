"""
Demo: ODE System Reconstruction from High-Dimensional Observations

This script demonstrates:
1. Simulating a low-dimensional ODE system (2-3 states)
2. Projecting to high-dimensional observations (10 outputs)
3. Using PCA to reduce dimensionality
4. Reconstructing the ODE structure from reduced data
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from typing import Tuple

np.random.seed(42)


class ODESystem:
    """A simple 2D or 3D nonlinear ODE system"""

    def __init__(self, n_states: int = 2):
        self.n_states = n_states

        if n_states == 2:
            # Lotka-Volterra-like system
            self.params = {'alpha': 1.5, 'beta': 1.0, 'gamma': 3.0, 'delta': 1.0}
        else:  # 3 states
            # Simple 3D chaotic-like system
            self.params = {'sigma': 10.0, 'rho': 8.0, 'beta': 2.66}

    def dynamics(self, t: float, x: np.ndarray) -> np.ndarray:
        """True ODE dynamics"""
        if self.n_states == 2:
            # Lotka-Volterra predator-prey
            dx0 = self.params['alpha'] * x[0] - self.params['beta'] * x[0] * x[1]
            dx1 = self.params['delta'] * x[0] * x[1] - self.params['gamma'] * x[1]
            return np.array([dx0, dx1])
        else:
            # Simplified Lorenz-like system
            dx0 = self.params['sigma'] * (x[1] - x[0])
            dx1 = x[0] * (self.params['rho'] - x[2]) - x[1]
            dx2 = x[0] * x[1] - self.params['beta'] * x[2]
            return np.array([dx0, dx1, dx2])


class ObservationModel:
    """Maps low-dimensional states to high-dimensional observations"""

    def __init__(self, n_states: int, n_outputs: int = 10):
        self.n_states = n_states
        self.n_outputs = n_outputs

        # Random observation matrix with some structure
        # Mix of linear and weakly nonlinear observations
        self.W = np.random.randn(n_outputs, n_states) * 0.5
        self.noise_level = 0.01

    def observe(self, x: np.ndarray) -> np.ndarray:
        """Map states to observations with small noise"""
        # Linear projection
        y = self.W @ x

        # Add small nonlinear terms (optional)
        if self.n_states == 2:
            y[0] += 0.1 * x[0]**2
            y[1] += 0.1 * x[0] * x[1]

        # Add measurement noise
        y += np.random.randn(self.n_outputs) * self.noise_level

        return y


def simulate_system(ode_system: ODESystem, x0: np.ndarray,
                   t_span: Tuple[float, float], n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the ODE system"""
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(ode_system.dynamics, t_span, x0, t_eval=t_eval, method='RK45')
    return sol.t, sol.y.T


def collect_observations(states: np.ndarray, obs_model: ObservationModel) -> np.ndarray:
    """Generate high-dimensional observations from states"""
    n_samples = states.shape[0]
    observations = np.zeros((n_samples, obs_model.n_outputs))

    for i in range(n_samples):
        observations[i] = obs_model.observe(states[i])

    return observations


def reconstruct_with_pca(observations: np.ndarray, n_components: int) -> Tuple[PCA, np.ndarray]:
    """Apply PCA to reduce dimensionality"""
    pca = PCA(n_components=n_components)
    reduced_states = pca.fit_transform(observations)

    print(f"\nPCA Analysis:")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative variance: {np.cumsum(pca.explained_variance_ratio_)}")

    return pca, reduced_states


def estimate_derivatives(states: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Estimate time derivatives using finite differences"""
    dt = t[1] - t[0]
    # Central differences for interior points
    derivatives = np.zeros_like(states)
    derivatives[1:-1] = (states[2:] - states[:-2]) / (2 * dt)
    # Forward/backward for endpoints
    derivatives[0] = (states[1] - states[0]) / dt
    derivatives[-1] = (states[-1] - states[-2]) / dt

    return derivatives


def build_library(states: np.ndarray, poly_order: int = 2) -> np.ndarray:
    """Build a library of candidate functions (polynomial terms)"""
    n_samples, n_dims = states.shape

    library_terms = [np.ones((n_samples, 1))]  # Constant term

    # Linear terms
    for i in range(n_dims):
        library_terms.append(states[:, i:i+1])

    # Quadratic terms
    if poly_order >= 2:
        for i in range(n_dims):
            library_terms.append(states[:, i:i+1]**2)

        # Cross terms
        for i in range(n_dims):
            for j in range(i+1, n_dims):
                library_terms.append((states[:, i] * states[:, j]).reshape(-1, 1))

    return np.hstack(library_terms)


def identify_dynamics(states: np.ndarray, derivatives: np.ndarray,
                      poly_order: int = 2, alpha: float = 0.01) -> np.ndarray:
    """Identify ODE structure using sparse regression"""
    library = build_library(states, poly_order)

    print(f"\nLibrary shape: {library.shape}")

    # Fit coefficients for each state derivative
    n_dims = states.shape[1]
    coefficients = np.zeros((library.shape[1], n_dims))

    for i in range(n_dims):
        model = Ridge(alpha=alpha)
        model.fit(library, derivatives[:, i])
        coefficients[:, i] = model.coef_

    return coefficients


def print_identified_equations(coefficients: np.ndarray, n_dims: int):
    """Print the identified equations in readable form"""
    print("\nIdentified ODE Structure:")

    term_names = ['1']
    for i in range(n_dims):
        term_names.append(f'x{i}')

    for i in range(n_dims):
        term_names.append(f'x{i}^2')

    if n_dims == 2:
        term_names.append('x0*x1')
    elif n_dims == 3:
        term_names.extend(['x0*x1', 'x0*x2', 'x1*x2'])

    for i in range(n_dims):
        eq = f"dx{i}/dt = "
        terms = []
        for j, coef in enumerate(coefficients[:, i]):
            if abs(coef) > 0.01:  # Only show significant terms
                terms.append(f"{coef:+.3f}*{term_names[j]}")
        eq += " ".join(terms)
        print(eq)


def main():
    """Run the full reconstruction demo"""

    print("="*60)
    print("ODE System Reconstruction Demo")
    print("="*60)

    # Configuration
    n_states = 2  # Try 2 or 3
    n_outputs = 10
    n_points = 500
    t_span = (0, 10)

    print(f"\nConfiguration:")
    print(f"True system dimension: {n_states}")
    print(f"Observation dimension: {n_outputs}")

    # Step 1: Create and simulate true system
    print("\n" + "-"*60)
    print("Step 1: Simulating true ODE system...")
    ode_system = ODESystem(n_states=n_states)

    if n_states == 2:
        x0 = np.array([4.0, 2.0])
    else:
        x0 = np.array([1.0, 1.0, 1.0])

    t, true_states = simulate_system(ode_system, x0, t_span, n_points)
    print(f"Simulated {n_points} time points")
    print(f"True states shape: {true_states.shape}")

    # Step 2: Generate high-dimensional observations
    print("\n" + "-"*60)
    print("Step 2: Generating high-dimensional observations...")
    obs_model = ObservationModel(n_states, n_outputs)
    observations = collect_observations(true_states, obs_model)
    print(f"Observations shape: {observations.shape}")

    # Step 3: Apply PCA
    print("\n" + "-"*60)
    print("Step 3: Applying PCA for dimensionality reduction...")
    pca, reduced_states = reconstruct_with_pca(observations, n_components=n_states)

    # Step 4: Estimate derivatives
    print("\n" + "-"*60)
    print("Step 4: Estimating time derivatives...")
    reduced_derivatives = estimate_derivatives(reduced_states, t)

    # Step 5: Identify dynamics
    print("\n" + "-"*60)
    print("Step 5: Identifying ODE structure...")
    coefficients = identify_dynamics(reduced_states, reduced_derivatives, poly_order=2)
    print_identified_equations(coefficients, n_states)

    # Step 6: Visualize results
    print("\n" + "-"*60)
    print("Step 6: Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: True states time series
    ax = axes[0, 0]
    for i in range(n_states):
        ax.plot(t, true_states[:, i], label=f'True x{i}', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('State value')
    ax.set_title('True States (Low-Dimensional)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Sample of high-dimensional observations
    ax = axes[0, 1]
    for i in range(min(5, n_outputs)):
        ax.plot(t, observations[:, i], label=f'Y{i}', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Observation value')
    ax.set_title('High-Dimensional Observations (Sample)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: PCA reduced states
    ax = axes[1, 0]
    for i in range(n_states):
        ax.plot(t, reduced_states[:, i], label=f'PCA PC{i}', linewidth=2, linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Principal component')
    ax.set_title('PCA-Reduced States')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Phase portrait comparison
    ax = axes[1, 1]
    if n_states == 2:
        ax.plot(true_states[:, 0], true_states[:, 1], 'b-', label='True', linewidth=2, alpha=0.7)
        ax.plot(reduced_states[:, 0], reduced_states[:, 1], 'r--', label='PCA', linewidth=2, alpha=0.7)
        ax.set_xlabel('x0 / PC0')
        ax.set_ylabel('x1 / PC1')
    else:  # 3D
        from mpl_toolkits.mplot3d import Axes3D
        # For 3D, just plot first two dimensions
        ax.plot(true_states[:, 0], true_states[:, 1], 'b-', label='True (x0-x1)', linewidth=2, alpha=0.7)
        ax.plot(reduced_states[:, 0], reduced_states[:, 1], 'r--', label='PCA (PC0-PC1)', linewidth=2, alpha=0.7)
        ax.set_xlabel('x0 / PC0')
        ax.set_ylabel('x1 / PC1')

    ax.set_title('Phase Portrait Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ode_reconstruction_results.png', dpi=150, bbox_inches='tight')
    print("Results saved to 'ode_reconstruction_results.png'")

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == "__main__":
    main()
