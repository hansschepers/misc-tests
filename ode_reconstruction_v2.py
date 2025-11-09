"""
Demo V2: Enhanced ODE System Reconstruction

Improvements:
- Compare 2D vs 3D systems
- Add reconstruction quality metrics
- Test robustness to noise
- Improve sparse identification
- Better visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class ODESystem:
    """Various ODE systems for testing"""

    def __init__(self, system_type: str = "lotka_volterra"):
        self.system_type = system_type

        if system_type == "lotka_volterra":
            self.n_states = 2
            self.params = {'alpha': 1.5, 'beta': 1.0, 'gamma': 3.0, 'delta': 1.0}
        elif system_type == "lorenz":
            self.n_states = 3
            self.params = {'sigma': 10.0, 'rho': 28.0, 'beta': 8.0/3.0}
        elif system_type == "rossler":
            self.n_states = 3
            self.params = {'a': 0.2, 'b': 0.2, 'c': 5.7}

    def dynamics(self, t: float, x: np.ndarray) -> np.ndarray:
        """True ODE dynamics"""
        if self.system_type == "lotka_volterra":
            dx0 = self.params['alpha'] * x[0] - self.params['beta'] * x[0] * x[1]
            dx1 = self.params['delta'] * x[0] * x[1] - self.params['gamma'] * x[1]
            return np.array([dx0, dx1])

        elif self.system_type == "lorenz":
            dx0 = self.params['sigma'] * (x[1] - x[0])
            dx1 = x[0] * (self.params['rho'] - x[2]) - x[1]
            dx2 = x[0] * x[1] - self.params['beta'] * x[2]
            return np.array([dx0, dx1, dx2])

        elif self.system_type == "rossler":
            dx0 = -x[1] - x[2]
            dx1 = x[0] + self.params['a'] * x[1]
            dx2 = self.params['b'] + x[2] * (x[0] - self.params['c'])
            return np.array([dx0, dx1, dx2])


class ObservationModel:
    """Maps low-dimensional states to high-dimensional observations"""

    def __init__(self, n_states: int, n_outputs: int = 10, noise_level: float = 0.01):
        self.n_states = n_states
        self.n_outputs = n_outputs
        self.noise_level = noise_level

        # Random observation matrix with normalization
        self.W = np.random.randn(n_outputs, n_states)
        # Normalize columns
        self.W = self.W / np.linalg.norm(self.W, axis=0)

    def observe(self, x: np.ndarray) -> np.ndarray:
        """Map states to observations with noise"""
        # Linear projection
        y = self.W @ x

        # Add small nonlinear mixing (optional, makes it more realistic)
        if self.n_states >= 2:
            y[0] += 0.05 * x[0]**2
            y[1] += 0.05 * x[0] * x[1]

        # Add measurement noise
        y += np.random.randn(self.n_outputs) * self.noise_level

        return y


def simulate_system(ode_system: ODESystem, x0: np.ndarray,
                   t_span: Tuple[float, float], n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the ODE system"""
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(ode_system.dynamics, t_span, x0, t_eval=t_eval,
                   method='RK45', rtol=1e-8, atol=1e-10)
    return sol.t, sol.y.T


def collect_observations(states: np.ndarray, obs_model: ObservationModel) -> np.ndarray:
    """Generate high-dimensional observations from states"""
    n_samples = states.shape[0]
    observations = np.zeros((n_samples, obs_model.n_outputs))

    for i in range(n_samples):
        observations[i] = obs_model.observe(states[i])

    return observations


def estimate_derivatives(states: np.ndarray, t: np.ndarray, method: str = "finite_diff") -> np.ndarray:
    """Estimate time derivatives"""
    if method == "finite_diff":
        derivatives = np.zeros_like(states)
        dt = np.diff(t)

        # Central differences for interior points
        for i in range(1, len(t)-1):
            dt_back = t[i] - t[i-1]
            dt_forward = t[i+1] - t[i]
            derivatives[i] = (states[i+1] - states[i-1]) / (dt_back + dt_forward)

        # Forward/backward for endpoints
        derivatives[0] = (states[1] - states[0]) / dt[0]
        derivatives[-1] = (states[-1] - states[-2]) / dt[-1]

    return derivatives


def build_library(states: np.ndarray, poly_order: int = 2) -> Tuple[np.ndarray, list]:
    """Build a library of candidate functions"""
    n_samples, n_dims = states.shape
    library_terms = []
    term_names = []

    # Constant
    library_terms.append(np.ones((n_samples, 1)))
    term_names.append('1')

    # Linear terms
    for i in range(n_dims):
        library_terms.append(states[:, i:i+1])
        term_names.append(f'x{i}')

    # Quadratic terms
    if poly_order >= 2:
        for i in range(n_dims):
            library_terms.append(states[:, i:i+1]**2)
            term_names.append(f'x{i}^2')

        # Cross terms
        for i in range(n_dims):
            for j in range(i+1, n_dims):
                library_terms.append((states[:, i] * states[:, j]).reshape(-1, 1))
                term_names.append(f'x{i}*x{j}')

    return np.hstack(library_terms), term_names


def sparse_identification(states: np.ndarray, derivatives: np.ndarray,
                         poly_order: int = 2, alpha: float = 0.01,
                         method: str = "lasso") -> Tuple[np.ndarray, list, float]:
    """Identify ODE structure using sparse regression"""
    library, term_names = build_library(states, poly_order)

    # Normalize library for better conditioning
    library_mean = library.mean(axis=0)
    library_std = library.std(axis=0) + 1e-10
    library_normalized = (library - library_mean) / library_std

    n_dims = states.shape[1]
    coefficients = np.zeros((library.shape[1], n_dims))
    scores = []

    for i in range(n_dims):
        if method == "lasso":
            model = Lasso(alpha=alpha, max_iter=5000)
        else:
            model = Ridge(alpha=alpha)

        model.fit(library_normalized, derivatives[:, i])

        # Un-normalize coefficients
        coef_unnormalized = model.coef_ / library_std
        coefficients[:, i] = coef_unnormalized

        # Calculate R² score
        y_pred = library @ coefficients[:, i]
        score = r2_score(derivatives[:, i], y_pred)
        scores.append(score)

    avg_score = np.mean(scores)
    return coefficients, term_names, avg_score


def print_equations(coefficients: np.ndarray, term_names: list,
                   title: str = "Identified Equations", threshold: float = 0.01):
    """Print equations in readable form"""
    print(f"\n{title}")
    print("=" * 60)

    n_dims = coefficients.shape[1]
    for i in range(n_dims):
        terms = []
        for j, coef in enumerate(coefficients[:, i]):
            if abs(coef) > threshold:
                sign = "+" if coef >= 0 else ""
                terms.append(f"{sign}{coef:.3f}*{term_names[j]}")

        if len(terms) == 0:
            eq = f"dx{i}/dt = 0"
        else:
            eq = f"dx{i}/dt = {' '.join(terms)}"
        print(eq)


def compute_reconstruction_metrics(true_states: np.ndarray,
                                   reduced_states: np.ndarray) -> Dict[str, float]:
    """Compute metrics for reconstruction quality"""
    # Normalize both for comparison (PCA has arbitrary scaling/rotation)
    true_norm = (true_states - true_states.mean(axis=0)) / true_states.std(axis=0)
    reduced_norm = (reduced_states - reduced_states.mean(axis=0)) / reduced_states.std(axis=0)

    # Correlation between trajectories
    n_dims = true_states.shape[1]
    correlations = []
    for i in range(n_dims):
        # Find best matching dimension (PCA can permute/rotate)
        max_corr = 0
        for j in range(n_dims):
            corr = abs(np.corrcoef(true_norm[:, i], reduced_norm[:, j])[0, 1])
            max_corr = max(max_corr, corr)
        correlations.append(max_corr)

    return {
        'mean_correlation': np.mean(correlations),
        'min_correlation': np.min(correlations),
    }


def run_experiment(system_type: str = "lotka_volterra", n_outputs: int = 10,
                   noise_level: float = 0.01, n_points: int = 500) -> Dict:
    """Run single experiment"""

    # Create and simulate system
    ode_system = ODESystem(system_type)
    n_states = ode_system.n_states

    if system_type == "lotka_volterra":
        x0 = np.array([4.0, 2.0])
        t_span = (0, 15)
    elif system_type == "lorenz":
        x0 = np.array([1.0, 1.0, 1.0])
        t_span = (0, 20)
    elif system_type == "rossler":
        x0 = np.array([1.0, 1.0, 1.0])
        t_span = (0, 50)

    t, true_states = simulate_system(ode_system, x0, t_span, n_points)

    # Generate observations
    obs_model = ObservationModel(n_states, n_outputs, noise_level)
    observations = collect_observations(true_states, obs_model)

    # Apply PCA
    pca = PCA(n_components=n_states)
    reduced_states = pca.fit_transform(observations)

    # Estimate derivatives
    true_derivatives = estimate_derivatives(true_states, t)
    reduced_derivatives = estimate_derivatives(reduced_states, t)

    # Identify dynamics
    coeffs, terms, score = sparse_identification(
        reduced_states, reduced_derivatives,
        poly_order=2, alpha=0.005, method="lasso"
    )

    # Compute metrics
    metrics = compute_reconstruction_metrics(true_states, reduced_states)
    metrics['identification_r2'] = score
    metrics['explained_variance'] = pca.explained_variance_ratio_

    return {
        'system_type': system_type,
        'n_states': n_states,
        'n_outputs': n_outputs,
        'noise_level': noise_level,
        't': t,
        'true_states': true_states,
        'observations': observations,
        'reduced_states': reduced_states,
        'pca': pca,
        'coefficients': coeffs,
        'term_names': terms,
        'metrics': metrics,
        'true_derivatives': true_derivatives,
        'reduced_derivatives': reduced_derivatives,
    }


def visualize_results(results: Dict, save_path: str):
    """Create comprehensive visualization"""

    system_type = results['system_type']
    n_states = results['n_states']
    t = results['t']
    true_states = results['true_states']
    observations = results['observations']
    reduced_states = results['reduced_states']
    metrics = results['metrics']

    if n_states == 2:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Row 1: Time series
        ax1 = fig.add_subplot(gs[0, :2])
        for i in range(n_states):
            ax1.plot(t, true_states[:, i], '-', label=f'True x{i}', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('State value')
        ax1.set_title(f'True States - {system_type.replace("_", " ").title()}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 2])
        for i in range(min(5, observations.shape[1])):
            ax2.plot(t, observations[:, i], alpha=0.6, linewidth=1)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Observation value')
        ax2.set_title('High-D Observations (sample)')
        ax2.grid(True, alpha=0.3)

        # Row 2: PCA results
        ax3 = fig.add_subplot(gs[1, :2])
        for i in range(n_states):
            ax3.plot(t, reduced_states[:, i], '--', label=f'PCA PC{i}', linewidth=2, alpha=0.8)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Principal component')
        ax3.set_title('PCA-Reduced States')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Variance explained
        ax4 = fig.add_subplot(gs[1, 2])
        exp_var = metrics['explained_variance']
        ax4.bar(range(len(exp_var)), exp_var, alpha=0.7)
        ax4.set_xlabel('Component')
        ax4.set_ylabel('Explained variance ratio')
        ax4.set_title('PCA Explained Variance')
        ax4.set_xticks(range(len(exp_var)))
        ax4.grid(True, alpha=0.3, axis='y')

        # Row 3: Phase portraits
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(true_states[:, 0], true_states[:, 1], 'b-', linewidth=2, alpha=0.7)
        ax5.set_xlabel('x0')
        ax5.set_ylabel('x1')
        ax5.set_title('True Phase Portrait')
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(reduced_states[:, 0], reduced_states[:, 1], 'r--', linewidth=2, alpha=0.7)
        ax6.set_xlabel('PC0')
        ax6.set_ylabel('PC1')
        ax6.set_title('PCA Phase Portrait')
        ax6.grid(True, alpha=0.3)

        # Metrics text
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        metrics_text = f"""Reconstruction Quality:

Mean Correlation: {metrics['mean_correlation']:.3f}
Min Correlation: {metrics['min_correlation']:.3f}
Identification R²: {metrics['identification_r2']:.3f}

Total Var Explained:
{np.sum(exp_var):.4f}
"""
        ax7.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                verticalalignment='center')

    else:  # 3D system
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        # Time series
        ax1 = fig.add_subplot(gs[0, :])
        for i in range(n_states):
            ax1.plot(t, true_states[:, i], '-', label=f'True x{i}', alpha=0.7)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('State value')
        ax1.set_title(f'True States - {system_type.replace("_", " ").title()}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 3D phase portrait - true
        ax2 = fig.add_subplot(gs[1, 0], projection='3d')
        ax2.plot(true_states[:, 0], true_states[:, 1], true_states[:, 2],
                'b-', linewidth=1, alpha=0.6)
        ax2.set_xlabel('x0')
        ax2.set_ylabel('x1')
        ax2.set_zlabel('x2')
        ax2.set_title('True 3D Trajectory')

        # 3D phase portrait - PCA
        ax3 = fig.add_subplot(gs[1, 1], projection='3d')
        ax3.plot(reduced_states[:, 0], reduced_states[:, 1], reduced_states[:, 2],
                'r-', linewidth=1, alpha=0.6)
        ax3.set_xlabel('PC0')
        ax3.set_ylabel('PC1')
        ax3.set_zlabel('PC2')
        ax3.set_title('PCA 3D Trajectory')

        # Variance explained
        ax4 = fig.add_subplot(gs[1, 2])
        exp_var = metrics['explained_variance']
        ax4.bar(range(len(exp_var)), exp_var, alpha=0.7)
        ax4.set_xlabel('Component')
        ax4.set_ylabel('Explained variance')
        ax4.set_title('PCA Explained Variance')
        ax4.set_xticks(range(len(exp_var)))
        ax4.grid(True, alpha=0.3, axis='y')

        # 2D projections
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(true_states[:, 0], true_states[:, 1], 'b-', linewidth=1, alpha=0.6)
        ax5.set_xlabel('x0')
        ax5.set_ylabel('x1')
        ax5.set_title('True (x0-x1 projection)')
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(reduced_states[:, 0], reduced_states[:, 1], 'r-', linewidth=1, alpha=0.6)
        ax6.set_xlabel('PC0')
        ax6.set_ylabel('PC1')
        ax6.set_title('PCA (PC0-PC1 projection)')
        ax6.grid(True, alpha=0.3)

        # Metrics
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        metrics_text = f"""Reconstruction Quality:

Mean Correlation: {metrics['mean_correlation']:.3f}
Min Correlation: {metrics['min_correlation']:.3f}
Identification R²: {metrics['identification_r2']:.3f}

Total Var Explained:
{np.sum(exp_var):.4f}
"""
        ax7.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                verticalalignment='center')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")


def main():
    """Run experiments"""

    print("="*70)
    print("ODE System Reconstruction - Enhanced Demo (V2)")
    print("="*70)

    # Experiment 1: 2D system (Lotka-Volterra)
    print("\n" + "="*70)
    print("EXPERIMENT 1: 2D Lotka-Volterra System")
    print("="*70)

    results_2d = run_experiment(
        system_type="lotka_volterra",
        n_outputs=10,
        noise_level=0.01,
        n_points=500
    )

    print(f"\nPCA Explained Variance: {results_2d['metrics']['explained_variance']}")
    print(f"Mean Correlation: {results_2d['metrics']['mean_correlation']:.3f}")
    print(f"Identification R²: {results_2d['metrics']['identification_r2']:.3f}")

    print_equations(results_2d['coefficients'], results_2d['term_names'],
                   title="Identified Equations (PCA space)", threshold=0.01)

    visualize_results(results_2d, 'ode_reconstruction_2d.png')

    # Experiment 2: 3D system (Lorenz)
    print("\n" + "="*70)
    print("EXPERIMENT 2: 3D Lorenz System")
    print("="*70)

    results_3d = run_experiment(
        system_type="lorenz",
        n_outputs=15,
        noise_level=0.02,
        n_points=800
    )

    print(f"\nPCA Explained Variance: {results_3d['metrics']['explained_variance']}")
    print(f"Mean Correlation: {results_3d['metrics']['mean_correlation']:.3f}")
    print(f"Identification R²: {results_3d['metrics']['identification_r2']:.3f}")

    print_equations(results_3d['coefficients'], results_3d['term_names'],
                   title="Identified Equations (PCA space)", threshold=0.05)

    visualize_results(results_3d, 'ode_reconstruction_3d.png')

    print("\n" + "="*70)
    print("All experiments complete!")
    print("="*70)


if __name__ == "__main__":
    main()
