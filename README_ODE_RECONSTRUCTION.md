# ODE System Reconstruction from High-Dimensional Observations

## Overview

This project demonstrates the feasibility of reconstructing low-dimensional ODE systems from high-dimensional observations using PCA and sparse system identification.

## Approach

1. **Forward Problem**: Simulate a low-D ODE system (2-3 states) and project to high-D observations (10-15 outputs)
2. **Dimensionality Reduction**: Apply PCA to recover latent low-D representation
3. **System Identification**: Use sparse regression (LASSO) to identify polynomial ODE structure
4. **Validation**: Compare reconstructed dynamics with ground truth

## Key Results

### Experiment 1: 2D Lotka-Volterra System

**Configuration:**
- True system: 2 states (predator-prey dynamics)
- Observations: 10 outputs
- Noise level: 1%

**Results:**
- ✅ **PCA variance explained**: 99.88% with 2 components
- ✅ **Mean correlation**: 0.923 (excellent trajectory matching)
- ✅ **Identification R²**: 0.971 (very good equation recovery)
- ✅ Phase portrait topology perfectly preserved

**Key Finding**: PCA successfully recovers the cyclic structure and the sparse regression identifies meaningful polynomial equations, though in the PCA coordinate system rather than the original coordinates.

### Experiment 2: 3D Lorenz System

**Configuration:**
- True system: 3 states (chaotic attractor)
- Observations: 15 outputs
- Noise level: 2%

**Results:**
- ✅ **PCA variance explained**: 98.75% with 3 components
- ✅ **Mean correlation**: 0.933 (excellent trajectory matching)
- ⚠️ **Identification R²**: -1.336 (poor equation recovery)
- ✅ Butterfly attractor structure preserved in phase space

**Key Finding**: PCA successfully recovers the Lorenz attractor structure and correlates strongly with true trajectories, but polynomial sparse identification struggles due to:
1. Chaotic dynamics (sensitive to noise)
2. Challenges in derivative estimation
3. Coordinate transformation from PCA

## Visualizations

### 2D System (Lotka-Volterra)
![2D Results](ode_reconstruction_2d.png)

Shows:
- Top: True states and high-D observations
- Middle: PCA-reduced states and variance explained
- Bottom: Phase portrait comparison (true vs PCA)

### 3D System (Lorenz)
![3D Results](ode_reconstruction_3d.png)

Shows:
- Top: True states time series
- Middle: 3D attractor comparison and variance explained
- Bottom: 2D projections and quality metrics

## Conclusions

### What Works Well:
1. **PCA for dimensionality reduction**: Captures >98% variance with correct number of components
2. **Trajectory correlation**: High correlation (>0.92) between true and reconstructed trajectories
3. **Topology preservation**: Phase space structure is maintained
4. **Non-chaotic systems**: Good equation recovery for simple periodic systems

### Challenges:
1. **Coordinate ambiguity**: PCA finds an arbitrary rotation/scaling of true coordinates
2. **Chaotic systems**: Derivative estimation and identification are difficult
3. **Equation interpretation**: Recovered equations are in PCA space, not original coordinates
4. **Nonlinear observations**: Strong nonlinearities in observation model can degrade performance

### Recommendations for Improvement:
1. **Better derivative estimation**: Use total variation regularization or neural ODE approaches
2. **Alternative identification**: Try neural networks or Gaussian processes instead of polynomials
3. **Inverse PCA mapping**: Develop methods to map identified equations back to original coordinates
4. **Longer trajectories**: More data helps with chaotic systems
5. **Multiple initial conditions**: Sample different parts of phase space

## Files

- `ode_reconstruction_demo.py` - Initial proof-of-concept (2D only)
- `ode_reconstruction_v2.py` - Enhanced version with 2D and 3D systems, metrics
- `requirements.txt` - Python dependencies
- `ode_reconstruction_results.png` - Initial demo results
- `ode_reconstruction_2d.png` - Detailed 2D results
- `ode_reconstruction_3d.png` - Detailed 3D results

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run initial demo (2D only)
python ode_reconstruction_demo.py

# Run enhanced demo (2D + 3D)
python ode_reconstruction_v2.py
```

## Future Directions

1. **SINDy with constraints**: Add physical constraints (energy conservation, etc.)
2. **Ensemble methods**: Run multiple PCA initializations
3. **Time-delay embedding**: For partial observations
4. **Koopman operator theory**: For better handling of nonlinear dynamics
5. **Neural ODEs**: End-to-end learning of latent dynamics

## References

- Brunton et al. (2016) - "Discovering governing equations from data by sparse identification of nonlinear dynamical systems"
- Champion et al. (2019) - "Data-driven discovery of coordinates and governing equations"
- Lusch et al. (2018) - "Deep learning for universal linear embeddings of nonlinear dynamics"
