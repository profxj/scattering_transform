# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The `scattering_transform` package provides fast implementations of the **scattering transform** for texture analysis in 2D fields (images) and 1D signals. It computes statistical descriptors that capture non-Gaussian structures beyond what power spectra can measure. The package is designed for scientific applications, particularly oceanographic and astronomical data analysis.

Key features:
- **5-10x faster** than kymatio for global scattering coefficients
- Supports both **analysis** (computing scattering statistics) and **synthesis** (generating new images with target statistics)
- GPU/CPU support with automatic device detection
- Multiple wavelet types (Morlet, bump-steerable, Gaussian, Shannon)

## Common Commands

### Running Analysis Scripts

```bash
# Run example with ST.py module
python example.py

# Run oceanographic data processing
python runs/MODIS/modis_coeff.py
python runs/VIIRS/viirs_coeff.py
```

### Interactive Development

```bash
# Launch Jupyter notebooks for exploration
jupyter notebook scattering.ipynb  # Main tutorial
jupyter notebook ST_image_synthesis.ipynb  # Synthesis examples
jupyter notebook scattering_modis_test.ipynb  # Real-world applications
```

### Testing

There is no formal test suite. Testing is done through:
- Jupyter notebooks with visual validation
- Comparison between `'fast'` and `'classic'` algorithms in ST.py
- Real data workflows in `runs/` directory

## Code Architecture

### Two-Layer Design

**Layer 1: ST.py (Core Engine)**
- Standalone module with only `numpy` and `torch>=1.7` dependencies
- Can be used independently from the scattering package
- Implements low-level fast algorithms for scattering transform
- Classes: `ST_2D`, `ST_1D`, `FiltersSet`, `Bispectrum_Calculator`

**Layer 2: scattering/ Package (High-Level API)**
- Built on top of ST.py concepts with user-friendly interface
- Adds synthesis capabilities, I/O utilities, preprocessing
- Main classes: `Scattering2d`, `AlphaScattering2d_cov`, `FiltersSet`

### Key Module Relationships

```
scattering/__init__.py
├── synthesis()                  # Main synthesis function
└── Scattering2d                 # Main analysis class
    ├── FiltersSet               # Wavelet generation
    ├── scattering_coef()        # Compute S0, S1, S2 coefficients
    ├── scattering_cov()         # Compute covariances (P00, C01, C11, P11)
    └── backend/                 # Device abstraction (torch, skcuda)

AlphaScattering2d_cov            # Phase-harmonic correlations
    └── forward()                # Compute alpha coefficients

ST.py (standalone)
├── ST_2D                        # Low-level 2D scattering
├── FiltersSet                   # Generate wavelet banks
└── Bispectrum_Calculator        # Binned bispectrum
```

### Scattering Coefficient Hierarchy

The scattering transform produces a hierarchy of coefficients:

- **S0**: 0th order (global mean) - shape `[N_image, 1]`
- **S1**: 1st order (wavelet modulus means) - shape `[N_image, J, L]`
  - `J` = number of scales (dyadic scales)
  - `L` = number of orientations
- **S2**: 2nd order (modulus of modulus) - shape `[N_image, J, L, J, L]`
  - Applied wavelets at two scales/orientations

**Covariance coefficients** (from `scattering_cov()`):
- **P00**: Power spectrum (orig × orig) - linear correlations
- **C01**: Original × modulus correlations - shape `[N_image, J, J, L]`
- **P11**: Modulus auto-correlations (j1=j2) - shape `[N_image, J, J, L, L]`
- **C11**: General modulus × modulus correlations - shape `[N_image, J, J, J, L, L, L]`

**Isotropic variants**: Averaged over orientations (suffix `_iso`), e.g., `S1_iso`, `C11_iso`

### Analysis Workflow

```python
import scattering

# 1. Create calculator with image dimensions and wavelet parameters
st_calc = scattering.Scattering2d(M=256, N=256, J=5, L=4, device='gpu')

# 2. Compute scattering coefficients
# For mean coefficients only:
result = st_calc.scattering_coef(images)  # Returns S0, S1, S2

# For covariance coefficients:
result = st_calc.scattering_cov(images)   # Returns dict with P00, S1, C01, P11, C11

# 3. Access specific coefficients
S1 = result['S1']              # 1st order
C11_iso = result['C11_iso']    # Isotropic modulus covariances
```

**Key parameters:**
- `M, N`: Image dimensions (must be power of 2 for efficiency)
- `J`: Number of dyadic scales (typically 5-8)
- `L`: Number of orientations (typically 4 or 8)
- `wavelets`: Type of wavelet (`'morlet'`, `'BS'`, `'gau'`, `'shannon'`)
- `device`: `'gpu'` or `'cpu'` (auto-detected if not specified)

**Advanced options:**
- `if_large_batch=True`: Memory-efficient mode for large datasets
- `C11_criteria='j2>=j1'`: Control which scale pairs to compute
- `remove_edge=True`: Mitigate edge effects
- `pseudo_coef=1`: Generalized scattering parameter (1 = standard)

### Synthesis Workflow

The synthesis process uses gradient descent to generate images that match target statistics:

```python
# Mode 1: Match statistics from target image(s)
image_syn = scattering.synthesis(
    estimator_name='s_cov_iso',  # Which statistics to match
    target=image_input,           # Target image(s)
    mode='image',
    M=256, N=256, J=5, L=4
)

# Mode 2: Specify target coefficient values directly
image_syn = scattering.synthesis(
    estimator_name='s_cov_iso',
    target=target_coefficients,   # Pre-computed coefficients
    mode='estimator',
    M=256, N=256, J=5, L=4,
    steps=400,                    # Optimization iterations
    learning_rate=0.5,
    optimizer='LBFGS'             # or 'Adam', 'NAdam'
)
```

**Common estimator names:**
- `'s_mean'`: Match S0, S1, S2 only
- `'s_cov_iso'`: Match isotropic covariances (most common)
- `'s_cov'`: Match full covariances (slower)
- `'alpha_cov'`: Match alpha-phase correlations
- `'all'`: Combine scattering + power spectrum + bispectrum

**Synthesis happens in two possible domains:**
- Spatial domain (default): Direct pixel optimization
- Fourier domain (`Fourier=True`): Optimize Fourier coefficients then transform back

### Backend and Device Management

The package uses a backend abstraction layer in `scattering/backend/`:

**Backend selection priority:**
1. Environment variable: `export SCATTERING_BACKEND_2D='torch'`
2. Config file: `~/.config/scattering/scattering.cfg`
3. Default: `'torch'`

**Available backends:**
- `'torch'`: Pure PyTorch (default, most portable)
- `'skcuda'`: CUDA-optimized via scikit-cuda

**Device handling:**
- Automatic GPU detection: Uses GPU if `torch.cuda.is_available()`
- Manual override: `device='cpu'` or `device='gpu'` in calculator initialization
- Tensors automatically moved to appropriate device

### ST.py Low-Level API

For direct use of the standalone ST.py module:

```python
import ST

# 1. Generate wavelet filters
J, L, M, N = 8, 4, 512, 512
filter_set = ST.FiltersSet(M, N, J, L).generate_morlet(precision='single')

# 2. Create calculator
ST_calculator = ST.ST_2D(filter_set, J, L, device='gpu')

# 3. Compute scattering coefficients
images = np.empty((30, M, N), dtype=np.float32)
S, S0, S1, S2, _, _, _, _ = ST_calculator.forward(
    images, J, L,
    j1j2_criteria='j2>j1',  # Only compute j2 > j1
    algorithm='fast'         # or 'classic' for verification
)

# 4. Compute phase harmonics (alpha correlations)
PH = ST_calculator.phase_harmonics(images, J, L)
# Returns dict with C00, C01, P11, C11, and isotropic variants
```

**Algorithm modes:**
- `'fast'`: Uses frequency space truncation (5-10x faster)
- `'classic'`: Full Fourier computation (for verification)

**Memory management:**
- `if_large_batch=True`: Sequential processing for large batches
- `if_large_batch=False`: Maximum parallelization (default)

### Preprocessing Utilities

```python
from scattering import remove_slope, whiten, binning2x2

# Remove linear trends (reduces edge effects)
images_clean = remove_slope(images)

# Standardize (zero mean, unit variance)
images_norm = whiten(images)

# Downsample by factor of 2
images_small = binning2x2(images)
```

### Common Wavelet Types

Specified via `wavelets` parameter:
- `'morlet'` (default): Off-center Gaussians in Fourier space
- `'BS'`: Bump-steerable wavelets
- `'gau'`: Gaussian with uniform orientation coverage
- `'shannon'`: Top-hat in Fourier space
- `'gau_harmonic'`: Harmonic angular profiles

Each has different properties for edge effects and orientation selectivity.

## Important Implementation Details

### Coordinate Conventions
- Images are `[N_image, M, N]` where `M` = height, `N` = width
- Wavelet scales indexed by `j1, j2, j3` (dyadic: 2^j pixels)
- Orientations indexed by `l1, l2, l3` (angles: 2π * l / L)

### Scale Selection Criteria
- `'j2>j1'`: Only compute where second scale finer than first (typical for S2)
- `'j2>=j1'`: Include equal scales (typical for C11)
- `'j1<=j2<j3'`: Three-scale criteria for C11

### Normalization Options
When computing covariances:
- `normalization='P00'`: Normalize by power spectrum (default)
- `normalization='P11'`: Normalize by modulus field power
- `normalization=None`: No normalization

### Edge Effects
Edge effects arise from finite image boundaries:
- Use `remove_edge=True` in `scattering_cov()` to mitigate
- Apply `remove_slope()` preprocessing to reduce boundary discontinuities
- Use appropriate wavelets (e.g., bump-steerable have better edge properties)

### Performance Tips
- Use power-of-2 image dimensions (256, 512, 1024) for FFT efficiency
- Set `if_large_batch=True` when N_images > ~100
- Use `device='gpu'` for large J, L, or image sizes
- Cache filter sets to avoid regeneration: `filter_set.generate_morlet(if_save=True)`

## File Organization

```
scattering_transform/
├── ST.py                        # Standalone low-level module
├── scattering/                  # Main package
│   ├── __init__.py             # High-level API (synthesis, etc.)
│   ├── Scattering2d.py         # Main 2D scattering class
│   ├── AlphaScattering2d_cov.py # Phase harmonics
│   ├── FiltersSet.py           # Wavelet generation
│   ├── polyspectra_calculators.py # Bispectrum
│   ├── backend/                # Device abstraction
│   └── utils.py                # Preprocessing utilities
├── runs/                       # Application scripts
│   ├── MODIS/                  # MODIS satellite data
│   ├── VIIRS/                  # VIIRS satellite data
│   └── cutout_utils.py         # Data extraction utilities
├── nb/                         # Jupyter notebooks for analysis
├── example.py                  # Basic usage example
└── *.ipynb                     # Tutorial notebooks
```

## Development Notes

### When adding new wavelet types:
1. Add generation method to `FiltersSet` class in `scattering/FiltersSet.py`
2. Ensure wavelets are complex tensors with shape `[J, L, M, N]`
3. Test with both `'fast'` and `'classic'` algorithms for consistency

### When adding new estimators for synthesis:
1. Define estimator function in `scattering/__init__.py`
2. Function signature: `estimator(images) -> dict of coefficients`
3. Add to `estimator_name` options in `synthesis()`
4. Ensure differentiability for gradient-based optimization

### When working with oceanographic data:
- Check `runs/MODIS/` and `runs/VIIRS/` for satellite data processing patterns
- Typical workflow: Load HDF5/NetCDF → Extract cutouts → Preprocess → Compute scattering
- Use `cutout_utils.py` for extracting spatial patches

### Debugging synthesis:
- Use `show=True` to visualize intermediate results
- Compare different optimizers (`LBFGS`, `Adam`, `NAdam`)
- Try `Fourier=True` if spatial synthesis is unstable
- Check loss convergence; plateau indicates matching statistics

## References

This implementation is optimized for speed compared to the `kymatio` package while maintaining flexibility. The mathematical framework is based on:
- Mallat, S. (2012). "Group Invariant Scattering"
- Bruna & Mallat (2013). "Invariant Scattering Convolution Networks"
- Phase harmonics: Mallat et al. (2020)

For questions, contact: scheng@ias.edu
