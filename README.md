# Hydra

[![Build Status](https://github.com/myersm0/Hydra.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/myersm0/Hydra.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Overview
A Julia package implementing **HYDRA** (Heterogeneity through Discriminative Analysis), a method for simultaneous classification and subtype identification in case-control studies. Given a set of features for controls and patients, HYDRA fits a convex polytope of multiple max-margin hyperplanes to separate the two groups; each face of the polytope implicitly defines a disease subtype.

Based on: Varol, Sotiras, Davatzikos, and the Alzheimer's Disease Neuroimaging Initiative. "HYDRA: Revealing heterogeneity of imaging and genetic patterns through a multiple max-margin discriminative analysis framework." [NeuroImage 145 (2017): 346-364](https://pmc.ncbi.nlm.nih.gov/articles/PMC5408358/).

The original implementation is in MATLAB: ([evarol/HYDRA](https://github.com/evarol/HYDRA)).

> Note: Not to be confused with the FluxML registered package [Hydra.jl](https://github.com/FluxML/Hydra.jl), which is entirely different!

## Installation
This package is not registered as it clashes with the other Hydra.jl already in the registry. Install directly from the repository.

Within Julia (version >= 1.12):
```julia
using Pkg
Pkg.add(url="https://github.com/myersm0/Hydra.jl")
```

## Usage
### Basic workflow
1. Prepare a feature matrix `X` (subjects × features) and a label vector `Y` where controls are `-1` and patients are `1`.
2. Call `hydra(X, Y)` with a `HydraConfig`.
3. Inspect the result using `ari()` to find the best number of clusters, and `assignments()` to get cluster labels.

```julia
using Hydra
using Clustering  # for assignments()

X = ...  # your feature matrix, n × d
Y = ...  # label vector: -1 for controls, 1 for patients

result = hydra(X, Y; config = HydraConfig(
    cluster_range = 1:1:5,
    num_folds = 5,
    num_consensus = 20,
))

# find the best k by ARI peak
ari(result)          # vector of ARI values across cluster_range
ari(result, 3)       # ARI for a specific k

# get cluster assignments
assignments(result, 3)   # vector of assignments at k=3; -1 for controls
```

### Covariate correction
If you have covariates (e.g. age, sex) to regress out before clustering, pass them as a matrix:
```julia
covariates = ...  # n × p matrix of covariates

result = hydra(X, Y; config = HydraConfig(), covariates = covariates)
```
Correction is done via GLM residualization using controls only, following the approach in the original paper.

### Configuration
`HydraConfig` accepts the following keyword arguments:

| Parameter | Default | Description |
|:--|:--|:--|
| `regularization` | `L2Regularization()` | `L1Regularization()` for sparsity |
| `c` | `0.25` | SVM regularization cost |
| `balance_classes` | `true` | weight classes by inverse frequency |
| `initialization` | `DPPHyperplaneInit()` | also: `KMeansInit()`, `RandomHyperplaneInit()`, `RandomAssignmentInit()` |
| `num_iterations` | `50` | max iterations per consensus run |
| `num_consensus` | `20` | number of consensus clustering runs |
| `cluster_range` | `1:1:10` | range of k values to evaluate |
| `num_folds` | `10` | cross-validation folds |

### Note about preprocessing
Features are z-scored internally. If you have covariates, those are also z-scored before residualization. Beyond that, it's left to the user to handle any domain-specific preprocessing (e.g. ROI extraction from neuroimaging data).

## Dependencies
Core solver uses [Clarabel.jl](https://github.com/oxfordcontrol/Clarabel.jl) for the weighted SVM quadratic programs. Consensus clustering and initialization use [Clustering.jl](https://github.com/JuliaStats/Clustering.jl). The `assignments()` accessor extends `Clustering.assignments`.

## License
GPL-3.0, following the original MATLAB implementation.


