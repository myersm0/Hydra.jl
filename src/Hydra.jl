module Hydra

using Clarabel
using Clustering: kmeans, assignments
using Distributions: Gamma
using LinearAlgebra
using Random: shuffle!
using SparseArrays
using Statistics: mean, std

import Clustering

include("config.jl")
export HydraConfig, HydraResult, ari

include("svm.jl")
export L1Regularization, L2Regularization

include("init.jl")
export RandomHyperplaneInit, RandomAssignmentInit, KMeansInit, DPPHyperplaneInit

include("consensus.jl")

include("covariate.jl")

include("solver.jl")

include("run.jl")
export hydra

end
