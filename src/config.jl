abstract type RegularizationType end
struct L1Regularization <: RegularizationType end
struct L2Regularization <: RegularizationType end

abstract type InitializationType end
struct RandomHyperplaneInit <: InitializationType end
struct RandomAssignmentInit <: InitializationType end
struct KMeansInit <: InitializationType end
struct DPPHyperplaneInit <: InitializationType end

struct HydraConfig
	regularization::RegularizationType
	c::Float64
	balance_classes::Bool
	initialization::InitializationType
	num_iterations::Int
	num_consensus::Int
	cluster_range::StepRange{Int, Int}
	num_folds::Int
end

function HydraConfig(;
		regularization::RegularizationType = L2Regularization(),
		c::Float64 = 0.25,
		balance_classes::Bool = true,
		initialization::InitializationType = DPPHyperplaneInit(),
		num_iterations::Integer = 50,
		num_consensus::Integer = 20,
		cluster_range::StepRange{<:Integer, <:Integer} = 1:1:10,
		num_folds::Integer = 10,
	)
	return HydraConfig(
		regularization, c, balance_classes, initialization,
		num_iterations, num_consensus, cluster_range, num_folds,
	)
end

struct SVMModel
	w::Vector{Float64}
	b::Float64
end

struct SolverResult
	assignments::Vector{Int}
	models::Vector{SVMModel}
	W::Matrix{Float64}
end

struct HydraResult
	assignments::Matrix{Int}
	ari::Vector{Float64}
	cluster_range::StepRange{Int, Int}
end

function ari(result::HydraResult)
	return result.ari
end

function ari(result::HydraResult, k::Integer)
	index = findfirst(==(k), result.cluster_range)
	isnothing(index) && error("k=$k not in cluster_range $(result.cluster_range)")
	return result.ari[index]
end

function Clustering.assignments(result::HydraResult)
	return result.assignments
end

function Clustering.assignments(result::HydraResult, k::Integer)
	index = findfirst(==(k), result.cluster_range)
	isnothing(index) && error("k=$k not in cluster_range $(result.cluster_range)")
	return result.assignments[:, index]
end
