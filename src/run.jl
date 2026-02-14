
function hydra(
		X::AbstractMatrix,
		Y::AbstractVector;
		config::HydraConfig = HydraConfig(),
		covariates::Union{Nothing, AbstractMatrix} = nothing,
	)
	X = Float64.(X)
	Y = Int.(Y)
	mu = mean(X, dims = 1)
	sigma = std(X, dims = 1)
	sigma[sigma .== 0] .= 1.0
	X = (X .- mu) ./ sigma
	if covariates !== nothing
		covariates = Float64.(covariates)
		cov_mu = mean(covariates, dims = 1)
		cov_sigma = std(covariates, dims = 1)
		cov_sigma[cov_sigma .== 0] .= 1.0
		covariates = (covariates .- cov_mu) ./ cov_sigma
		X = residualize_covariates(X, Y, covariates)
	end
	cluster_values = collect(config.cluster_range)
	n = size(X, 1)
	patient_mask = Y .== 1
	fold_ids = make_cv_partition(n, config.num_folds)
	fold_assignments = Dict{Int, Matrix{Int}}()
	for K in cluster_values
		fold_assignments[K] = zeros(Int, n, config.num_folds)
	end
	for fold in 1:config.num_folds
		train = fold_ids .!= fold
		for K in cluster_values
			result = solve_hydra(X[train, :], Y[train], K, config)
			fold_assignments[K][train, fold] = result.assignments
		end
	end
	ari_values = Vector{Float64}(undef, length(cluster_values))
	for (i, K) in enumerate(cluster_values)
		patient_folds = fold_assignments[K][patient_mask, :]
		ari_values[i] = cross_validation_stability(patient_folds)
	end
	final_assignments = -ones(Int, n, length(cluster_values))
	for (i, K) in enumerate(cluster_values)
		patient_folds = fold_assignments[K][patient_mask, :]
		final_assignments[patient_mask, i] = consensus_clustering(patient_folds, K)
	end
	return HydraResult(final_assignments, ari_values, config.cluster_range)
end

function make_cv_partition(n::Integer, num_folds::Integer)
	base_size = div(n, num_folds)
	remainder = mod(n, num_folds)
	fold_ids = Vector{Int}(undef, n)
	pos = 1
	for fold in 1:num_folds
		fold_size = base_size + (fold <= remainder ? 1 : 0)
		fold_ids[pos:pos+fold_size-1] .= fold
		pos += fold_size
	end
	return shuffle!(fold_ids)
end

