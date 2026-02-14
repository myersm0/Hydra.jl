
function solve_hydra(
		X::AbstractMatrix,
		Y::AbstractVector,
		K::Integer,
		config::HydraConfig,
	)
	n = size(X, 1)
	idx_p = findall(Y .== 1)
	n_p = length(idx_p)
	IDX = zeros(Int, n_p, config.num_consensus)
	for run in 1:config.num_consensus
		W = ones(n, K) ./ K
		W[Y .== 1, :] = initialize(X, Y, K, config.initialization)
		S = zeros(n, K)
		for iter in 1:config.num_iterations
			for j in 1:K
				cp, cn = class_weights(W, Y, j, config.balance_classes)
				model = solve_weighted_svm(
					X, Y, W[:, j], config.c, cp, cn, config.regularization,
				)
				S[:, j] = predict_svm(X, model)
			end
			idx = argmax.(eachrow(S[Y .== 1, :]))
			W_old = copy(W)
			W[Y .== 1, :] .= 0.0
			for (i, j) in enumerate(idx)
				W[idx_p[i], j] = 1.0
			end
			if norm(W .- W_old) < 1e-6
				break
			end
		end
		IDX[:, run] = argmax.(eachrow(S[Y .== 1, :]))
	end
	if config.num_consensus > 1
		final_idx = consensus_clustering(IDX, K)
	else
		final_idx = IDX[:, 1]
	end
	W = zeros(n, K)
	for (i, j) in enumerate(final_idx)
		W[idx_p[i], j] = 1.0
	end
	W[Y .== -1, :] .= 1.0 / K
	models = Vector{SVMModel}(undef, K)
	for j in 1:K
		cp, cn = class_weights(W, Y, j, config.balance_classes)
		models[j] = solve_weighted_svm(
			X, Y, W[:, j], config.c, cp, cn, config.regularization,
		)
	end
	Yhat = copy(Y)
	Yhat[Y .== 1] = final_idx
	return SolverResult(Yhat, models, W)
end

function class_weights(
		W::AbstractMatrix,
		Y::AbstractVector,
		j::Integer,
		balance::Bool,
	)
	if !balance
		return 1.0, 1.0
	end
	cn = 1.0 / mean(W[Y .== -1, j])
	cp = 1.0 / mean(W[Y .== 1, j])
	nrm = cn + cp
	return cp / nrm, cn / nrm
end

