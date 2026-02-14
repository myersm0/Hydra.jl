
function initialize(
		X::AbstractMatrix,
		Y::AbstractVector,
		K::Integer,
		::RandomHyperplaneInit,
	)
	idx_p = findall(Y .== 1)
	idx_n = findall(Y .== -1)
	X_p = X[idx_p, :]
	n_p = length(idx_p)
	row_norms = sqrt.(sum(X_p .^ 2, dims = 2))
	X_normalized = X_p ./ max.(row_norms, eps())
	projections = zeros(n_p, K)
	for j in 1:K
		ip = rand(1:length(idx_p))
		in_ = rand(1:length(idx_n))
		direction = X[idx_p[ip], :] .- X[idx_n[in_], :]
		direction ./= max(norm(direction), eps())
		projections[:, j] = X_normalized * direction
	end
	return proportional_assignment(projections)
end

function initialize(
		X::AbstractMatrix,
		Y::AbstractVector,
		K::Integer,
		::RandomAssignmentInit,
	)
	n_p = count(Y .== 1)
	return sample_dirichlet(ones(K), n_p)
end

function initialize(
		X::AbstractMatrix,
		Y::AbstractVector,
		K::Integer,
		::KMeansInit,
	)
	X_p = X[Y .== 1, :]
	n_p = size(X_p, 1)
	result = kmeans(X_p', K; maxiter = 200)
	S = zeros(n_p, K)
	for (i, cluster_id) in enumerate(assignments(result))
		S[i, cluster_id] = 1.0
	end
	return S
end

function initialize(
		X::AbstractMatrix,
		Y::AbstractVector,
		K::Int,
		::DPPHyperplaneInit,
	)
	idx_p = findall(Y .== 1)
	idx_n = findall(Y .== -1)
	X_p = X[idx_p, :]
	n_p = length(idx_p)
	n = size(X, 1)
	d = size(X, 2)
	directions = zeros(n, d)
	for j in 1:n
		ip = rand(1:length(idx_p))
		in_ = rand(1:length(idx_n))
		directions[j, :] = X[idx_p[ip], :] .- X[idx_n[in_], :]
	end
	KW = directions * directions'
	kw_diag = diag(KW)
	normalizer = sqrt.(kw_diag * kw_diag')
	normalizer[normalizer .== 0] .= 1.0
	KW ./= normalizer
	selected = sample_kdpp(KW, K)
	row_norms = sqrt.(sum(X_p .^ 2, dims = 2))
	X_normalized = X_p ./ max.(row_norms, eps())
	projections = zeros(n_p, K)
	for j in 1:K
		projections[:, j] = X_normalized * directions[selected[j], :]
	end
	return proportional_assignment(projections)
end

function proportional_assignment(projections::AbstractMatrix)
	shifted = projections .- 1.0
	lower = min.(shifted, 0.0)
	inv_lower = 1.0 ./ lower
	for idx in eachindex(inv_lower)
		if isinf(inv_lower[idx])
			inv_lower[idx] = shifted[idx]
		end
	end
	for i in axes(inv_lower, 1)
		row = @view inv_lower[i, :]
		positive_mask = row .> 0
		if any(positive_mask)
			row[.!positive_mask] .= 0.0
		else
			min_val = minimum(row)
			row ./= min_val
			row[row .< 1.0] .= 0.0
		end
	end
	row_sums = sum(inv_lower, dims = 2)
	row_sums[row_sums .== 0] .= 1.0
	return inv_lower ./ row_sums
end

## DPP sampling

function sample_kdpp(kernel_matrix::AbstractMatrix, k::Integer)
	eigenvalues, eigenvectors = eigen(Symmetric(kernel_matrix))
	eigenvalues = real.(eigenvalues)
	eigenvectors = real.(eigenvectors)
	selected_eigen = sample_k_eigenvalues(eigenvalues, k)
	V = eigenvectors[:, selected_eigen]
	return sample_from_basis(V, k)
end

function sample_k_eigenvalues(eigenvalues::AbstractVector, k::Integer)
	E = elementary_symmetric_polynomials(eigenvalues, k)
	n = length(eigenvalues)
	selected = Int[]
	remaining = k
	i = n
	while remaining > 0
		if i == remaining
			marginal = 1.0
		else
			marginal = eigenvalues[i] * E[remaining, i] / E[remaining + 1, i + 1]
		end
		if rand() < marginal
			push!(selected, i)
			remaining -= 1
		end
		i -= 1
	end
	return selected
end

function elementary_symmetric_polynomials(eigenvalues::AbstractVector, k::Integer)
	n = length(eigenvalues)
	E = zeros(k + 1, n + 1)
	E[1, :] .= 1.0
	for l in 2:(k + 1)
		for j in 2:(n + 1)
			E[l, j] = E[l, j - 1] + eigenvalues[j - 1] * E[l - 1, j - 1]
		end
	end
	return E
end

function sample_from_basis(V::AbstractMatrix, k::Integer)
	selected = zeros(Int, k)
	V = copy(V)
	for i in k:-1:1
		P = sum(V .^ 2, dims = 2) |> vec
		P ./= sum(P)
		cumulative = cumsum(P)
		selected[i] = searchsortedfirst(cumulative, rand())
		pivot_col = findfirst(j -> V[selected[i], j] != 0, 1:size(V, 2))
		pivot = V[:, pivot_col]
		V = V[:, setdiff(1:size(V, 2), pivot_col)]
		V .-= pivot .* (V[selected[i], :]' ./ pivot[selected[i]])
		for a in 1:size(V, 2)
			for b in 1:(a - 1)
				V[:, a] .-= dot(V[:, a], V[:, b]) .* V[:, b]
			end
			col_norm = norm(V[:, a])
			if col_norm > eps()
				V[:, a] ./= col_norm
			end
		end
	end
	return sort!(selected)
end

function sample_dirichlet(alpha::AbstractVector, n::Integer)
	k = length(alpha)
	raw = Matrix{Float64}(undef, n, k)
	for j in 1:k
		raw[:, j] = rand(Gamma(alpha[j], 1.0), n)
	end
	return raw ./ sum(raw, dims = 2)
end

