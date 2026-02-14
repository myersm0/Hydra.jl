function consensus_clustering(labels::AbstractMatrix, K::Integer)
	n = size(labels, 1)
	cooc = zeros(n, n)
	for i in 1:(n - 1)
		for j in (i + 1):n
			cooc[i, j] = count(labels[i, :] .== labels[j, :])
		end
	end
	cooc .+= cooc'
	return spectral_cluster(cooc, K)
end

function spectral_cluster(similarity::AbstractMatrix, K::Integer)
	n = size(similarity, 1)
	degree = vec(sum(similarity, dims = 2))
	d_inv_sqrt = zeros(n)
	for i in eachindex(degree)
		d_inv_sqrt[i] = degree[i] > 0 ? degree[i]^(-0.5) : 0.0
	end
	Ln = I - Diagonal(d_inv_sqrt) * similarity * Diagonal(d_inv_sqrt)
	Ln = Symmetric(Ln)
	decomposition = eigen(Ln)
	V = real.(decomposition.vectors[:, 1:K])
	if any(isnan, V) || any(isinf, V)
		L = Diagonal(degree) - similarity
		decomposition = eigen(Symmetric(L))
		V = real.(decomposition.vectors[:, 1:K])
	end
	result = kmeans(V', K; maxiter = 200)
	return assignments(result)
end

function ari(c1::AbstractVector, c2::AbstractVector)
	C = contingency_table(c1, c2)
	n = sum(C)
	sum_rows_sq = sum(sum(C, dims = 2) .^ 2)
	sum_cols_sq = sum(sum(C, dims = 1) .^ 2)
	sum_cells_sq = sum(C .^ 2)
	num_pairs = n * (n - 1) ÷ 2
	cross_term = 0.5 * (sum_rows_sq + sum_cols_sq)
	agreements = num_pairs + sum_cells_sq - cross_term
	expected = (
		n * (n^2 + 1) -
		(n + 1) * sum_rows_sq -
		(n + 1) * sum_cols_sq +
		2 * (sum_rows_sq * sum_cols_sq) / n
	) / (2 * (n - 1))
	return num_pairs == expected ? 0.0 : (agreements - expected) / (num_pairs - expected)
end

function contingency_table(c1::AbstractVector, c2::AbstractVector)
	table = zeros(Int, maximum(c1), maximum(c2))
	for i in eachindex(c1)
		table[c1[i], c2[i]] += 1
	end
	return table
end

function cross_validation_stability(fold_assignments::AbstractMatrix)
	num_folds = size(fold_assignments, 2)
	ari_values = Float64[]
	for i in 1:(num_folds - 1)
		for j in (i + 1):num_folds
			valid = (fold_assignments[:, i] .!= 0) .& (fold_assignments[:, j] .!= 0)
			if count(valid) > 1
				push!(ari_values, ari(
					fold_assignments[valid, i],
					fold_assignments[valid, j],
				))
			end
		end
	end
	return isempty(ari_values) ? 0.0 : mean(ari_values)
end

