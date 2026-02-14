
function solve_weighted_svm(
		X::AbstractMatrix,
		Y::AbstractVector,
		sample_weights::AbstractVector,
		c::Float64,
		cp::Float64,
		cn::Float64,
		::L2Regularization,
	)
	if any(isnan, (cp, cn))
		return SVMModel(zeros(size(X, 2)), 0.0)
	end
	n, d = size(X)
	cw = Vector{Float64}(undef, n)
	for i in eachindex(Y)
		cw[i] = Y[i] == 1 ? cp : cn
	end
	P = spdiagm(0 => vcat(ones(d), 0.0, zeros(n)))
	q = vcat(zeros(d + 1), c .* sample_weights .* cw)
	YX = Diagonal(Float64.(Y)) * X
	A_svm = hcat(-YX, -Float64.(Y), -sparse(I, n, n))
	A_slack = hcat(spzeros(n, d + 1), -sparse(I, n, n))
	A = sparse(vcat(A_svm, A_slack))
	b = vcat(-ones(n), zeros(n))
	cones = [Clarabel.NonnegativeConeT(2 * n)]
	settings = Clarabel.Settings(verbose = false)
	solver = Clarabel.Solver()
	Clarabel.setup!(solver, triu(P), q, A, b, cones, settings)
	result = Clarabel.solve!(solver)
	return SVMModel(result.x[1:d], result.x[d + 1])
end

function solve_weighted_svm(
		X::AbstractMatrix,
		Y::AbstractVector,
		sample_weights::AbstractVector,
		c::Float64,
		cp::Float64,
		cn::Float64,
		::L1Regularization,
	)
	any(isnan, (cp, cn)) && return SVMModel(zeros(size(X, 2)), 0.0)

	n, d = size(X)
	cw = Vector{Float64}(undef, n)
	for i in eachindex(Y)
		cw[i] = Y[i] == 1 ? cp : cn
	end

	penalty = c .* sample_weights .* cw

	P = spdiagm(0 => vcat(zeros(2 * d), penalty))
	q = vcat(ones(2 * d), zeros(n))

	YX = Diagonal(Float64.(Y)) * X
	A_svm = hcat(-YX, YX, -sparse(I, n, n))

	num_bounded = 2 * d + n
	A_bounds = -sparse(I, num_bounded, num_bounded)

	A = sparse(vcat(A_svm, A_bounds))
	b = vcat(-ones(n), zeros(num_bounded))
	cones = [Clarabel.NonnegativeConeT(n + num_bounded)]

	settings = Clarabel.Settings(verbose = false)
	solver = Clarabel.Solver()
	Clarabel.setup!(solver, triu(P), q, A, b, cones, settings)
	result = Clarabel.solve!(solver)

	w_pos = result.x[1:d]
	w_neg = result.x[d+1:2*d]

	return SVMModel(w_pos .- w_neg, 0.0)
end

function predict_svm(X::AbstractMatrix, model::SVMModel)
	return X * model.w .+ model.b
end

