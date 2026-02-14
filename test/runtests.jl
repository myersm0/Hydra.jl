using Hydra
using Test
using Clustering
using LinearAlgebra
using Random
using Statistics: mean, cor

using Hydra: 
	solve_hydra, solve_weighted_svm, predict_svm, initialize,
	consensus_clustering, residualize_covariates, spectral_cluster,
	contingency_table, SVMModel, SolverResult, HydraConfig,
	L1Regularization, L2Regularization,
	RandomHyperplaneInit, RandomAssignmentInit, KMeansInit, DPPHyperplaneInit

function make_synthetic(;
		num_controls = 100,
		num_per_subtype = 50,
		num_features = 10,
		num_subtypes = 3,
		separation = 2.0,
		noise = 0.5,
	)
	controls = randn(num_controls, num_features) * noise
	patients = zeros(num_subtypes * num_per_subtype, num_features)
	true_labels = zeros(Int, num_subtypes * num_per_subtype)
	for s in 1:num_subtypes
		direction = zeros(num_features)
		direction[s] = separation
		offset = (s - 1) * num_per_subtype
		for i in 1:num_per_subtype
			patients[offset + i, :] = randn(num_features) * noise .+ direction
			true_labels[offset + i] = s
		end
	end
	X = vcat(controls, patients)
	Y = vcat(fill(-1, num_controls), fill(1, size(patients, 1)))
	return X, Y, true_labels
end

@testset "Hydra.jl" begin
	@testset "ARI" begin
		@test ari([1, 1, 2, 2], [1, 1, 2, 2]) ≈ 1.0
		@test ari([1, 1, 2, 2], [2, 2, 1, 1]) ≈ 1.0
		@test ari([1, 1, 1, 1], [1, 2, 1, 2]) ≈ 0.0 atol = 1e-10
		c1 = repeat(1:3, inner = 10)
		c2 = copy(c1)
		@test ari(c1, c2) ≈ 1.0
	end

	@testset "Contingency table" begin
		table = contingency_table([1, 1, 2, 2, 3], [1, 2, 2, 3, 3])
		@test size(table) == (3, 3)
		@test sum(table) == 5
		@test table[1, 1] == 1
		@test table[1, 2] == 1
		@test table[2, 2] == 1
		@test table[2, 3] == 1
		@test table[3, 3] == 1
	end

	@testset "Weighted SVM — L2" begin
		Random.seed!(1)
		n = 50
		X = vcat(randn(n, 2) .+ [2.0 0.0], randn(n, 2) .- [2.0 0.0])
		Y = vcat(fill(1, n), fill(-1, n))
		w = ones(2 * n)
		model = solve_weighted_svm(X, Y, w, 1.0, 1.0, 1.0, L2Regularization())
		predictions = predict_svm(X, model)
		accuracy = mean((predictions .> 0) .== (Y .== 1))
		@test accuracy > 0.9
	end

	@testset "Weighted SVM — L1" begin
		Random.seed!(2)
		n = 50
		X = vcat(randn(n, 2) .+ [2.0 0.0], randn(n, 2) .- [2.0 0.0])
		Y = vcat(fill(1, n), fill(-1, n))
		w = ones(2 * n)
		model = solve_weighted_svm(X, Y, w, 1.0, 1.0, 1.0, L1Regularization())
		predictions = predict_svm(X, model)
		accuracy = mean((predictions .> 0) .== (Y .== 1))
		@test accuracy > 0.9
	end

	@testset "Weighted SVM — NaN weights return zero model" begin
		X = randn(10, 3)
		Y = vcat(fill(1, 5), fill(-1, 5))
		model = solve_weighted_svm(X, Y, ones(10), 1.0, NaN, 1.0, L2Regularization())
		@test all(model.w .== 0)
		@test model.b == 0.0
	end

	@testset "Initialization — all strategies run" begin
		Random.seed!(3)
		X, Y, _ = make_synthetic(num_controls = 30, num_per_subtype = 15)
		k = 3
		inits_to_test = [
			RandomHyperplaneInit(), 
			RandomAssignmentInit(), 
			KMeansInit(), 
			DPPHyperplaneInit()
		]
		for init in inits_to_test
			S = initialize(X, Y, k, init)
			n_p = count(Y .== 1)
			@test size(S) == (n_p, k)
			@test all(S .>= 0)
			row_sums = sum(S, dims = 2)
			@test all(isapprox.(row_sums, 1.0, atol = 1e-10))
		end
	end

	@testset "Consensus clustering" begin
		Random.seed!(4)
		n = 30
		num_runs = 10
		IDX = zeros(Int, n, num_runs)
		for run in 1:num_runs
			IDX[1:10, run] .= 1
			IDX[11:20, run] .= 2
			IDX[21:30, run] .= 3
		end
		result = consensus_clustering(IDX, 3)
		@test length(result) == n
		@test length(unique(result)) == 3
		@test all(result[1:10] .== result[1])
		@test all(result[11:20] .== result[11])
		@test all(result[21:30] .== result[21])
	end

	@testset "Covariate residualization" begin
		Random.seed!(5)
		n = 100
		age = randn(n) * 10 .+ 50
		X_clean = randn(n, 5)
		X = X_clean .+ age * ones(1, 5) * 0.5
		Y = vcat(fill(-1, 50), fill(1, 50))
		covariates = reshape(age, :, 1)
		X_residualized = residualize_covariates(X, Y, covariates)
		correlation_before = abs(cor(age, X[:, 1]))
		correlation_after = abs(cor(age, X_residualized[:, 1]))
		@test correlation_after < correlation_before
		@test correlation_after < 0.3
	end

	@testset "Core solver — subtype recovery" begin
		Random.seed!(10)
		X, Y, true_labels = make_synthetic(separation = 2.5)
		config = HydraConfig(
			num_consensus = 10,
			num_iterations = 50,
			cluster_range = 3:1:3,
			num_folds = 3,
		)
		result = solve_hydra(X, Y, 3, config)
		predicted = result.assignments[Y .== 1]
		recovery_ari = ari(true_labels, predicted)
		@test recovery_ari > 0.7
	end

	@testset "Full pipeline — correct k selection" begin
		Random.seed!(42)
		X, Y, true_labels = make_synthetic(separation = 2.5)
		config = HydraConfig(
			cluster_range = 1:1:5,
			num_folds = 3,
			num_consensus = 10,
			num_iterations = 30,
		)
		result = hydra(X, Y; config = config)
		@test size(result.assignments) == (size(X, 1), 5)
		@test length(result.ari) == 5
		@test result.cluster_range == 1:1:5
		best_k = collect(result.cluster_range)[argmax(ari(result))]
		@test best_k == 3
	end

	@testset "Accessors" begin
		Random.seed!(42)
		X, Y, _ = make_synthetic()
		config = HydraConfig(
			cluster_range = 2:1:4,
			num_folds = 3,
			num_consensus = 5,
			num_iterations = 20,
		)
		result = hydra(X, Y; config = config)
		@test length(ari(result)) == 3
		@test ari(result, 2) == result.ari[1]
		@test ari(result, 3) == result.ari[2]
		@test ari(result, 4) == result.ari[3]
		@test_throws ErrorException ari(result, 7)
		a = assignments(result)
		@test size(a) == size(result.assignments)
		a3 = assignments(result, 3)
		@test length(a3) == size(X, 1)
		@test all(a3[Y .== -1] .== -1)
		@test all(a3[Y .== 1] .> 0)
		@test_throws ErrorException assignments(result, 7)
	end
end
