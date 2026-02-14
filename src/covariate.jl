
function residualize_covariates(
		X::AbstractMatrix,
		Y::AbstractVector,
		covariates::AbstractMatrix,
	)
	X_cn = X[Y .== -1, :]
	C_cn = covariates[Y .== -1, :]
	B = hcat(C_cn, ones(size(C_cn, 1)))
	Z = X_cn' * B / (B' * B)
	β = Z[:, 1:end-1]
	return (X' .- β * covariates')'
end

