using LinearAlgebra, Optim#, ForwardDiff

export SFW!

"""
Solve the BLASSO problem using Sliding Frank-Wolfe (SFW) algorithm.
Handles both 1D and 2D cases.
"""
function SFW!(blasso::BLASSO, options::Dict{Symbol,<:Any}=Dict{Symbol,Any}())::BLASSO
    y, λ = blasso.y, blasso.λ
    ops = blasso.operators
    ϕ, Φ, adjΦ = ops.ϕ, ops.Φ, ops.adjΦ

    maxits = get(options, :maxits, 100)
    gridsize = get(options, :gridsize, 21)
    tol = get(options, :tol, 1.00001)

    xgrid = coarse_grid(blasso.domain, gridsize)

    xvals = blasso.dim == 1 ? Float64[] : [Float64[], Float64[]]

    M = sum(abs2.(y)) / (2 * λ)
    avals = Float64[]

    for k in 1:maxits
        xnew, η_max = compute_next_spike(xgrid, xvals, avals, adjΦ, Φ, y, λ)

        η_max < tol && break

        lasso_update!(xnew, xvals, avals, M, k, adjΦ, Φ, y, λ)
        optimise_amplitudes!(xvals, avals, ϕ, y, λ)
        xvals, avals = local_descent(xvals, avals, Φ, y, λ, k)
    end

    blasso.μ = blasso.dim == 1 ? DiscreteMeasure(xvals, avals) : DiscreteMeasure(collect(xvals), avals)

    return blasso
end

function coarse_grid(domain::Vector{T}, gridsize::Int) where {T<:Real}
    range(extrema(domain)..., length=gridsize)
end

function coarse_grid(domain::Vector{Vector{T}}, gridsize::Int) where {T<:Real}
    range1 = coarse_grid(domain[1], gridsize)
    range2 = coarse_grid(domain[2], gridsize)
    grid1 = [x1 for x1 in range1, x2 in range2]
    grid2 = [x2 for x1 in range1, x2 in range2]
    return [grid1, grid2]
end

function η(
    grid::AbstractVector{<:Number},
    x::AbstractVector{T},
    a::AbstractVector{T},
    adjΦ::Function,
    Φ::Function,
    y::AbstractVector{<:Number},
    λ::Real
) where {T<:Real}
    adjΦ(y - Φ(x, a), grid=grid) / λ
end

function η(
    grid::AbstractVector{<:AbstractArray{<:Number}},
    x::AbstractVector{<:AbstractArray{T}},
    a::AbstractVector{T},
    adjΦ::Function,
    Φ::Function,
    y::AbstractVector{<:Number},
    λ::Real
) where {T<:Real}
    adjΦ(y - Φ(x..., a), grid_x1=grid[1], grid_x2=grid[2]) / λ
end

function compute_next_spike(
    xgrid::AbstractVector{T},
    xvals::AbstractVector{T},
    avals::AbstractVector{T},
    adjΦ::Function,
    Φ::Function,
    y::AbstractVector{<:Number},
    λ::Real
) where {T<:Real}
    η_k = η(xgrid, xvals, avals, adjΦ, Φ, y, λ)
    idx = argmax(abs.(η_k))
    return xgrid[idx], abs(η_k[idx])
end

function compute_next_spike(
    xgrid::AbstractVector{<:AbstractArray{T}},
    xvals::AbstractVector{<:AbstractArray{T}},
    avals::AbstractVector{T},
    adjΦ::Function,
    Φ::Function,
    y::AbstractArray{<:Number},
    λ::Real
) where {T<:Real}
    η_k = η(xgrid, xvals, avals, adjΦ, Φ, y, λ)
    idx = argmax(abs.(η_k))
    return getindex.(xgrid, idx), abs(η_k[idx])
end

function lasso_update!(
    xnew::T,
    xvals::AbstractVector{T},
    avals::AbstractVector{T},
    M::Real,
    k::Int,
    adjΦ::Function,
    Φ::Function,
    y::AbstractVector{<:Number},
    λ::Real
) where {T<:Real}
    mfun(grid) = -0.5 * norm(η(grid, xvals, avals, adjΦ, Φ, y, λ))^2

    result = optimize(mfun, [xnew], LBFGS(); autodiff=:forward)
    xnew = Optim.minimizer(result)[1]
    anew = M * sign(η([xnew], xvals, avals, adjΦ, Φ, y, λ)[1])

    push!(xvals, xnew)
    avals .*= (1 - 2 / (k + 2))
    push!(avals, 2 / (k + 2) * anew)
end

function lasso_update!(
    xnew::AbstractVector{T},
    xvals::AbstractVector{<:AbstractVector{T}},
    avals::AbstractVector{T},
    M::Real,
    k::Int,
    adjΦ::Function,
    Φ::Function,
    y::AbstractVector{<:Number},
    λ::Real
) where {T<:Real}
    mfun(grid) = -0.5 * norm(η([[x] for x in grid], xvals, avals, adjΦ, Φ, y, λ))^2

    result = optimize(mfun, xnew, LBFGS(); autodiff=:forward)
    xnew = Optim.minimizer(result)
    anew = M * sign(η([[x] for x in xnew], xvals, avals, adjΦ, Φ, y, λ)[1])

    push!.(xvals, xnew)
    avals .*= (1 - 2 / (k + 2))
    push!(avals, 2 / (k + 2) * anew)
end

function optimise_amplitudes!(
    xvals::AbstractVector{T},
    avals::AbstractVector{T},
    ϕ::Function,
    y::AbstractVector{<:Number},
    λ::Real
) where {T<:Real}
    X = real(ϕ(xvals)' * ϕ(xvals))
    Xty = real(ϕ(xvals)' * y)
    τ = 1 / norm(X)

    for _ in 1:100
        avals .-= τ * (X * avals .- Xty)
        avals .= sign.(avals) .* max.(abs.(avals) .- τ * λ, 0)
    end
end

function optimise_amplitudes!(
    xvals::AbstractVector{<:AbstractVector{T}},
    avals::AbstractVector{T},
    ϕ::Function,
    y::AbstractVector{<:Number},
    λ::Real
) where {T<:Real}
    X = real(ϕ(xvals...)' * ϕ(xvals...))
    Xty = real(ϕ(xvals...)' * y)
    τ = 1 / norm(X)

    for _ in 1:100
        avals .-= τ * (X * avals .- Xty)
        avals .= sign.(avals) .* max.(abs.(avals) .- τ * λ, 0)
    end
end

function local_descent(
    xvals::AbstractVector{T},
    avals::AbstractVector{T},
    Φ::Function,
    y::AbstractVector{<:Number},
    λ::Real,
    k::Int
) where {T<:Real}
    function func(ax)
        a, x = ax[1:k], ax[(k+1):end]
        λ * norm(a, 1) + 0.5norm(y - Φ(x, a))^2
    end

    ax0 = vcat(avals, xvals)
    result = optimize(func, ax0, BFGS(), autodiff=:forward)
    ax_opt = Optim.minimizer(result)
    avals = ax_opt[1:k]
    xvals = mod.(ax_opt[(k+1):end], 1)
    return xvals, avals
end

function local_descent(
    xvals::AbstractVector{<:AbstractVector{T}},
    avals::AbstractVector{T},
    Φ::Function,
    y::AbstractVector{<:Number},
    λ::Real,
    k::Int
) where {T<:Real}
    x1vals, x2vals = xvals
    function func(xa)
        x1 = xa[1:k]
        x2 = xa[k+1:2k]
        a = xa[2k+1:end]
        return λ * norm(a, 1) + 0.5 * norm(y - Φ(x1, x2, a))^2
    end

    ax0 = vcat(x1vals, x2vals, avals)
    result = optimize(func, ax0, BFGS(), autodiff=:forward)
    xa_opt = Optim.minimizer(result)
    x1vals = xa_opt[1:k]
    x2vals = xa_opt[k+1:2k]
    avals = xa_opt[2k+1:end]
    return [x1vals, x2vals], avals
end
