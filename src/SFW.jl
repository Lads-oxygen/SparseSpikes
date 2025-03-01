using LinearAlgebra, Optim, ProgressMeter, BenchmarkTools, LineSearches

export SFW!

function clamp_strict!(x, lo, hi)
    if x ≤ lo
        return lo + 1e-10
    elseif x ≥ hi
        return hi - 1e-10
    else
        return x
    end
end

function clamp_strict!(x::AbstractArray, lo::AbstractArray, hi::AbstractArray)
    @inbounds for i in eachindex(x, lo, hi)
        x[i] = clamp_strict!(x[i], lo[i], hi[i])
    end
    return x
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
    λ::Real,
    # domain::Vector{<:Real}
) where {T<:Real}
    mfun(grid) = -0.5 * sum(abs2, η(grid, xvals, avals, adjΦ, Φ, y, λ))

    xnew = clamp_strict!(xnew, domain[1], domain[2])

    result = optimize(mfun, [xnew], LBFGS(); autodiff=:forward)
    xnew = Optim.minimizer(result)[1]
    xnew = clamp_strict!(xnew, domain[1], domain[2])
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
    λ::Real,
    # domain::AbstractVector{<:AbstractVector{<:Real}}
) where {T<:Real}
    mfun(grid) = -0.5 * sum(abs2, η([[x] for x in grid], xvals, avals, adjΦ, Φ, y, λ))

    xnew[1] = clamp_strict!(xnew[1], domain[1][1], domain[1][2])
    xnew[2] = clamp_strict!(xnew[2], domain[2][1], domain[2][2])

    result = optimize(mfun, xnew, LBFGS(); autodiff=:forward)
    xnew = Optim.minimizer(result)

    xnew[1] = clamp_strict!(xnew[1], domain[1][1], domain[1][2])
    xnew[2] = clamp_strict!(xnew[2], domain[2][1], domain[2][2])
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

function local_descent_BFGS!(
    xvals::AbstractVector{T},
    avals::AbstractVector{T},
    Φ::Function,
    y::AbstractVector{<:Number},
    λ::Real,
    k::Int,
    domain::AbstractVector{<:Real},
    amplitudes_lo::Real
) where {T<:Real}
    function func(xa)
        x, a = xa[1:k], xa[(k+1):end]
        λ * norm(a, 1) + 0.5 * sum(abs2, y - Φ(x, a))
    end

    xa0 = vcat(xvals, avals)
    lower_bounds = vcat(fill(domain[1], k), fill(amplitudes_lo, k))
    upper_bounds = vcat(fill(domain[2], k), fill(Inf, k))
    xa0 = clamp_strict!(xa0, lower_bounds, upper_bounds)

    result = optimize(func, lower_bounds, upper_bounds, xa0, Fminbox(BFGS()); autodiff=:forward)
    xa_opt = Optim.minimizer(result)

    xvals .= xa_opt[1:k]
    avals .= xa_opt[(k+1):end]
end

function local_descent_BFGS!(
    xvals::AbstractVector{<:AbstractVector{T}},
    avals::AbstractVector{T},
    Φ::Function,
    y::AbstractVector{<:Number},
    λ::Real,
    k::Int,
    domain::AbstractVector{<:AbstractVector{<:Real}},
    amplitudes_lo::Real
) where {T<:Real}
    x1vals, x2vals = xvals

    function func(xa)
        x1 = @view xa[1:k]
        x2 = @view xa[k+1:2k]
        a = @view xa[2k+1:end]
        return λ * norm(a, 1) + 0.5 * sum(abs2, y - Φ(x1, x2, a))
    end

    ax0 = vcat(x1vals, x2vals, avals)
    lower_bounds = vcat(fill(domain[1][1], k), fill(domain[2][1], k), fill(amplitudes_lo, k))
    upper_bounds = vcat(fill(domain[1][2], k), fill(domain[2][2], k), fill(Inf, k))
    ax0 = clamp_strict!(ax0, lower_bounds, upper_bounds)

    result = optimize(func, lower_bounds, upper_bounds, ax0, Fminbox(BFGS(linesearch=LineSearches.BackTracking())); autodiff=:forward)
    xa_opt = Optim.minimizer(result)

    x1vals .= xa_opt[1:k]
    x2vals .= xa_opt[k+1:2k]
    avals .= xa_opt[2k+1:end]
end

function local_descent_LBFGS!(
    xvals::AbstractVector{<:AbstractVector{T}},
    avals::AbstractVector{T},
    Φ::Function,
    y::AbstractVector{<:Number},
    λ::Real,
    k::Int,
    domain::AbstractVector{<:AbstractVector{<:Real}},
    amplitudes_lo::Real
) where {T<:Real}
    x1vals, x2vals = xvals

    function func(xa)
        x1 = @view xa[1:k]
        x2 = @view xa[k+1:2k]
        a = @view xa[2k+1:end]
        return λ * norm(a, 1) + 0.5 * sum(abs2, y - Φ(x1, x2, a))
    end

    ax0 = vcat(x1vals, x2vals, avals)
    lower_bounds = vcat(fill(domain[1][1], k), fill(domain[2][1], k), fill(amplitudes_lo, k))
    upper_bounds = vcat(fill(domain[1][2], k), fill(domain[2][2], k), fill(Inf, k))
    ax0 = clamp_strict!(ax0, lower_bounds, upper_bounds)

    result = optimize(func, lower_bounds, upper_bounds, ax0, Fminbox(LBFGS(linesearch=LineSearches.BackTracking())); autodiff=:forward)
    xa_opt = Optim.minimizer(result)

    x1vals .= xa_opt[1:k]
    x2vals .= xa_opt[k+1:2k]
    avals .= xa_opt[2k+1:end]
end

function local_descent_smooth!(
    xvals::AbstractVector{<:AbstractVector{T}},
    avals::AbstractVector{T},
    Φ::Function,
    y::AbstractVector{<:Number},
    λ::Real,
    k::Int,
    domain::AbstractVector{<:AbstractVector{<:Real}},
    amplitudes_lo::Real
) where {T<:Real}
    x1vals, x2vals = xvals

    function func(xuv)
        x1 = @view xuv[1:k]
        x2 = @view xuv[k+1:2k]
        u = @view xuv[2k+1:3k]
        v = @view xuv[3k+1:end]
        0.5λ * (sum(abs2, u) + sum(abs2, v)) + 0.5 * sum(abs2, y - Φ(x1, x2, u .* v))
    end

    u = avals
    v = ones(k)

    xuv0 = vcat(x1vals, x2vals, u, v)
    lower_bounds = vcat(fill(domain[1][1], k), fill(domain[2][1], k), fill(amplitudes_lo, k), fill(amplitudes_lo, k))
    upper_bounds = vcat(fill(domain[1][2], k), fill(domain[2][2], k), fill(Inf, k), fill(Inf, k))
    xuv0 = clamp_strict!(xuv0, lower_bounds, upper_bounds)

    result = optimize(
        func,
        lower_bounds,
        upper_bounds,
        xuv0,
        Fminbox(BFGS(linesearch=LineSearches.BackTracking())),
        Optim.Options(time_limit=15, store_trace=true);
        autodiff=:forward
    )
    xuv_opt = Optim.minimizer(result)

    x1vals .= xuv_opt[1:k]
    x2vals .= xuv_opt[k+1:2k]

    avals .= xuv_opt[2k+1:3k] .* xuv_opt[3k+1:end]
end

function local_descent_smooth_inner!(
    xvals::AbstractVector{<:AbstractVector{T}},
    avals::AbstractVector{T},
    Φ::Function,
    y::AbstractVector{<:Number},
    λ::Real,
    k::Int,
    domain::AbstractVector{<:AbstractVector{<:Real}},
    amplitudes_lo::Real
) where {T<:Real}
    x1vals, x2vals = xvals

    function func(xu)
        x1 = @view xu[1:k]
        x2 = @view xu[k+1:2k]
        u = @view xu[2k+1:end]
        M = Φ(x1, x2, diagm(u))
        v = (M' * M + λ * I(k)) \ M' * y
        0.5λ * (sum(abs2, u) + sum(abs2, v)) + 0.5 * sum(abs2, y - M * v)
    end

    u0 = avals
    xu0 = vcat(x1vals, x2vals, u0)
    lower_bounds = vcat(fill(domain[1][1], k), fill(domain[2][1], k), fill(amplitudes_lo, k))
    upper_bounds = vcat(fill(domain[1][2], k), fill(domain[2][2], k), fill(Inf, k))
    xu0 = clamp_strict!(xu0, lower_bounds, upper_bounds)

    result = optimize(func, lower_bounds, upper_bounds, xu0, Fminbox(BFGS(linesearch=LineSearches.BackTracking())); autodiff=:forward)
    xu_opt = Optim.minimizer(result)

    x1vals .= xu_opt[1:k]
    x2vals .= xu_opt[k+1:2k]

    u_opt = xu_opt[2k+1:end]
    M = Φ(x1vals, x2vals, diagm(u_opt))
    v = (M' * M + λ * I(k)) \ M' * y

    avals .= u_opt .* v
end

const DESCENT_METHODS = Dict(
    :BFGS => local_descent_BFGS!,
    :LBFGS => local_descent_LBFGS!,
    :smooth => local_descent_smooth!,
    :smooth_inner => local_descent_smooth_inner!
)

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
    descent = get(options, :descent, :BFGS)
    positivity = get(options, :positivity, false)
    amplitudes_lo = positivity ? 0.0 : -Inf

    xgrid = coarse_grid(blasso.domain, gridsize)

    xvals = blasso.dim == 1 ? Float64[] : [Float64[], Float64[]]

    M = sum(abs2.(y)) / (2 * λ)
    avals = Float64[]

    # println("| Iteration | Lasso time | Lasso gc  | Lasso compile | LD time  | LD gc   | LD compile  |")
    # println("|-----------|------------|-----------|---------------|----------|---------|-------------|")

    # @showprogress desc = "SFW iterations: " for k in 1:maxits
    for k in 1:maxits
        xnew, η_max = compute_next_spike(xgrid, xvals, avals, adjΦ, Φ, y, λ)

        η_max < tol && break

        lt = @timed lasso_update!(xnew, xvals, avals, M, k, adjΦ, Φ, y, λ)
        optimise_amplitudes!(xvals, avals, ϕ, y, λ)

        dt = @timed DESCENT_METHODS[descent](xvals, avals, Φ, y, λ, k, blasso.domain, amplitudes_lo)
        # println("| $(rpad(k, 9)) | $(rpad(round(lt.time, digits=4), 10)) | $(rpad(round(lt.gctime, digits=4), 9)) | $(rpad(round(lt.compile_time, digits=4), 13)) | $(rpad(round(dt.time, digits=4), 8)) | $(rpad(round(dt.gctime, digits=4), 7)) | $(rpad(round(dt.compile_time, digits=4), 11)) |")
    end

    blasso.μ = blasso.dim == 1 ? DiscreteMeasure(xvals, avals) : DiscreteMeasure(collect(xvals), avals)

    return blasso
end