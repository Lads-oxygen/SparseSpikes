using LinearAlgebra, Optim, ProgressMeter, BenchmarkTools, LineSearches

using Lasso

export SFW!, BSFW!

function clamp_strict(x, lo, hi)
    if x ≤ lo
        return lo + 1e-20
    elseif x ≥ hi
        return hi - 1e-20
    else
        return x
    end
end

function clamp_strict!(x::AbstractArray, lo::AbstractArray, hi::AbstractArray)
    @inbounds for i in eachindex(x, lo, hi)
        x[i] = clamp_strict(x[i], lo[i], hi[i])
    end
end

function prune!(xvals::AbstractVector{T}, avals::AbstractVector{T}, factor::Real=1e-5) where {T<:Real}
    isempty(avals) && return

    maxval = maximum(avals)
    idx = findall(x -> abs(x) < factor * maxval, avals)
    deleteat!(xvals, idx)
    deleteat!(avals, idx)
end

function prune!(xvals::AbstractVector{<:AbstractVector{T}}, avals::AbstractVector{T}, factor::Real=1e-5) where {T<:Real}
    isempty(avals) && return

    maxval = maximum(avals)
    idx = findall(x -> abs(x) < factor * maxval, avals)
    deleteat!.(xvals, Ref(idx))
    deleteat!(avals, idx)
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

# #TODO: make it local ascent as well
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
    x0 = getindex.(xgrid, idx)

    function max_η_k(x)
        xpos = [[x[1]], [x[2]]]
        return -abs(η(xpos, xvals, avals, adjΦ, Φ, y, λ)[1])
    end

    result = optimize(max_η_k, x0, LBFGS(); autodiff=:forward)
    xnew = Optim.minimizer(result)
    η_max = -Optim.minimum(result)

    return xnew, η_max
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
    domain::Vector{<:Real},
    amplitudes_lo::Real
) where {T<:Real}
    mfun(grid) = -0.5 * sum(abs2, η(grid, xvals, avals, adjΦ, Φ, y, λ))

    xnew = clamp_strict(xnew, domain[1], domain[2])

    result = optimize(mfun, [xnew], LBFGS(); autodiff=:forward)
    xnew = Optim.minimizer(result)[1]

    xnew = clamp_strict(xnew, domain[1], domain[2])

    anew = M * sign(η([xnew], xvals, avals, adjΦ, Φ, y, λ)[1])
    anew = clamp_strict(anew, amplitudes_lo, Inf)

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
    domain::AbstractVector{<:AbstractVector{<:Real}},
    amplitudes_lo::Real
) where {T<:Real}
    mfun(grid) = -0.5 * sum(abs2, η([[x] for x in grid], xvals, avals, adjΦ, Φ, y, λ))

    xnew[1] = clamp_strict(xnew[1], domain[1]...)
    xnew[2] = clamp_strict(xnew[2], domain[2]...)

    result = optimize(mfun, xnew, LBFGS(); autodiff=:forward)
    xnew = Optim.minimizer(result)

    xnew[1] = clamp_strict(xnew[1], domain[1]...)
    xnew[2] = clamp_strict(xnew[2], domain[2]...)

    anew = M * sign(η([[x] for x in xnew], xvals, avals, adjΦ, Φ, y, λ)[1])

    anew = clamp_strict(anew, amplitudes_lo, Inf)

    push!.(xvals, xnew)
    avals .*= (1 - 2 / (k + 2))

    push!(avals, 2 / (k + 2) * anew)
end

function optimise_amplitudes!(
    xvals::AbstractVector{T},
    avals::AbstractVector{T},
    ϕ::Function,
    y::AbstractVector{<:Number},
    λ::Real,
    amplitudes_lo::Real
) where {T<:Real}
    X = real(ϕ(xvals)' * ϕ(xvals))
    Xty = real(ϕ(xvals)' * y)
    τ = 1 / norm(X)

    for _ in 1:100
        avals .-= τ * (X * avals .- Xty)
        avals .= max.(sign.(avals) .* max.(abs.(avals) .- τ * λ, 0), amplitudes_lo)
    end
end

function optimise_amplitudes!(
    xvals::AbstractVector{<:AbstractVector{T}},
    avals::AbstractVector{T},
    ϕ::Function,
    y::AbstractVector{<:Number},
    λ::Real,
    amplitudes_lo::Real
) where {T<:Real}
    X = real(ϕ(xvals...)' * ϕ(xvals...))
    Xty = real(ϕ(xvals...)' * y)
    τ = 1 / norm(X)

    for _ in 1:100
        avals .-= τ * (X * avals .- Xty)
        avals .= max.(sign.(avals) .* max.(abs.(avals) .- τ * λ, 0), amplitudes_lo)
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
    clamp_strict!(xa0, lower_bounds, upper_bounds)

    result = optimize(
        func,
        xa0,
        BFGS(linesearch=LineSearches.BackTracking()),
        Optim.Options(g_tol=1e-4);
        autodiff=:forward
    )
    xa_opt = Optim.minimizer(result)

    clamp_strict!(xa_opt, lower_bounds, upper_bounds)

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
        return λ * norm(a, 1) + 0.5 * sum(abs2, y .- Φ(x1, x2, a))
    end

    ax0 = vcat(x1vals, x2vals, avals)
    lower_bounds = vcat(fill(domain[1][1], k), fill(domain[2][1], k), fill(amplitudes_lo, k))
    upper_bounds = vcat(fill(domain[1][2], k), fill(domain[2][2], k), fill(Inf, k))
    clamp_strict!(ax0, lower_bounds, upper_bounds)

    result = optimize(
        func,
        ax0,
        BFGS(linesearch=LineSearches.BackTracking()),
        Optim.Options(g_tol=1e-4);
        autodiff=:forward
    )
    xa_opt = Optim.minimizer(result)

    clamp_strict!(xa_opt, lower_bounds, upper_bounds)

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
        return λ * norm(a, 1) + 0.5 * sum(abs2, y .- Φ(x1, x2, a))
    end

    ax0 = vcat(x1vals, x2vals, avals)
    lower_bounds = vcat(fill(domain[1][1], k), fill(domain[2][1], k), fill(amplitudes_lo, k))
    upper_bounds = vcat(fill(domain[1][2], k), fill(domain[2][2], k), fill(Inf, k))
    clamp_strict!(ax0, lower_bounds, upper_bounds)

    result = optimize(
        func,
        ax0,
        LBFGS(linesearch=LineSearches.BackTracking()),
        Optim.Options(g_tol=1e-4);
        autodiff=:forward
    )
    xa_opt = Optim.minimizer(result)

    xa_opt = clamp_strict!(xa_opt, lower_bounds, upper_bounds)

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
        x1 = xuv[1:k]
        x2 = xuv[k+1:2k]
        u = xuv[2k+1:3k]
        v = xuv[3k+1:end]
        return 0.5λ * (sum(abs2, u) + sum(abs2, v)) + 0.5 * sum(abs2, y - Φ(x1, x2, u .* v))
    end

    u0 = sqrt.(abs.(avals))
    v0 = avals ./ u0

    xuv0 = vcat(x1vals, x2vals, u0, v0)
    lower_bounds = vcat(fill(domain[1][1], k), fill(domain[2][1], k), fill(0, k), fill(amplitudes_lo, k))
    upper_bounds = vcat(fill(domain[1][2], k), fill(domain[2][2], k), fill(Inf, k), fill(Inf, k))
    clamp_strict!(xuv0, lower_bounds, upper_bounds)

    result = optimize(
        func,
        xuv0,
        BFGS(linesearch=LineSearches.BackTracking()),
        Optim.Options(f_tol=1e-6, g_tol=1e-4);
        autodiff=:forward
    )
    xuv_opt = Optim.minimizer(result)

    clamp_strict!(xuv_opt, lower_bounds, upper_bounds)

    x1vals .= xuv_opt[1:k]
    x2vals .= xuv_opt[k+1:2k]

    avals .= xuv_opt[2k+1:3k] .* xuv_opt[3k+1:end]
end

function replace_nonfinite!(A)
    for i in eachindex(A)
        if !isfinite(A[i])
            A[i] = 0.0
        end
    end
    return A
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
        replace_nonfinite!(M)
        v = (M'M + λ * I(k)) \ M'y
        return 0.5λ * (sum(abs2, u) + sum(abs2, v)) + 0.5 * sum(abs2, y - M * v)
    end

    u0 = sqrt.(abs.(avals))
    xu0 = vcat(x1vals, x2vals, u0)
    lower_bounds = vcat(fill(domain[1][1], k), fill(domain[2][1], k), fill(amplitudes_lo, k))
    upper_bounds = vcat(fill(domain[1][2], k), fill(domain[2][2], k), fill(Inf, k))
    clamp_strict!(xu0, lower_bounds, upper_bounds)

    result = optimize(
        func,
        xu0,
        BFGS(linesearch=LineSearches.BackTracking()),
        Optim.Options(g_tol=1e-4);
        autodiff=:forward
    )
    xu_opt = Optim.minimizer(result)

    clamp_strict!(xu_opt, lower_bounds, upper_bounds)

    x1vals .= xu_opt[1:k]
    x2vals .= xu_opt[k+1:2k]

    u_opt = xu_opt[2k+1:end]
    M = Φ(x1vals, x2vals, diagm(u_opt))
    v = (M'M + λ * I(k)) \ M'y

    avals .= u_opt .* v
end

const DESCENT_METHODS = Dict(
    :BFGS => local_descent_BFGS!,
    :LBFGS => local_descent_LBFGS!,
    :smooth => local_descent_smooth!,
    :smooth_inner => local_descent_smooth_inner!,
    # :lasso_inner => local_descent_lasso_inner!
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
    amplitudes_lo = positivity ? 0 : -Inf

    xgrid = coarse_grid(blasso.domain, gridsize)

    xvals = blasso.dim == 1 ? Float64[] : [Float64[], Float64[]]

    M = sum(abs2.(y)) / (2 * λ)
    avals = Float64[]

    @showprogress desc = "SFW iterations: " for k in 1:maxits
        # for k in 1:maxits

        xnew, η_max = compute_next_spike(xgrid, xvals, avals, adjΦ, Φ, y, λ)

        η_max < tol && break

        lt = @timed lasso_update!(xnew, xvals, avals, M, k, adjΦ, Φ, y, λ, blasso.domain, amplitudes_lo) #TODO need to check this!!

        optimise_amplitudes!(xvals, avals, ϕ, y, λ, amplitudes_lo)

        DESCENT_METHODS[descent](xvals, avals, Φ, y, λ, k, blasso.domain, amplitudes_lo)
    end

    # prune!(xvals, avals, 0.01)

    blasso.μ = blasso.dim == 1 ? DiscreteMeasure(xvals, avals) : DiscreteMeasure(collect(xvals), avals)

    return blasso
end

"""
Solve the BLASSO problem using Boosted Sliding Frank-Wolfe (SFW) algorithm.
Handles both 1D and 2D cases.
"""
function BSFW!(blasso::BLASSO, options::Dict{Symbol,<:Any}=Dict{Symbol,Any}())::BLASSO
    y, λ = blasso.y, blasso.λ
    ops = blasso.operators
    ϕ, Φ, adjΦ = ops.ϕ, ops.Φ, ops.adjΦ

    maxits = get(options, :maxits, 100)
    gridsize = get(options, :gridsize, 21)
    tol1 = get(options, :tol1, 1.05)
    tol2 = get(options, :tol2, 1.00001)
    descent = get(options, :descent, :BFGS)
    positivity = get(options, :positivity, false)
    amplitudes_lo = positivity ? 0 : -Inf

    xgrid = coarse_grid(blasso.domain, gridsize)

    xvals = blasso.dim == 1 ? Float64[] : [Float64[], Float64[]]

    M = sum(abs2.(y)) / (2 * λ)
    avals = Float64[]

    xnew = blasso.dim == 1 ? Float64[] : [Float64[], Float64[]]
    η_max = 0.0
    slid = false

    @showprogress desc = "BSFW iterations: " for k in 1:maxits
        # for k in 1:maxits
        if !slid
            xnew, η_max = compute_next_spike(xgrid, xvals, avals, adjΦ, Φ, y, λ)
        end

        if η_max > tol1
            slid = false
            lasso_update!(xnew, xvals, avals, M, k, adjΦ, Φ, y, λ, blasso.domain, amplitudes_lo)

            optimise_amplitudes!(xvals, avals, ϕ, y, λ, amplitudes_lo)

            prune!(xvals, avals, 0.001)
        else
            slid = true
            if length(avals) > 0
                DESCENT_METHODS[descent](xvals, avals, Φ, y, λ, length(avals), blasso.domain, amplitudes_lo)

                prune!(xvals, avals, 0.001)
            end

            xnew, η_max = compute_next_spike(xgrid, xvals, avals, adjΦ, Φ, y, λ)

            η_max < tol2 && break
        end
    end

    blasso.μ = blasso.dim == 1 ? DiscreteMeasure(xvals, avals) : DiscreteMeasure(collect(xvals), avals)

    return blasso
end