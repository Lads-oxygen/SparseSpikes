using LinearAlgebra, Optim, ProgressMeter, LineSearches

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

function prune!(xvals::AbstractVector{T}, avals::AbstractVector{T}, factor::Real=1e-2) where {T<:Real}
    isempty(avals) && return

    maxval = maximum(avals)
    idx = findall(x -> abs(x) < factor * maxval, avals)
    deleteat!(xvals, idx)
    deleteat!(avals, idx)
end

function prune!(xvals::AbstractVector{<:AbstractVector{T}}, avals::AbstractVector{T}, factor::Real=1e-2) where {T<:Real}
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
    x0 = [xgrid[idx]]
    η_val = η_k[idx]

    function max_η_k(x)
        x_temp = [x[1]]
        return sign(η_val) * -η(x_temp, xvals, avals, adjΦ, Φ, y, λ)[1]
    end

    xnew, η_max = try
        result = optimize(max_η_k, x0, LBFGS(linesearch=LineSearches.BackTracking()), Optim.Options(g_tol=1e-4); autodiff=:forward)
        (Optim.minimizer(result)[1], -Optim.minimum(result))
    catch e
        println("Optimisation failed, falling back to grid search. Error: ", e)
        # Fallback to a simple grid search if optimization fails
        (x0[1], abs(η_val))
    end

    return xnew, η_max
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
    x0 = [xgrid[1][idx], xgrid[2][idx]]
    η_val = η_k[idx]

    function max_η_k(x)
        x_temp = [[x[1]], [x[2]]]
        return sign(η_val) * -η(x_temp, xvals, avals, adjΦ, Φ, y, λ)[1]
    end

    xnew, η_max = try
        result = optimize(max_η_k, x0, LBFGS(linesearch=LineSearches.BackTracking()), Optim.Options(g_tol=1e-4); autodiff=:forward)
        (Optim.minimizer(result), -Optim.minimum(result))
    catch e
        println("Optimisation failed, falling back to grid search. Error: ", e)
        # Fallback to a simple grid search if optimization fails
        (x0, abs(η_val))
    end

    return xnew, η_max
end

function add_spike!(
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
    anew = M * sign(η([xnew], xvals, avals, adjΦ, Φ, y, λ)[1])

    push!(xvals, xnew)
    avals .*= (1 - 2 / (k + 2))
    push!(avals, 2 / (k + 2) * anew)
end

function add_spike!(
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
    λ::Real,
    min_amplitude::Real
) where {T<:Real}
    X = real(ϕ(xvals)' * ϕ(xvals))
    Xty = real(ϕ(xvals)' * y)
    τ = 1 / norm(X)

    for _ in 1:100
        avals .-= τ * (X * avals .- Xty)
        avals .= max.(sign.(avals) .* max.(abs.(avals) .- τ * λ, 0), min_amplitude)
    end
end

function optimise_amplitudes!(
    xvals::AbstractVector{<:AbstractVector{T}},
    avals::AbstractVector{T},
    ϕ::Function,
    y::AbstractVector{<:Number},
    λ::Real,
    min_amplitude::Real
) where {T<:Real}
    X = real(ϕ(xvals...)' * ϕ(xvals...))
    Xty = real(ϕ(xvals...)' * y)
    τ = 1 / norm(X)

    for _ in 1:100
        avals .-= τ * (X * avals .- Xty)
        avals .= max.(sign.(avals) .* max.(abs.(avals) .- τ * λ, 0), min_amplitude)
    end
end

function local_descent!(
    xvals::AbstractVector{T},
    avals::AbstractVector{T},
    Φ::Function,
    y::AbstractVector{<:Number},
    λ::Real,
    domain::AbstractVector{<:Real},
    min_amplitude::Real,
    optimiser::Optim.AbstractOptimizer,
) where {T<:Real}
    s = length(avals)

    function func(xa)
        x, a = xa[1:s], xa[(s+1):end]
        λ * norm(a, 1) + 0.5 * sum(abs2, y - Φ(x, a))
    end

    xa0 = vcat(xvals, avals)
    lower_bounds = vcat(fill(domain[1], s), fill(min_amplitude, s))
    upper_bounds = vcat(fill(domain[2], s), fill(Inf, s))
    clamp_strict!(xa0, lower_bounds, upper_bounds)

    result = optimize(
        func,
        xa0,
        optimiser,
        Optim.Options(g_tol=1e-4);
        autodiff=:forward
    )
    xa_opt = Optim.minimizer(result)

    clamp_strict!(xa_opt, lower_bounds, upper_bounds)

    xvals .= xa_opt[1:s]
    avals .= xa_opt[(s+1):end]
end

function local_descent!(
    xvals::AbstractVector{<:AbstractVector{T}},
    avals::AbstractVector{T},
    Φ::Function,
    y::AbstractVector{<:Number},
    λ::Real,
    domain::AbstractVector{<:AbstractVector{<:Real}},
    min_amplitude::Real,
    optimiser::Optim.AbstractOptimizer,
) where {T<:Real}
    x1vals, x2vals = xvals

    s = length(avals)

    function func(xa)
        x1 = @view xa[1:s]
        x2 = @view xa[s+1:2s]
        a = @view xa[2s+1:end]
        return λ * norm(a, 1) + 0.5 * sum(abs2, y .- Φ(x1, x2, a))
    end

    ax0 = vcat(x1vals, x2vals, avals)
    lower_bounds = vcat(fill(domain[1][1], s), fill(domain[2][1], s), fill(min_amplitude, s))
    upper_bounds = vcat(fill(domain[1][2], s), fill(domain[2][2], s), fill(Inf, s))
    clamp_strict!(ax0, lower_bounds, upper_bounds)

    result = optimize(
        func,
        ax0,
        optimiser,
        Optim.Options(g_tol=1e-4);
        autodiff=:forward
    )
    xa_opt = Optim.minimizer(result)

    clamp_strict!(xa_opt, lower_bounds, upper_bounds)

    x1vals .= xa_opt[1:s]
    x2vals .= xa_opt[s+1:2s]
    avals .= xa_opt[2s+1:end]
end

"""
Solve the BLASSO problem using Sliding Frank-Wolfe (SFW) algorithm.
Handles both 1D and 2D cases.

# Arguments
- `blasso::BLASSO`: BLASSO problem to solve.
- `options::Dict{Symbol,<:Any}`: Options dictionary with the following supported keys:
  - `:maxits::Int` (default: 100): Maximum number of iterations.
  - `:progress::Bool` (default: true): Whether to display progress bar.
  - `:gridsize::Int` (default: 11): Size of the coarse grid for spike detection. Should be proportional to discretisation of forward operator.
  - `:optimiser::Symbol` (default: `:LBFGS`): Optimizer algorithm. Options are:
      - `:LBFGS`: Limited-memory BFGS optimizer
      - `:BFGS`: BFGS optimizer
  - `:positivity::Bool` (default: false): Whether to enforce positivity constraint on amplitudes.

# Returns
- `blasso::BLASSO`: BLASSO problem with the recovered measure in the `μ` field.
"""
function SFW!(blasso::BLASSO, options::Dict{Symbol,<:Any}=Dict{Symbol,Any}())::BLASSO
    y, λ = blasso.y, blasso.λ
    ops = blasso.operators
    ϕ, Φ, adjΦ = ops.ϕ, ops.Φ, ops.adjΦ

    maxits = get(options, :maxits, 100)
    progress = get(options, :progress, true)
    gridsize = get(options, :gridsize, 11)
    optimiser_symbol = get(options, :optimiser, :LBFGS)
    optimiser = (optimiser_symbol == :LBFGS ? LBFGS : BFGS)(linesearch=LineSearches.BackTracking())
    positivity = get(options, :positivity, false)
    min_amplitude = positivity ? 0 : -Inf

    xgrid = coarse_grid(blasso.domain, gridsize)

    xvals = blasso.dim == 1 ? Float64[] : [Float64[], Float64[]]
    avals = Float64[]

    # Initialize from existing measure if available
    if !isnothing(blasso.μ)
        xvals = copy(blasso.μ.x)
        avals = copy(blasso.μ.a)
        optimise_amplitudes!(xvals, avals, ϕ, y, λ, min_amplitude)
    else
        xvals = blasso.dim == 1 ? Float64[] : [Float64[], Float64[]]
        avals = Float64[]
    end

    M = sum(abs2.(y)) / (2 * λ)

    prog = ProgressUnknown(desc="SFW iterations: ")

    for k in length(avals):maxits
        xnew, η_max = compute_next_spike(xgrid, xvals, avals, adjΦ, Φ, y, λ)

        progress && next!(prog)

        η_max < 1.00001 && break

        add_spike!(xnew, xvals, avals, M, k, adjΦ, Φ, y, λ)

        optimise_amplitudes!(xvals, avals, ϕ, y, λ, min_amplitude)

        local_descent!(xvals, avals, Φ, y, λ, blasso.domain, min_amplitude, optimiser)

        prune!(xvals, avals)
    end

    blasso.μ = blasso.dim == 1 ? DiscreteMeasure(xvals, avals) : DiscreteMeasure(collect(xvals), avals)

    return blasso
end

"""
Solve the BLASSO problem using Boosted Sliding Frank-Wolfe (BSFW) algorithm.
Handles both 1D and 2D cases.

# Arguments
- `blasso::BLASSO`: BLASSO problem to solve.
- `options::Dict{Symbol,<:Any}`: Options dictionary with the following supported keys:
  - `:maxits::Int` (default: 100): Maximum number of iterations.
  - `:progress::Bool` (default: true): Whether to display progress bar.
  - `:gridsize::Int` (default: 11): Size of the coarse grid for spike detection. Should be proportional to discretisation of forward operator.
  - `:optimiser::Symbol` (default: `:LBFGS`): Optimizer algorithm. Options are:
      - `:LBFGS`: Limited-memory BFGS optimizer
      - `:BFGS`: BFGS optimizer
  - `:positivity::Bool` (default: false): Whether to enforce positivity constraint on amplitudes.

# Returns
- `blasso::BLASSO`: BLASSO problem with the recovered measure in the `μ` field.
"""
function BSFW!(blasso::BLASSO, options::Dict{Symbol,<:Any}=Dict{Symbol,Any}())::BLASSO
    y, λ = blasso.y, blasso.λ
    ops = blasso.operators
    ϕ, Φ, adjΦ = ops.ϕ, ops.Φ, ops.adjΦ

    maxits = get(options, :maxits, 100)
    progress = get(options, :progress, true)
    gridsize = get(options, :gridsize, 11)
    optimiser_symbol = get(options, :optimiser, :LBFGS)
    optimiser = (optimiser_symbol == :LBFGS ? LBFGS : BFGS)(linesearch=LineSearches.BackTracking())
    positivity = get(options, :positivity, false)
    min_amplitude = positivity ? 0 : -Inf

    xgrid = coarse_grid(blasso.domain, gridsize)

    xvals = blasso.dim == 1 ? Float64[] : [Float64[], Float64[]]

    M = sum(abs2.(y)) / (2 * λ)
    avals = Float64[]

    xnew = blasso.dim == 1 ? Float64[] : [Float64[], Float64[]]
    η_max = 0.0

    prog = ProgressUnknown(desc="BSFW iterations: ")
    for k in 1:maxits
        progress && next!(prog)

        if η_max > 1.00001

            xnew, η_max = compute_next_spike(xgrid, xvals, avals, adjΦ, Φ, y, λ)

            add_spike!(xnew, xvals, avals, M, k, adjΦ, Φ, y, λ)

            optimise_amplitudes!(xvals, avals, ϕ, y, λ, min_amplitude)

            prune!(xvals, avals)
        else
            if length(avals) > 0
                local_descent!(xvals, avals, Φ, y, λ, blasso.domain, min_amplitude, optimiser)
            end

            prune!(xvals, avals)

            xnew, η_max = compute_next_spike(xgrid, xvals, avals, adjΦ, Φ, y, λ)

            η_max < 1.00001 && break
        end
    end

    blasso.μ = blasso.dim == 1 ? DiscreteMeasure(xvals, avals) : DiscreteMeasure(collect(xvals), avals)

    return blasso
end