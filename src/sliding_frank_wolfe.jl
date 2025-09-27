using Optim, LineSearches, ProgressMeter
using LinearAlgebra: norm

export FW!, SFW!, BSFW!, build_η, compute_η_max

@inline function clamp_strict(x::T, lo::T, hi::T) where {T<:Real}
    @assert hi > lo
    δ = eps(T)
    if x ≤ lo
        return lo + δ
    elseif x ≥ hi
        return hi - δ
    else
        return x
    end
end

@inline function clamp_strict!(x::AbstractArray, lo::AbstractArray, hi::AbstractArray)
    @inbounds @simd for i in eachindex(x, lo, hi)
        x[i] = clamp_strict(x[i], lo[i], hi[i])
    end
    return x
end

function prune!(x::AbstractVector{T}, a::AbstractVector{T}, factor::Real=1e-2) where {T<:Real}
    maxval = maximum(a)
    idx = findall(x -> abs(x) < factor * maxval, a)
    deleteat!(x, idx)
    deleteat!(a, idx)
end

function prune!(x::AbstractVector{<:AbstractVector{T}}, a::AbstractVector{T}, factor::Real=1e-2) where {T<:Real}
    maxval = maximum(a)
    idx = findall(x -> abs(x) < factor * maxval, a)
    deleteat!.(x, Ref(idx))
    deleteat!(a, idx)
end

function build_η(
    x::AbstractVector{T},
    a::AbstractVector{T},
    adjΦ::Function,
    Φ::Function,
    y::AbstractVector{<:Number},
    λ::Real
) where {T<:Real}
    return grid -> adjΦ(y - Φ(x, a), grid=grid) / λ
end

function build_η(
    x::AbstractVector{<:AbstractArray{T}},
    a::AbstractVector{T},
    adjΦ::Function,
    Φ::Function,
    y::AbstractVector{<:Number},
    λ::Real
) where {T<:Real}
    return grids -> adjΦ(y - Φ(x..., a), grid=grids) / λ
end

function compute_η_max_on_grid(
    η::Function,
    xgrid::AbstractVector{T},
) where {T<:Real}
    abs_η_grid = abs.(η(xgrid))
    idx = argmax(abs_η_grid)
    x0 = xgrid[idx]
    η_max = abs_η_grid[idx]
    return x0, η_max
end

function compute_η_max(
    η::Function,
    xgrid::AbstractVector{T},
) where {T<:Real}
    τg = 1e-8
    τc = -1e-8
    τ_seed = 0.95
    δx = 1e-8

    abs_η_grid = abs.(η(xgrid))
    thresh = τ_seed * maximum(abs_η_grid)
    idxs = findall(>=(thresh), abs_η_grid)
    x0s = xgrid[idxs]
    abs_η_vals = abs_η_grid[idxs]

    ηs(x) = η([x])[1]
    η′(x) = ForwardDiff.derivative(ηs, x)
    η′′(x) = ForwardDiff.derivative(η′, x)

    # Replace local minimum seed with a seed either side
    for (idx, x0) in pairs(x0s)
        if abs(η′(x0)) < τg && η′′(x0) ≥ τc
            xs = vcat(x0 .+ δx, x0 .- δx)
            best_idx = argmax(abs.(η(xs)))
            x0 = xs[best_idx]
            abs_η_vals[idx] = abs(ηs(x0))
        end
    end

    Δ = step(xgrid)
    domain = extrema(xgrid)

    abs_η(z) = -abs(η([z[1]])[1])

    cands = [(
        try
            lb = max(domain[1], x0 - Δ)
            ub = min(domain[2], x0 + Δ)
            x0 = clamp_strict(x0, lb, ub)
            result = optimize(
                abs_η,
                [lb],
                [ub],
                [x0],
                Fminbox(LBFGS(linesearch=LineSearches.BackTracking())),
                Optim.Options(show_warnings=false); autodiff=:forward)
            (Optim.minimizer(result)[1], -Optim.minimum(result))
        catch e
            @warn "Optimisation failed, falling back to grid search. Error:" error = e
            (x0, abs_η_vals[idx])
        end
    ) for (idx, x0) in pairs(x0s)]

    idx = argmax(last.(cands))
    x_new, η_max = cands[idx]

    return x_new, η_max
end

function compute_η_max(
    η::Function,
    xgrids::AbstractVector{<:AbstractArray{T}},
) where {T<:Real}
    τg = 1e-8
    τc = -1e-8
    τ_seed = 0.95
    δx = 1e-8
    
    abs_η_grid = abs.(η(xgrids))
    thresh = τ_seed * maximum(abs_η_grid)
    idxs = findall(>=(thresh), abs_η_grid)
    x0s = [xgrids[2][idxs], xgrids[1][idxs]]
    x0s = permutedims(reduce(hcat, x0s))  # size: d × N
    abs_η_vals = abs_η_grid[idxs]

    ηs(z) = η([[z[1]], [z[2]]])[1]
    η′(z) = ForwardDiff.gradient(ηs, z)
    η′′(z) = ForwardDiff.hessian(ηs, z)

    # Replace local minimum seed with a seed either side
    for (idx, x0) in enumerate(eachcol(x0s))
        g = η′(x0)
        H = η′′(x0)
        if norm(g) < τg && minimum(eigvals(H)) ≥ τc
            candidates = [
                [clamp_strict(x0[1] + δx, domain[1][1], domain[1][2]), x0[2]],
                [clamp_strict(x0[1] - δx, domain[1][1], domain[1][2]), x0[2]],
                [x0[1], clamp_strict(x0[2] + δx, domain[2][1], domain[2][2])],
                [x0[1], clamp_strict(x0[2] - δx, domain[2][1], domain[2][2])]
            ]
            vals = [abs(ηs(c)) for c in candidates]
            best = argmax(vals)
            x0s[:, idx] .= candidates[best]
            abs_η_vals[idx] = vals[best]
        end
    end

    Δ = xgrids[1][2, 2] - xgrids[1][1, 1]
    domain = (extrema(xgrids[1]), extrema(xgrids[2]))

    abs_η(z) = -abs(η([[z[1]], [z[2]]])[1])

    cands = [(
        try
            lb = [max(domain[1][1], x0[1] - Δ), max(domain[2][1], x0[2] - Δ)]
            ub = [min(domain[1][2], x0[1] + Δ), min(domain[2][2], x0[2] + Δ)]
            x0 = [clamp_strict(x0[1], lb[1], ub[1]), clamp_strict(x0[2], lb[2], ub[2])]
            result = optimize(abs_η, lb, ub, x0, Fminbox(LBFGS(linesearch=LineSearches.BackTracking())), Optim.Options(show_warnings=false); autodiff=:forward)
            (Optim.minimizer(result), -Optim.minimum(result))
        catch e
            @warn "Optimisation failed, falling back to grid search. Error:" error = e
            (x0, abs_η_vals[idx])
        end
    ) for (idx, x0) in enumerate(eachcol(x0s))]

    idx = argmax(last.(cands))
    x_new, η_max = cands[idx]

    return x_new, η_max
end

function add_spike!(
    x_new::T,
    η::Function,
    x::AbstractVector{T},
    a::AbstractVector{T},
    M::Real,
    k::Int,
) where {T<:Real}
    if any(isapprox(xi, x_new) for xi in x)
        return
    end
    anew = M * sign(η([x_new])[1])
    a .*= (1 - 2 / (k + 2))
    push!(x, x_new)
    push!(a, 2 / (k + 2) * anew)
end

function add_spike!(
    x_new::AbstractVector{T},
    η::Function,
    x::AbstractVector{<:AbstractVector{T}},
    a::AbstractVector{T},
    M::Real,
    k::Int,
) where {T<:Real}
    if any(i -> all(j -> isapprox(x[j][i], x_new[j]), eachindex(x)), eachindex(x[1]))
        return
    end
    anew = M * sign(η([[x] for x in x_new])[1])
    a .*= (1 - 2 / (k + 2))
    push!.(x, x_new)
    push!(a, 2 / (k + 2) * anew)
end

function optimise_amplitudes!(
    x::Union{AbstractVector{T},AbstractVector{<:AbstractVector{T}}},
    a::AbstractVector{T},
    ϕ::Function,
    y::AbstractVector{<:Number},
    λ::Real,
    min_amplitude::Real
) where {T<:Real}
    Φx = x isa AbstractVector{<:AbstractVector} ? ϕ(x...) : ϕ(x)
    G = real(Φx' * Φx)
    g = real(Φx' * y)
    τ = 1 / max(norm(G), eps(eltype(G)))

    for _ in 1:100
        a .-= τ * (G * a .- g)
        a .= max.(sign.(a) .* max.(abs.(a) .- τ * λ, 0), min_amplitude)
    end
end

function local_descent!(
    x::AbstractVector{T},
    a::AbstractVector{T},
    Φ::Function,
    y::AbstractVector{<:Number},
    λ::Real,
    domain::AbstractVector{<:Real},
    min_amplitude::Real,
    optimiser::Optim.AbstractOptimizer,
) where {T<:Real}
    s = length(a)

    function obj(xa)
        xv = @view xa[1:s]
        av = @view xa[s+1:end]
        λ * sum(abs, av) + 0.5 * sum(abs2, y .- Φ(xv, av))
    end

    xa0 = vcat(x, a)
    lower = vcat(fill(domain[1], s), fill(min_amplitude, s))
    upper = vcat(fill(domain[2], s), fill(Inf, s))
    clamp_strict!(xa0, lower, upper)

    result = optimize(
        obj,
        xa0,
        optimiser,
        Optim.Options(show_warnings=false);
        autodiff=:forward
    )
    xa_opt = Optim.minimizer(result)

    clamp_strict!(xa_opt, lower, upper)

    x .= xa_opt[1:s]
    a .= xa_opt[s+1:end]
end

function local_descent!(
    x::AbstractVector{<:AbstractVector{T}},
    a::AbstractVector{T},
    Φ::Function,
    y::AbstractVector{<:Number},
    λ::Real,
    domain::AbstractVector{<:AbstractVector{<:Real}},
    min_amplitude::Real,
    optimiser::Optim.AbstractOptimizer,
) where {T<:Real}
    x1, x2 = x

    s = length(a)

    function func(xa)
        x1v = @view xa[1:s]
        x2v = @view xa[s+1:2*s]
        av = @view xa[2*s+1:end]
        return λ * norm(av, 1) + 0.5 * sum(abs2, y .- Φ(x1v, x2v, av))
    end

    xa0 = vcat(x1, x2, a)
    lower = T.(vcat(fill(domain[1][1], s), fill(domain[2][1], s), fill(min_amplitude, s)))
    upper = T.(vcat(fill(domain[1][2], s), fill(domain[2][2], s), fill(Inf, s)))
    clamp_strict!(xa0, lower, upper)

    result = optimize(
        func,
        xa0,
        optimiser,
        Optim.Options(show_warnings=false);
        autodiff=:forward
    )
    xa_opt = Optim.minimizer(result)

    clamp_strict!(xa_opt, lower, upper)

    x1 .= xa_opt[1:s]
    x2 .= xa_opt[s+1:2*s]
    a .= xa_opt[2*s+1:end]
end

"""
Solve the LASSO problem using Frank-Wolfe (FW) algorithm.
Handles both 1D and 2D cases.

# Arguments
- `b::BLASSO`: BLASSO problem to solve.
- `options::Dict{Symbol,<:Any}`: Options dictionary with the following supported keys:
  - `:maxits::Int` (default: 100): Maximum number of iterations.
  - `:progress::Bool` (default: true): Whether to display progress bar.
  - `:optimiser::Symbol` (default: `:LBFGS`): Optimizer algorithm. Options are:
      - `:LBFGS`: Limited-memory BFGS optimizer
      - `:BFGS`: BFGS optimizer
  - `:η_tol::Real` (default: 1e-8): Tolerance for stopping criterion based on η.
  - `:positivity::Bool` (default: false): Whether to enforce positivity constraint on amplitudes.

# Returns
- `b::BLASSO`: BLASSO problem with the recovered measure in the `μ` field.
"""
function FW!(b::BLASSO, options::Dict{Symbol,<:Any}=Dict{Symbol,Any}())::BLASSO
    y, λ, domain = b.y, b.λ, b.domain
    ϕ, Φ, adjΦ = b.ops

    maxits = get(options, :maxits, 100)
    progress = get(options, :progress, true)
    η_tol = get(options, :η_tol, 1e-5)
    positivity = get(options, :positivity, false)
    min_amplitude = positivity ? 0 : -Inf

    xgrid = grid(domain, b.n_coarse_grid)

    # Warm start if μ is provided
    if sparsity(b.μ) != 0
        x = copy(b.μ.x)
        a = copy(b.μ.a)
        optimise_amplitudes!(x, a, ϕ, y, λ, min_amplitude)
    else
        x = b.d == 1 ? Float64[] : [Float64[], Float64[]]
        a = Float64[]
    end

    M = sum(abs2.(y)) / (2 * λ)
    η_k = nothing

    prog = ProgressUnknown(desc="SFW iterations: ")

    for k in length(a):maxits-1
        progress && next!(prog)

        η_k = build_η(x, a, adjΦ, Φ, y, λ)

        x_new, η_max = compute_η_max_on_grid(η_k, xgrid)

        η_max < (1 + η_tol) && break

        add_spike!(x_new, η_k, x, a, M, k)

        optimise_amplitudes!(x, a, ϕ, y, λ, min_amplitude)

        prune!(x, a, 1e-2)
    end

    b.η = η_k
    b.μ = DiscreteMeasure(x, a)
    return b
end

"""
Solve the BLASSO problem using Sliding Frank-Wolfe (SFW) algorithm.
Handles both 1D and 2D cases.

# Arguments
- `b::BLASSO`: BLASSO problem to solve.
- `options::Dict{Symbol,<:Any}`: Options dictionary with the following supported keys:
  - `:maxits::Int` (default: 100): Maximum number of iterations.
  - `:progress::Bool` (default: true): Whether to display progress bar.
  - `:optimiser::Symbol` (default: `:LBFGS`): Optimizer algorithm. Options are:
      - `:LBFGS`: Limited-memory BFGS optimizer
      - `:BFGS`: BFGS optimizer
  - `:η_tol::Real` (default: 1e-8): Tolerance for stopping criterion based on η.
  - `:positivity::Bool` (default: false): Whether to enforce positivity constraint on amplitudes.

# Returns
- `b::BLASSO`: BLASSO problem with the recovered measure in the `μ` field.
"""
function SFW!(b::BLASSO, options::Dict{Symbol,<:Any}=Dict{Symbol,Any}())::BLASSO
    y, λ, domain = b.y, b.λ, b.domain
    ϕ, Φ, adjΦ = b.ops

    maxits = get(options, :maxits, 100)
    progress = get(options, :progress, true)
    optimiser_symbol = get(options, :optimiser, :LBFGS)
    optimiser = (optimiser_symbol == :LBFGS ? LBFGS : BFGS)(linesearch=LineSearches.BackTracking())
    η_tol = get(options, :η_tol, 1e-5)
    positivity = get(options, :positivity, false)
    min_amplitude = positivity ? 0 : -Inf

    xgrid = grid(domain, b.n_coarse_grid)

    # Warm start if μ is provided
    if sparsity(b.μ) != 0
        x = copy(b.μ.x)
        a = copy(b.μ.a)
        local_descent!(x, a, Φ, y, λ, domain, min_amplitude, optimiser)
    else
        x = b.d == 1 ? Float64[] : [Float64[], Float64[]]
        a = Float64[]
    end

    M = sum(abs2.(y)) / (2 * λ)
    η_k = nothing

    prog = ProgressUnknown(desc="SFW iterations: ")

    for k in length(a):maxits-1
        progress && next!(prog)

        η_k = build_η(x, a, adjΦ, Φ, y, λ)

        x_new, η_max = compute_η_max(η_k, xgrid)
        
        η_max < (1 + η_tol) && break

        add_spike!(x_new, η_k, x, a, M, k)

        optimise_amplitudes!(x, a, ϕ, y, λ, min_amplitude)

        local_descent!(x, a, Φ, y, λ, b.domain, min_amplitude, optimiser)
    end

    b.η = η_k
    b.μ = DiscreteMeasure(x, a)
    return b
end

"""
Solve the BLASSO problem using Boosted Sliding Frank-Wolfe (BSFW) algorithm.
Handles both 1D and 2D cases.

# Arguments
- `b::BLASSO`: BLASSO problem to solve.
- `options::Dict{Symbol,<:Any}`: Options dictionary with the following supported keys:
  - `:maxits::Int` (default: 100): Maximum number of iterations.
  - `:progress::Bool` (default: true): Whether to display progress bar.
  - `:optimiser::Symbol` (default: `:LBFGS`): Optimizer algorithm. Options are:
      - `:LBFGS`: Limited-memory BFGS optimizer
      - `:BFGS`: BFGS optimizer
  - `:η_tol::Real` (default: 1e-8): Tolerance for stopping criterion based on η.
  - `:positivity::Bool` (default: false): Whether to enforce positivity constraint on amplitudes.

# Returns
- `b::BLASSO`: BLASSO problem with the recovered measure in the `μ` field.
"""
function BSFW!(b::BLASSO, options::Dict{Symbol,<:Any}=Dict{Symbol,Any}())::BLASSO
    y, λ, domain = b.y, b.λ, b.domain
    ϕ, Φ, adjΦ = b.ops

    maxits = get(options, :maxits, 100)
    progress = get(options, :progress, true)
    optimiser_symbol = get(options, :optimiser, :LBFGS)
    optimiser = (optimiser_symbol == :LBFGS ? LBFGS : BFGS)(linesearch=LineSearches.BackTracking())
    η_tol = get(options, :η_tol, 1e-5)
    positivity = get(options, :positivity, false)
    min_amplitude = positivity ? 0 : -Inf

    xgrid = grid(domain, b.n_coarse_grid)

    # Warm start if μ is provided
    if sparsity(b.μ) != 0
        x = copy(b.μ.x)
        a = copy(b.μ.a)
        local_descent!(x, a, Φ, y, λ, domain, min_amplitude, optimiser)
    else
        x = b.d == 1 ? Float64[] : [Float64[], Float64[]]
        a = Float64[]
    end

    M = sum(abs2.(y)) / (2 * λ)

    x_new = nothing
    η_max = nothing
    η_k = nothing
    slid = false

    prog = ProgressUnknown(desc="BSFW iterations: ")

    for k in 0:maxits-1
        progress && next!(prog)

        if slid == false
            η_k = build_η(x, a, adjΦ, Φ, y, λ)
            x_new, η_max = compute_η_max(η_k, xgrid)
        end

        if η_max > (1.1)
            add_spike!(x_new, η_k, x, a, M, k)

            optimise_amplitudes!(x, a, ϕ, y, λ, min_amplitude)

            prune!(x, a, 1e-3)

            slid = false
        else
            if !isempty(a)
                local_descent!(x, a, Φ, y, λ, domain, min_amplitude, optimiser)

                prune!(x, a, 1e-3)
            end

            η_k = build_η(x, a, adjΦ, Φ, y, λ)
            x_new, η_max = compute_η_max(η_k, xgrid)

            η_max < (1 + η_tol) && break

            slid = true
        end
    end

    b.η = η_k
    b.μ = DiscreteMeasure(x, a)
    return b
end