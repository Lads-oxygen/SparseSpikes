using LinearAlgebra: norm

export MDP!

"""
Solve the BLASSO problem using Morozov's discrepancy principle.

# Arguments
- `b::BLASSO`: BLASSO problem to solve.
- `options::Dict()`: Options for the solver. Relevant keys:
    - `:maxits::Int` (default: `50`): Maximum number of iterations.
    - `:progress::Bool` (default: true): Whether to display progress bar.
    - `:base_solver::Symbol` (default: `:SFW`): Underlying solver to use.
    - `:verbose::Bool` (default: `false`): Whether to print debugging information.
    - `:store_reg_path::Bool` (default: `false`): Whether to store the regularisation path.
    - `:τδ::Real` (required): Noise level x τ
    - `:q::Real` (default: `0.9`): Regularisation parameter reduction factor.

# Returns
- `b::BLASSO`: BLASSO problem with the recovered measure in the `μ` field. If `:SDP` is used, the dual solution is also stored in the `p` field.
"""
function MDP!(b::BLASSO,
    options::Dict{Symbol,<:Any}=Dict{Symbol,Any}())

    maxits = get(options, :maxits, 50)
    options[:maxits] = get(options, :inner_maxits, 50)
    progress = get(options, :progress, true)
    options[:progress] = false
    base_solver = get(options, :base_solver, :SFW)
    warm_start = get(options, :warm_start, true)
    verbose = get(options, :verbose, false)
    store_reg_path = get(options, :store_reg_path, false)
    η_tol = get(options, :η_tol, 1e-5)
    τδ = get(options, :τδ, nothing)
    q = get(options, :q, 0.9)

    if isnothing(τδ)
        throw(ArgumentError("τδ must be provided in options."))
    elseif τδ ≤ 0
        throw(ArgumentError("τδ must be strictly greater than 0"))
    elseif q ≤ 0 || q ≥ 1
        throw(ArgumentError("q must be strictly between 0 and 1"))
    end

    if store_reg_path
        b.reg_path = Dict(
            :λs => Float64[],
            :μs => DiscreteMeasure[],
        )
    end

    # Choose smallest λ such that Φ(μ) = 0, as starting point
    # b.λ = λ₀(build_grid(b.domain, b.n_coarse_grid), b.ops.ϕ, b.y, η_tol)
    b.λ = 3

    r = norm(b.y)

    prog = ProgressUnknown(desc="MDP iterations: ")

    for _ in 1:maxits
        progress && next!(prog)

        verbose && println("λ = $(b.λ)")

        !warm_start && (b.μ = nothing)
        solve!(b, base_solver, options=options)
        r = norm(b.ops.Φₓ(b.μ...) - b.y)

        if store_reg_path && !isnothing(b.μ)
            push!(b.reg_path[:λs], b.λ)
            push!(b.reg_path[:μs], deepcopy(b.μ))
        end

        verbose && println("r = $r, τδ = $(τδ)")
        verbose && println("μ = $(b.μ)")

        r < τδ && break

        b.λ *= q
    end
    return b
end