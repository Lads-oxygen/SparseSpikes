using LinearAlgebra: norm

export FH!

"""
Solve the BLASSO problem using Courbot's Fast Homotopy Algorithm.

# Arguments
- `b::BLASSO`: BLASSO problem to solve.
- `options::Dict()`: Options for the solver. Relevant keys:
    - `:maxits::Int` (default: `50`): Maximum number of iterations.
    - `:progress::Bool` (default: `true`): Whether to display progress bar.
    - `:base_solver::Symbol` (default: `:SFW`): Underlying solver to use.
    - `:verbose::Bool` (default: `false`): Whether to print debugging information.
    - `:store_reg_path::Bool` (default: `false`): Whether to store the regularisation path.
    - `:τδ::Real` (required): Noise level x τ
    - `:c::Real` (default: `1.1`): Homotopy shrinkage constant.

# Returns
- `b::BLASSO`: BLASSO problem with the recovered measure in the `μ` field. If `:SDP` is used, the dual solution is also stored in the `p` field.
"""
function FH!(b::BLASSO,
    options::Dict{Symbol,<:Any}=Dict{Symbol,Any}())

    maxits = get(options, :maxits, 50)
    options[:maxits] = get(options, :inner_maxits, 50)
    progress = get(options, :progress, true)
    options[:progress] = false
    base_solver = get(options, :base_solver, :SFW)
    verbose = get(options, :verbose, false)
    store_reg_path = get(options, :store_reg_path, false)
    η_tol = get(options, :η_tol, 1e-5)
    τδ = get(options, :τδ, nothing)
    c = get(options, :c, 0.1)

    if isnothing(τδ)
        throw(ArgumentError("τδ must be provided in options."))
    elseif τδ ≤ 0
        throw(ArgumentError("τδ must be strictly greater than 0"))
    elseif c ≤ 0
        throw(ArgumentError("c must be strictly greater than 0"))
    end

    if store_reg_path
        b.reg_path = Dict(
            :λs => Float64[],
            :μs => DiscreteMeasure[],
        )
    end

    xgrid = grid(b.domain, b.n_coarse_grid)

    # Choose smallest λ such that Φ(μ) = 0, as starting point
    ϕ, Φ, adjΦ = b.ops
    b.λ = λ₀(xgrid, ϕ, b.y, η_tol) - eps(Float64)

    r = norm(b.y)

    prog = ProgressUnknown(desc="FH iterations: ")

    for _ in 1:maxits
        progress && next!(prog)

        verbose && println("λ = $(b.λ)")

        solve!(b, base_solver, options=options)
        r = norm(Φ(b.μ...) - b.y)

        if store_reg_path && !isnothing(b.μ)
            push!(b.reg_path[:λs], b.λ)
            push!(b.reg_path[:μs], deepcopy(b.μ))
        end

        verbose && println("r = $r, τδ = $(τδ)")
        verbose && println("μ = $(b.μ)")

        r < τδ && break

        if isnothing(b.η)
            display(b.λ)
        end

        η_max = compute_η_max(b.η, xgrid)[2]
        b.λ *= η_max / (1 + c)
    end
    return b
end