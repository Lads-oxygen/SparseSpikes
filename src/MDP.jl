using ..SparseSpikes

export MDP!

"""
Solve the BLASSO problem using the specified solver (`:SDP` or `:SFW`).

# Arguments
- `b::BLASSO`: BLASSO problem to solve.
- `options::Dict()`: Options for the solver. Relevant keys:
    - `:base_solver` (`Symbol`): Underlying solver to use (default `:SFW`).
    - `:δ` (`Real`): Noise level (required).
    - `:τ` (`Real`): Discrepancy constant (default `1.1`).
    - `:q` (`Real`): Regularisation parameter reduction factor (default `0.9`).

# Returns
- `b::BLASSO`: BLASSO problem with the recovered measure in the `μ` field. If `:SDP` is used, the dual solution is also stored in the `p` field.
"""
function MDP!(b::BLASSO,
    options::Dict{Symbol,<:Any}=Dict{Symbol,Any}())

    base_solver = get(options, :base_solver, :SFW)
    δ = get(options, :δ, nothing)
    τ = get(options, :τ, 1.1)
    q = get(options, :q, 0.9)

    if isnothing(δ)
        throw(ArgumentError("Noise level δ must be provided in options."))
    elseif τ <= 1
        throw(ArgumentError("τ must be strictly greater than 1"))
    elseif q <= 0 || q >= 1
        throw(ArgumentError("q must be strictly between 0 and 1"))
    end

    # Choose smallest λ such that Φ(μ) = 0, as starting point
    Φ, adjΦ = b.operators.Φ, b.operators.adjΦ
    b.λ = maximum(adjΦ(b.y))

    r = norm(b.y)
    while r > τ * δ
        println("λ : ", b.λ)
        @time solve!(b, base_solver, options=options)
        r = norm(Φ(b.μ...) - b.y)
        b.λ *= q
        if b.λ < 1e-3δ
            throw(ArgumentError("Regularisation parameter λ has become too small."))
        end
        println("r : ", r)
        println("τδ: ", τ * δ)
        println("μ : ", b.μ)
    end
    return b
end