using ..SparseSpikes
using Optim

export BLASSO, solve!


"""
BLASSO struct to hold the problem data and solutions.

# Fields
- `y`: Observation vector.
- `operators`: Operators structure.
- `domain`: Domain of the problem.
- `λ`: Regularisation parameter.
- `μ`: Recovered measure.
- `p`: Dual solution.
- `dim`: Spatial dimension of the problem.
"""
mutable struct BLASSO
    y::AbstractVector{<:Number}                              # Observation vector/matrix
    operators::Operators                                     # Operators structure
    domain::Union{Vector,Vector{Vector}}                     # Domain of the problem
    λ::Union{<:Real,Nothing}                                 # Regularisation parameter (nullable)
    μ::Union{DiscreteMeasure,Nothing}                        # Recovered measure
    p::Union{Vector{<:Complex},Nothing}                      # Dual solution
    dim::Int                                                 # Spatial dimension of the problem

    function BLASSO(
        y::AbstractVector,
        operators::Operators,
        domain::Vector,
        λ::Union{<:Real,Nothing}=nothing,
    )
        dim = isa(domain, Vector{<:Real}) ? 1 : 2
        new(vec(y), operators, domain, λ, nothing, nothing, dim)
    end
end

"""
Solve the BLASSO problem using the specified solver (`:SDP` or `:SFW`).

# Arguments
- `prob::BLASSO`: BLASSO problem to solve.
- `solver::Symbol=:SFW`: Solver to use.
- `δ::Union{<:Real,Nothing}`: Noise level.
- `τ::Union{<:Real,Nothing}`: Discrepancy constant.
- `λ0::Real=1e3`: Initial regularisation parameter.
- `q::Real=0.5`: Regularisation parameter reduction factor.
- `options::Dict()`: Options for the solver.

# Returns
- `prob::BLASSO`: BLASSO problem with the recovered measure in the `μ` field. If `:SDP` is used, the dual solution is also stored in the `p` field.
"""
function solve!(prob::BLASSO,
    solver::Symbol=:SFW;
    δ::Union{<:Real,Nothing}=nothing,
    τ::Union{<:Real,Nothing}=nothing,
    λ0::Real=1e3,
    q::Real=0.9,
    options::Dict{Symbol,<:Any}=Dict{Symbol,Any}())

    if q <= 0 || q >= 1
        throw(ArgumentError("q must be strictly between 0 and 1"))
    end
    if τ !== nothing && τ <= 1
        throw(ArgumentError("τ must be strictly greater than 1"))
    end

    if isnothing(prob.λ)
        if δ === nothing || τ === nothing
            error("Either regularisation parameter λ must be provided or both noise level δ and discrepancy constant τ must be specified.")
        end
        prob.λ = λ0
        r = Inf
        while r > τ * δ
            println("λ: ", prob.λ)
            @time solve!(prob, solver, options=options)
            r = norm(prob.operators.Φ(prob.μ...) - prob.y)
            prob.λ *= q
            if prob.λ < 1e-3δ
                throw(ArgumentError("Regularisation parameter λ has become too small."))
            end
            println("r: ", r)
            println("τδ: ", τ * δ)
            println("prob.μ: ", prob.μ)
        end
    else
        if solver == :SFW
            return SFW!(prob, options)
        elseif solver == :BSFW
            return BSFW!(prob, options)
        elseif solver == :SDP
            return SDP!(prob)
        else
            throw(ArgumentError("Solver must be either :SDP, :SFW or :BSFW."))
        end
    end
    return prob
end