using ..SparseSpikes

export BLASSO


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
    y::AbstractVector{<:Number}             # Observation vector/matrix
    operators::Operators                    # Operators structure
    domain::Union{Vector,Vector{Vector}}    # Domain of the problem
    λ::Union{<:Real,Nothing}                # Regularisation parameter
    μ::Union{DiscreteMeasure,Nothing}       # Recovered measure
    p::Union{Vector{<:Complex},Nothing}     # Dual solution
    dim::Int                                # Spatial dimension of the problem

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