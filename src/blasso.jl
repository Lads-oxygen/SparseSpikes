export BLASSO

"""
BLASSO struct to hold the problem data and solutions.

# Fields
- `y`: Observation vector.
- `operators`: Operators structure.
- `domain`: Domain of the problem.
- `n_grid`: Number of grid points.
- `λ`: Regularisation parameter.
- `μ`: Recovered measure.
- `p`: Dual solution.
- `η`: Dual certificate function.
- `d`: Spatial dimension of the problem.
- `reg_path`: Regularisation path data (if stored).
"""
mutable struct BLASSO
    y::AbstractVector{<:Number}
    ops::Operators
    domain::Union{Vector,Vector{Vector}}
    n_coarse_grid::Int
    λ::Union{<:Real,Nothing}
    μ::Union{DiscreteMeasure,Nothing}
    p::Union{Vector{<:Complex},Nothing}
    η::Union{Function,Nothing}
    d::Int
    reg_path::Union{Dict{Symbol,Any},Nothing}

    function BLASSO(
        y::AbstractVector,
        ops::Operators,
        domain::AbstractVector,
        n_coarse_grid::Int;
        λ::Union{<:Real,Nothing}=nothing,
    )
        d = isa(domain, Vector{<:Real}) ? 1 : 2
        new(vec(y), ops, domain, n_coarse_grid, λ, nothing, nothing, nothing, d, nothing)
    end
end