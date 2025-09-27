using ForwardDiff, Optim

export grid, DiscreteMeasure, sparsity, build_ηV, complex_grad, λ₀

function grid(domain::AbstractVector{T}, n_grid::Int) where {T<:Real}
    range(domain[1], domain[2]; length=n_grid)
end

function grid(domain::AbstractVector{<:AbstractVector{T}}, n_grid::Int) where {T<:Real}
    d = length(domain)
    @assert all(length.(domain) .== 2) "each element of domain must be a length-2 (min,max) vector"
    if d == 1
        return range(domain[1][1], domain[1][2]; length = n_grid)
    elseif d == 2
        rx = collect(range(domain[1][1], domain[1][2]; length = n_grid))
        ry = collect(range(domain[2][1], domain[2][2]; length = n_grid))
        X = repeat(reshape(rx, n_grid, 1), 1, n_grid) # x-coordinates
        Y = repeat(reshape(ry, 1, n_grid), n_grid, 1) # y-coordinates
        return [X, Y]  # Vector{Matrix{T}} of length 2
    else
        error("Only 1D or 2D domains are supported for vector-of-vectors input.")
    end
end

struct DiscreteMeasure{T<:AbstractFloat}
    x::Union{Vector{T},Vector{Vector{T}}}
    a::Vector{T}
    d::Int
    s::Int

    function DiscreteMeasure(x::Vector{T}, a::Vector{T}) where {T<:AbstractFloat}
        s = length(x)
        @assert length(a) == s
        new{T}(x, a, 1, s)
    end

    function DiscreteMeasure(x::Vector{Vector{T}}, a::Vector{T}) where {T<:AbstractFloat}
        d = length(x)
        s = length(x[1])
        @assert all(length(xi) == s for xi in x)
        @assert length(a) == s
        new{T}(x, a, d, s)
    end
end

DiscreteMeasure(x::Matrix{T}, a::Vector{T}) where {T<:Real} = DiscreteMeasure{T}(x, a)

DiscreteMeasure(x::Vector{T}, a::Vector{T}) where {T<:Real} =
    DiscreteMeasure(reshape(x, 1, :), a)

DiscreteMeasure(x1::Vector{T}, x2::Vector{T}, a::Vector{T}) where {T<:Real} = (
    @assert length(x1) == length(x2) == length(a),
DiscreteMeasure(vcat(x1', x2'), a)
)

function DiscreteMeasure(xs::NTuple{D,Vector{T}}, a::Vector{T}) where {D,T<:Real}
    s = length(a)
    @assert all(length(x) == s for x in xs)
    xmat = reduce(vcat, (x' for x in xs)) # D × s
    DiscreteMeasure(xmat, a)
end

sparsity(μ::DiscreteMeasure) = μ.s
sparsity(::Nothing) = 0

function Base.iterate(μ::DiscreteMeasure, state::Int=1)
    if state == 1
        return μ.x, 2
    elseif state == 2
        return μ.a, 3
    else
        return nothing
    end
end


"""
Build the pre-certificate function for a given measure.

# Arguments
- `μ0`: Discrete measure.
- `ops`: Operators.

# Returns
- `ηV`: Pre-certificate.
"""
function build_ηV(μ0::DiscreteMeasure, ops::Operators)::Function
    b = [sign.(μ0.a); zeros(μ0.d * μ0.s)]
    ϕ = ops.ϕ
    if μ0.d == 1
        dϕ = x -> complex_grad(ϕ, x)
        Γx = [ϕ(μ0.x)'; dϕ(μ0.x)']
    elseif μ0.d == 2
        d1ϕ = (x1, x2) -> complex_grad(ξ -> ϕ(ξ, x2), x1)
        d2ϕ = (x1, x2) -> complex_grad(ξ -> ϕ(x1, ξ), x2)
        Γx = [
            ϕ(μ0.x...)';
            d1ϕ(μ0.x...)';
            d2ϕ(μ0.x...)'
        ]
    else
        error("Not implemented")
    end
    pV = Γx \ b
    return grid -> ops.adjΦ(pV; grid=grid)
end

"""
Compute the gradient of a complex-valued function.

# Arguments
- `f`: Complex-valued function.
- `x`: Point at which to evaluate the gradient.

# Returns
- `grad`: Gradient of the function at `x`.
"""
function complex_grad(f, x)
    fx = f(x)# TODO: avoid function call
    if eltype(fx) <: Complex
        real_part = ForwardDiff.jacobian(ξ -> real(f(ξ)), x)
        imag_part = ForwardDiff.jacobian(ξ -> imag(f(ξ)), x)
        jac = real_part + im * imag_part
    else
        jac = ForwardDiff.jacobian(f, x)
    end
    grad = reshape(sum(jac, dims=2), size(fx))
    return grad
end

"""
Compute λ₀ := ||Φ*y||_∞.

# Arguments
- `grid`: Grid of potential spike locations
- `ϕ`: Kernel function
- `y`: Observations
- `η_tol`

# Returns
- `λ₀`: Smallest λ for which μ_λ = 0.
"""
function λ₀(grid, ϕ, y, η_tol)
    vals = abs.(map(x -> dot(ϕ([x]), y), grid))
    best_x = grid[argmax(vals)]

    fneg(x) = -abs(dot(ϕ([x[1]]), y))

    res = optimize(fneg, [best_x], LBFGS(linesearch=LineSearches.BackTracking()), Optim.Options(show_warnings=false))

    return -Optim.minimum(res) / (1 + η_tol)
end

"""
Compute λ₀ := sup_{(x1,x2)} |ϕ(x1,x2)' y| over a 2D tensor grid.

Arguments:
- grids :: Vector of length 2, each an AbstractMatrix (X-coordinate grid, Y-coordinate grid)
- ϕ      :: kernel taking (x1_vector, x2_vector) and returning columns = atoms
- y      :: observation vector
- η_tol  :: 

Returns:
- λ₀ (Float64)
"""
function λ₀(grids::Vector{<:AbstractMatrix}, ϕ, y, η_tol)
    @assert length(grids) == 2 "Expected two grids (x1, x2)."
    grid_x1, grid_x2 = grids
    @assert size(grid_x1) == size(grid_x2) "Grid shapes must match."

    best_val = -Inf
    best_x1 = 0.0
    best_x2 = 0.0
    @inbounds for I in eachindex(grid_x1)
        x1 = grid_x1[I]
        x2 = grid_x2[I]
        v = dot(ϕ([x1], [x2]), y)
        av = abs(v)
        if av > best_val
            best_val = av
            best_x1 = x1
            best_x2 = x2
        end
    end

    x1_min, x1_max = extrema(vec(grid_x1))
    x2_min, x2_max = extrema(vec(grid_x2))

    fneg(z)::Float64 = -abs(dot(ϕ([z[1]], [z[2]]), y))

    lower = [x1_min, x2_min]
    upper = [x1_max, x2_max]
    x0 = [best_x1, best_x2]

    result = optimize(fneg, lower, upper, x0,
        Fminbox(LBFGS(linesearch=LineSearches.BackTracking())),
        Optim.Options(show_warnings=false))

    return -Optim.minimum(result) / (1 + η_tol)
end