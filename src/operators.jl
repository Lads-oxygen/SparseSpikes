export Operators, fourier_operators_1D, fourier_operators_2D, gaussian_operators_1D, gaussian_operators_2D

struct Operators
    ϕ::Function
    Φ::Function
    adjΦ::Function
end

"""
    fourier_operators_1D(fc, plt_grid)

Forward operator using Fourier kernel with frequency cutoff `fc` 
and sampling grid `plt_grid` for the adjoint.
"""
function fourier_operators_1D(fc::Int, plt_grid::AbstractVector{<:Real})::Operators
    function ϕ(x::AbstractVector{<:Real})
        return exp.(-2im * π * (-fc:fc) * x') / sqrt(2fc + 1)
    end
    function Φ(x::AbstractVector{<:Real}, a::AbstractVector{<:Real})
        return ϕ(x) * a
    end
    function adjΦ(k::AbstractArray{<:Complex}; grid::AbstractVector{<:Real}=plt_grid)
        return real(ϕ(grid)' * k)
    end
    return Operators(ϕ, Φ, adjΦ)
end

# seperable 2d fourier?
"""
    fourier_operators_2D(fc, plt_grid_x1, plt_grid_x2)

Forward operator using Fourier kernel with frequency cutoff `fc` 
and sampling grid `plt_grid_x1` and `plt_grid_x2` for the adjoint.
"""
function fourier_operators_2D(fc::Int, plt_grid_x1::AbstractMatrix{<:Real}, plt_grid_x2::AbstractMatrix{<:Real})::Operators
    k1 = repeat(-fc:fc, 2fc + 1)
    k2 = vcat([fill(k, 2fc + 1) for k in -fc:fc]...)
    K = hcat(k1, k2)
    function ϕ(x1::AbstractArray{<:Real}, x2::AbstractArray{<:Real})::AbstractMatrix{<:Complex}
        return (hcat([exp.(-2im * π * (k1 * vec(x1) + k2 * vec(x2))) / (2fc + 1) for (k1, k2) in eachrow(K)]...))'
    end
    function Φ(x1::AbstractArray{<:Real}, x2::AbstractArray{<:Real}, a::AbstractArray{<:Real})::Vector{<:Complex}
        return ϕ(x1, x2) * a
    end
    function adjΦ(k::AbstractArray{<:Complex}; grid_x1::AbstractArray{<:Real}=plt_grid_x1, grid_x2::AbstractArray{<:Real}=plt_grid_x2)
        return real(reshape(ϕ(grid_x1, grid_x2)' * k, size(grid_x1)...))
    end
    return Operators(ϕ, Φ, adjΦ)
end

"""
    gaussian_operators_1D(σ, plt_grid)

Forward operator using a discretised Gaussian kernel with standard deviation `σ` on grid `plt_grid`.
"""
function gaussian_operators_1D(σ::Real, plt_grid::AbstractVector{<:Real})::Operators
    function gauss1D(x::AbstractVector{<:Real}, μ::Real, σ::Real)
        return exp.(-((x .- μ) .^ 2) ./ (2 * σ^2)) ./ sqrt(2π * σ^2)
    end
    function gauss1DN(x::AbstractVector{<:Real}, μ::Real, σ::Real)
        return gauss1D(x, μ, σ) ./ maximum(gauss1D(x, μ, σ))
    end
    function ϕ(x::AbstractVector{<:Real})
        if isempty(x)
            return zeros(Float64, length(plt_grid), 0)
        else
            return hcat([gauss1DN(plt_grid, xi, σ) for xi in x]...)
        end
    end
    function Φ(x::AbstractVector{<:Real}, a::AbstractVector{<:Real})
        return ϕ(x) * a
    end
    function adjΦ(k::AbstractArray{<:Real}; grid::AbstractVector{<:Real}=plt_grid)
        return ϕ(grid)' * k
    end
    return Operators(ϕ, Φ, adjΦ)
end

"""
    gaussian_operators_2D(σ, plt_grid_x1, plt_grid_x2)

Forward operator using a discretised Gaussian kernel with standard deviation `σ` on grid `plt_grid_x1` and `plt_grid_x2`.
"""
function gaussian_operators_2D(σ::Real, plt_grid_x1::AbstractMatrix{<:Real}, plt_grid_x2::AbstractMatrix{<:Real})::Operators
    function gauss2D(x1::AbstractMatrix{<:Real}, x2::AbstractMatrix{<:Real}, μ::AbstractVector{<:Real}, σ::Real)
        return vcat(exp.(-((x1 .- μ[1]) .^ 2 .+ (x2 .- μ[2]) .^ 2) ./ (σ^2)) ./ (2π * σ^2)...)
    end
    function gauss2DN(x1::AbstractMatrix{<:Real}, x2::AbstractMatrix{<:Real}, μ::AbstractVector{<:Real}, σ::Real)
        return gauss2D(x1, x2, μ, σ) ./ maximum(gauss2D(x1, x2, μ, σ))
    end
    function ϕ(x1::AbstractVector{<:Real}, x2::AbstractVector{<:Real})
        if isempty(x1)
            return zeros(Float64, length(plt_grid_x1), 0)
        else
            return hcat([gauss2DN(plt_grid_x1, plt_grid_x2, [x1[i], x2[i]], σ) for i in eachindex(x1)]...)
        end
    end
    function Φ(x1::AbstractVector{<:Real}, x2::AbstractVector{<:Real}, a::AbstractVector{<:Real})
        return ϕ(x1, x2) * a
    end
    function adjΦ(k::AbstractArray{<:Real}; grid_x1::AbstractArray{<:Real}=plt_grid_x1, grid_x2::AbstractArray{<:Real}=plt_grid_x2)
        return real(reshape(ϕ(vec(grid_x1), vec(grid_x2))' * k, size(grid_x1)...))
    end
    return Operators(ϕ, Φ, adjΦ)
end