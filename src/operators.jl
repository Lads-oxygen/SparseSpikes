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
    K = -fc:fc
    scale = 1 / sqrt(2fc + 1)

    function ϕ(x::AbstractVector{T}) where {T<:Real}
        return @. exp(-2im * π * K * x') * scale
    end

    function Φ(x::AbstractVector{<:Real}, a::AbstractVector{<:Real})
        return ϕ(x) * a
    end

    function adjΦ(k::AbstractArray{<:Complex}; grid::AbstractVector{<:Real}=plt_grid)
        return real(ϕ(grid)' * k)
    end

    return Operators(ϕ, Φ, adjΦ)
end

"""
    fourier_operators_2D(fc, plt_grid_x1, plt_grid_x2)

Forward operator using Fourier kernel with frequency cutoff `fc` 
and sampling grid `plt_grid_x1` and `plt_grid_x2` for the adjoint.
"""
function fourier_operators_2D(fc::Int, plt_grid_x1::AbstractMatrix{<:Real}, plt_grid_x2::AbstractMatrix{<:Real})::Operators
    k1 = -fc:fc
    k2 = -fc:fc
    scale = 1 / (2fc + 1)
    nfreq2 = (2fc + 1)^2

    function ϕ(x1::AbstractArray{T1}, x2::AbstractArray{T2})::AbstractMatrix{<:Complex} where {T1<:Real,T2<:Real}
        points = hcat(vec(x1), vec(x2))
        n_points = size(points, 1)
        T = promote_type(T1, T2, Float64)
        result = Matrix{Complex{T}}(undef, nfreq2, n_points)

        @inbounds for i in 1:n_points
            idx = 1
            for k2i in k2, k1i in k1
                result[idx, i] = exp(-2im * π * T(k1i * points[i, 1] + k2i * points[i, 2])) * scale
                idx += 1
            end
        end
        return result
    end

    function Φ(x1::AbstractArray{T}, x2::AbstractArray{T}, a::AbstractArray{T})::Vector{<:Complex} where {T<:Real}
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
    dσ2 = 1 / σ^2
    grid_length = length(plt_grid)

    function gauss1D(μ::AbstractVector{T}) where {T<:Real}
        distances = @. (plt_grid - μ)^2
        output = @. exp(-distances * dσ2)
        output ./= maximum(output)
        return output
    end

    function ϕ(x::AbstractVector{T}) where {T<:Real}
        n_points = length(x)
        result = Matrix{T}(undef, grid_length, n_points)
        for i in 1:n_points
            view(result, :, i) .= gauss1D([x[i]])
        end
        return result
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
    σ2 = 1 / (2 * σ^2)
    grid_length = length(plt_grid_x1)
    grid_points = hcat(vec(plt_grid_x1), vec(plt_grid_x2))

    function gauss2D(μ::AbstractVector{T}, output::AbstractVector{<:Real}) where {T<:Real}
        @fastmath @inbounds begin
            distances = sum(abs2, grid_points .- permutedims(μ), dims=2)
            @. output = exp(-distances * σ2)
            output ./= maximum(output)
            # output ./= sum(output)
        end
        return output
    end

    function gauss2D(μ::AbstractVector{T}) where {T<:Real}
        # Use the same type as the computation result to ensure compatibility with ForwardDiff
        S = typeof(exp(-zero(T) * σ2))
        output = Vector{S}(undef, size(grid_points, 1))
        return gauss2D(μ, output)
    end

    function ϕ(x1::AbstractVector{T1}, x2::AbstractVector{T2}) where {T1<:Real,T2<:Real}
        n_points = length(x1)
        # T = promote_type(T1, T2, Float32)
        T = promote_type(eltype(T1), eltype(T2), Float32)
        result = Matrix{T}(undef, grid_length, n_points)
        for i in 1:n_points
            view(result, :, i) .= gauss2D([x1[i], x2[i]])
        end
        return result
    end

    # Modified Φ that avoids allocating the full matrix
    function Φ(x1::AbstractVector{<:Real}, x2::AbstractVector{<:Real}, a::AbstractVecOrMat{<:Real})
        n_points = length(x1)

        # Use a more flexible type that can handle dual numbers
        T = promote_type(eltype(x1), eltype(x2), eltype(a), Float32)
        result = zeros(T, grid_length)

        if n_points < 100
            return ϕ(x1, x2) * a
        end

        # Multithreaded implementation for large point sets to avoid creating large matrix when calling kernel ϕ
        buffers = [similar(result) for _ in 1:Threads.nthreads()]
        results = [zeros(T, grid_length) for _ in 1:Threads.nthreads()]

        Threads.@threads for i in 1:n_points
            tid = Threads.threadid()
            local_μ = zeros(T, 2)
            local_μ[1] = x1[i]
            local_μ[2] = x2[i]
            gauss2D(local_μ, buffers[tid])
            axpy!(a[i], buffers[tid], results[tid])
        end

        # Combine results from all threads
        for tid in 1:Threads.nthreads()
            axpy!(1.0, results[tid], result)
        end

        return result
    end

    function adjΦ(k::AbstractArray{<:Real}; grid_x1::AbstractArray{<:Real}=plt_grid_x1, grid_x2::AbstractArray{<:Real}=plt_grid_x2)
        return real(reshape(ϕ(vec(grid_x1), vec(grid_x2))' * k, size(grid_x1)...))
    end

    return Operators(ϕ, Φ, adjΦ)
end