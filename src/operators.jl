export Operators, fourier_operators_1D, fourier_operators_2D, gaussian_operators_1D, gaussian_operators_2D

struct Operators
    ϕ::Function
    Φ::Function
    adjΦ::Function
    kind::Symbol
end

"""
Forward operator using Fourier kernel with frequency cutoff `fc` 
and sampling grid `plt_grid` for the adjoint.
"""
function fourier_operators_1D(fc::Int, plt_grid::AbstractVector{<:Real})::Operators
    K = collect(-fc:fc)
    nK = 2 * fc + 1
    norm_cst = 1 / sqrt(nK)

    function ϕ(x::AbstractVector{<:Real})
        return @. cis(-2π * K * x') * norm_cst
    end

    Φ(x::AbstractVector{T}, a::AbstractVector{T}) where {T<:Real} = ϕ(x) * a

    adjΦ(k::AbstractArray{<:Complex}; grid::AbstractVector{<:Real}=plt_grid) = real(ϕ(grid)' * k)

    return Operators(ϕ, Φ, adjΦ, :fourier)
end

"""
Forward operator using Fourier kernel with frequency cutoff `fc` 
and sampling grids `plt_grids` for the adjoint.
"""
function fourier_operators_2D(fc::Int, plt_grids::Vector{<:AbstractMatrix{<:Real}})::Operators
    k1 = -fc:fc
    k2 = -fc:fc
    nK = 2 * fc + 1
    nF = nK^2
    norm_cst = 1 / nK

    function ϕ(x1::AbstractVector{T1}, x2::AbstractVector{T2}) where {T1<:Real,T2<:Real}
        RT = promote_type(T1, T2, typeof(norm_cst))
        sc = RT(norm_cst)
        twopi = RT(2π)
        M = Matrix{Complex{RT}}(undef, nF, length(x1))
        @inbounds for j in eachindex(x1)
            xx = RT(x1[j])
            yy = RT(x2[j])
            idx = 1
            @inbounds for κ2 in k2, κ1 in k1
                θ = -twopi * (κ1 * xx + κ2 * yy)
                M[idx, j] = cis(θ) * sc
                idx += 1
            end
        end
        return M
    end
    ϕ(x::AbstractVector{<:AbstractVector{<:Real}}) = ϕ(x...)

    Φ(x1::AbstractVector{T}, x2::AbstractVector{T}, a::AbstractVector{T}) where {T<:Real} = ϕ(x1, x2) * a
    Φ(x::AbstractVector{<:AbstractVector{T}}, a::AbstractVector{T}) where {T<:Real} = ϕ(x) * a

    function adjΦ(k::AbstractVector; grid=plt_grids)
        g1, g2 = grid
        realkT = real(eltype(k))
        RT = promote_type(realkT, eltype(g1), eltype(g2), typeof(norm_cst))
        kC = Complex{RT}.(k)
        sc = RT(norm_cst)
        k1v = RT.(k1)
        k2v = RT.(k2)

        adj_value_at(x, y) = begin
            xRT = RT(x)
            yRT = RT(y)
            acc = zero(Complex{RT})
            idx = 1
            @inbounds for κ2 in k2v, κ1 in k1v
                acc += kC[idx] * cis(2π * (κ1 * xRT + κ2 * yRT))
                idx += 1
            end
            return real(acc * sc)
        end

        if isa(g1, AbstractMatrix) && isa(g2, AbstractMatrix)
            @assert size(g1) == size(g2)
            nx, ny = size(g1)
            out = Matrix{RT}(undef, nx, ny)
            @inbounds for j in 1:ny, i in 1:nx
                out[i, j] = adj_value_at(g1[i, j], g2[i, j])
            end
            return permutedims(out)
        elseif isa(g1, AbstractVector) && isa(g2, AbstractVector)
            nx, ny = length(g1), length(g2)
            out = Matrix{RT}(undef, nx, ny)
            @inbounds for j in 1:ny
                yv = g2[j]
                for i in 1:nx
                    out[i, j] = adj_value_at(g1[i], yv)
                end
            end
            return permutedims(out)
        else
            throw(ArgumentError("grid must be [X,Y] matrices or [xvec,yvec] vectors"))
        end
    end

    return Operators(ϕ, Φ, adjΦ, :fourier)
end

"""
Forward operator using a discretised Gaussian kernel with standard deviation `σ` on grid `coarse_grid`.
"""
function gaussian_operators_1D(σ::Real, coarse_grid::AbstractVector{<:Real})::Operators
    N = length(coarse_grid)
    invσ2 = (1 / σ)^2

    function _gauss!(μ, buf)
        @inbounds @simd for i in 1:N
            dx = coarse_grid[i] - μ
            buf[i] = exp(-dx * dx * invσ2)
        end
        buf
    end

    function gauss1D(μ::T) where {T<:Real}
        RT = promote_type(T, eltype(coarse_grid))
        buf = Vector{RT}(undef, N)
        _gauss!(RT(μ), buf)
    end

    function ϕ(x::AbstractVector{T}) where {T<:Real}
        s = length(x)
        RT = promote_type(T, eltype(coarse_grid))
        K = Matrix{RT}(undef, N, s)
        @inbounds for j in 1:s
            K[:, j] = gauss1D(x[j])
        end
        K
    end

    Φ(x::AbstractVector, a::AbstractVector) = ϕ(x) * a

    adjΦ(k::AbstractVector; grid::AbstractVector{<:Real}=coarse_grid) = ϕ(grid)' * k

    return Operators(ϕ, Φ, adjΦ, :gaussian)
end

"""
Forward operator using a Gaussian kernel with standard deviation `σ`, sampled on grids `meas_grids`.
"""
function gaussian_operators_2D(σ::Real, meas_grids::AbstractVector{<:AbstractArray{<:Real}})::Operators
    G1, G2 = meas_grids
    g1 = vec(G1)
    g2 = vec(G2)
    N = length(g1)
    inv2σ2 = 1 / (2σ^2)

    function gauss2D(μ1::T1, μ2::T2) where {T1<:Real,T2<:Real}
        RT = promote_type(T1, T2, eltype(g1))
        v = Vector{RT}(undef, N)
        @inbounds @simd for i in 1:N
            d1 = g1[i] - μ1
            d2 = g2[i] - μ2
            v[i] = exp(-(d1 * d1 + d2 * d2) * RT(inv2σ2))
        end
        v
    end

    function ϕ(x1::AbstractVector{T1}, x2::AbstractVector{T2}) where {T1<:Real,T2<:Real}
        @assert length(x1) == length(x2)
        s = length(x1)
        RT = promote_type(T1, T2)
        M = Matrix{RT}(undef, N, s)
        @inbounds for j in 1:s
            M[:, j] = gauss2D(x1[j], x2[j])
        end
        M
    end
    ϕ(x::AbstractVector{<:AbstractVector{<:Real}}) = ϕ(x...)
    function ϕ(x::AbstractVector)
        @assert length(x) == 2
        return ϕ([x[1]], [x[2]])
    end

    Φ(x1::AbstractVector{T}, x2::AbstractVector{T}, a::AbstractVector{T}) where {T<:Real} = ϕ(x1, x2) * a
    Φ(x::AbstractVector{<:AbstractVector{T}}, a::AbstractVector{T}) where {T<:Real} = ϕ(x) * a

    @inline gridvecs(h1::AbstractMatrix, h2::AbstractMatrix) = (vec(h1), vec(h2))
    @inline gridvecs(h1::AbstractVector, h2::AbstractVector) = (
        repeat(h1, inner=length(h2)), repeat(h2, outer=length(h1))
    )
    @inline aspair(g) = isa(g, Tuple) ? g : (g[1], g[2])
    @inline gridshape(g) = isa(first(aspair(g)), AbstractMatrix) ? size(first(aspair(g))) :
                           (length(aspair(g)[1]), length(aspair(g)[2]))

    @inline function kernel_matrix(dst)
        Xt, Yt = gridvecs(aspair(dst)...)
        ϕ(Xt, Yt)
    end

    function adjΦ(k::AbstractVector; grid=meas_grids)
        Xm, Ym = gridvecs(aspair(meas_grids)...)
        @assert length(k) == length(Xm)

        cert = kernel_matrix(grid)' * k
        permutedims(reshape(cert, gridshape(grid)))
    end

    return Operators(ϕ, Φ, adjΦ, :gaussian)
end


function Base.iterate(ops::Operators, state::Int=1)
    if state == 1
        return (ops.ϕ, 2)
    elseif state == 2
        return (ops.Φ, 3)
    elseif state == 3
        return (ops.adjΦ, 4)
    else
        return nothing
    end
end