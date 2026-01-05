export Operators, fourier_operators_1D, fourier_operators_2D, gaussian_operators_1D, gaussian_operators_2D

struct Operators
    ϕ!::Function
    ϕ::Function   # feature kernel of the forward operator Φ
    ∇ϕ!::Function
    ∇ϕ::Function
    Δϕ!::Function
    Δϕ::Function
    cϕ::Function   # correlation kernel of the the correlation operator Φ*Φ
    ∇cϕ::Function
    Δcϕ::Function
    Φₓ!::Function
    Φₓ::Function  # discretised forward operator
    Φₓᴴ!::Function
    Φₓᴴ::Function # discretised adjoint operator
    Γₓ!::Function # Γₓ := (Φₓ, ∇ₓΦₓ)
    ηλ::Function  # dual polynomial (returns a function)
    ηᵥ::Function  # vanishing derivatives pre-certificate
    kind::Symbol
end

"""
Forward operator using Fourier kernel with frequency cutoff `fc` 
and sampling grid `plt_grid` for the adjoint.
"""
function fourier_operators_1D(fc::Int, plt_grid::AbstractVector{<:Real})::Operators
    K = collect(-fc:fc)
    nK = 2 * fc + 1
    ω = 2π
    Kω = ω .* K
    iKω = im .* Kω
    Kω2 = (ω .* K) .^ 2
    norm_cst = inv(sqrt(nK))
    @inline wrap01(Δ) = Δ - round(Δ)

    function ϕ!(buf::AbstractVector{<:Complex}, x::Real)
        @inbounds @simd for i in 1:nK
            buf[i] = norm_cst * cis(Kω[i] * x)
        end
        buf
    end

    function ϕ(x::Real)
        buf = Vector{ComplexF64}(undef, nK)
        ϕ!(buf, x)
    end

    function ∇ϕ!(buf::AbstractVector{<:Complex}, x::Real)
        @inbounds @simd for i in 1:nK
            buf[i] = iKω[i] * (norm_cst * cis(Kω[i] * x))
        end
        buf
    end

    function ∇ϕ(x::Real)
        buf = Vector{ComplexF64}(undef, nK)
        ∇ϕ!(buf, x)
    end

    function Δϕ!(buf::AbstractVector{<:Complex}, x::Real)
        @inbounds @simd for i in 1:nK
            buf[i] = -Kω2[i] * (norm_cst * cis(Kω[i] * x))
        end
        buf
    end

    function Δϕ(x::Real)
        buf = Vector{ComplexF64}(undef, nK)
        Δϕ!(buf, x)
    end

    function cϕ(Δ::Real)
        Δw = wrap01(Δ)
        s = zero(ComplexF64)
        @inbounds @simd for i in 1:nK
            s += cis(Kω[i] * Δw)
        end
        s / nK
    end
    cϕ(x₁::Real, x₂::Real) = cϕ(wrap01(x₂ - x₁))

    function ∇cϕ(Δ::Real)
        Δw = wrap01(Δ)
        s = zero(ComplexF64)
        @inbounds @simd for i in 1:nK
            s += K[i] * cis(Kω[i] * Δw)
        end
        (im * ω) * (s / nK)
    end
    ∇cϕ(x₁::Real, x₂::Real) = ∇cϕ(wrap01(x₁ - x₂)) # w.r.t. first arg; for second use -(…).

    function Δcϕ(Δ::Real)
        Δw = wrap01(Δ)
        s = zero(ComplexF64)
        @inbounds @simd for i in 1:nK
            s += (K[i]^2) * cis(Kω[i] * Δw)
        end
        -(ω^2) * (s / nK)
    end
    Δcϕ(x₁::Real, x₂::Real) = Δcϕ(wrap01(x₁ - x₂))

    function Φₓ!(out::AbstractVector{<:Complex}, x::AbstractVector{<:Real}, a::AbstractVector{<:Real})
        @assert length(x) == length(a)
        fill!(out, 0)
        @inbounds for j in eachindex(a)
            xj = x[j]
            α = a[j]
            @simd for i in 1:nK
                out[i] += α * (norm_cst * cis(Kω[i] * xj))
            end
        end
        out
    end
    function Φₓ(x::AbstractVector{<:Real}, a::AbstractVector{<:Real})
        out = zeros(ComplexF64, nK)
        Φₓ!(out, x, a)
    end

    function Φₓᴴ!(out::AbstractVector{<:Real}, k::AbstractVector{<:Complex}; grid::AbstractVector{<:Real}=plt_grid)
        M = length(grid)
        @inbounds for i in 1:M
            gi = grid[i]
            s = zero(ComplexF64)
            @simd for m in 1:nK
                s += (norm_cst * cis(-Kω[m] * gi)) * k[m]
            end
            out[i] = real(s)
        end
        out
    end

    function Φₓᴴ(k::AbstractVector{<:Complex}; grid::AbstractVector{<:Real}=plt_grid)
        T = promote_type(real(eltype(k)), eltype(grid))
        out = zeros(T, length(grid))
        Φₓᴴ!(out, k; grid=grid)
    end

    function Γₓ!(out::AbstractVector{<:Complex}, x::AbstractVector{<:Real}, a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
        @assert length(x) == length(a) == length(b)
        fill!(out, 0)
        @inbounds for j in eachindex(a)
            xj = x[j]
            α = a[j]
            β = b[j]
            @simd for i in 1:nK
                φ = norm_cst * cis(Kω[i] * xj)
                out[i] += φ * (α + β * iKω[i])      # a*ϕ + b*∂ϕ
            end
        end
        out
    end

    function ηλ(x::AbstractVector{<:Real}, a::AbstractVector{<:Real}, y::AbstractVector{<:Complex}, λ::Real)
        pλ = (y - Φₓ(x, a)) / λ
        grid -> Φₓᴴ(pλ; grid=grid)
    end

    function ηᵥ(x::AbstractVector{<:Real}, a::AbstractVector{<:Real})
        Nₓ = length(x)
        b = zeros(ComplexF64, 2Nₓ)
        b[1:Nₓ] .= sign.(a)   # [sign(a); 0]

        G = zeros(ComplexF64, 2Nₓ, 2Nₓ)
        @inbounds for i in 1:Nₓ, j in 1:Nₓ
            Δ = wrap01(x[j] - x[i])        # Δ = x_j - x_i
            A = cϕ(Δ)                      # ⟨ϕ_i, ϕ_j⟩
            B = ∇cϕ(Δ)                     # ⟨ϕ_i, ∂ϕ_j⟩ = ∂_{x_j} c
            C = -Δcϕ(Δ)                    # ⟨∂ϕ_i, ∂ϕ_j⟩
            G[i, j] = A
            G[i, Nₓ+j] = B
            G[Nₓ+i, j] = -B         # ∂_{x_i} c = -∂_{x_j} c
            G[Nₓ+i, Nₓ+j] = C
        end

        λ = G \ b
        pᵥ = zeros(ComplexF64, nK)

        Γₓ!(pᵥ, x, λ[1:Nₓ] .|> real, λ[Nₓ+1:end] .|> real)
        grid -> Φₓᴴ(pᵥ; grid=grid)
    end

    return Operators(ϕ!, ϕ, ∇ϕ!, ∇ϕ, Δϕ!, Δϕ, cϕ, ∇cϕ, Δcϕ, Φₓ!, Φₓ, Φₓᴴ!, Φₓᴴ, Γₓ!, ηλ, ηᵥ, :fourier)
end

"""
Forward operator using Fourier kernel with frequency cutoff `fc` 
and sampling grids `plt_grids` for the adjoint.
"""
function fourier_operators_2D(fc::Int, plt_grids::AbstractVector{<:AbstractArray{<:Real}})::Operators
    K1 = collect(-fc:fc)
    K2 = collect(-fc:fc)
    nK = 2fc + 1                  # per-axis band size
    nF = nK * nK                  # total Fourier coeffs
    ω = 2π
    norm_cst = inv(nK)            # = 1/sqrt(nF) → unit-norm atoms in C^{nF}

    @inline aspair(g) = isa(g, Tuple) ? g : (g[1], g[2])
    @inline gridvecs(h₁::AbstractMatrix, h₂::AbstractMatrix) = (vec(h₁), vec(h₂))
    @inline gridvecs(h₁::AbstractVector, h₂::AbstractVector) = (h₁, h₂)
    @inline gridshape(g) = isa(first(aspair(g)), AbstractMatrix) ? size(first(aspair(g))) :
                           (length(aspair(g)[1]), length(aspair(g)[2]))
    @inline wrap01(Δ) = Δ - round(Δ)  # torus [0,1)

    function ϕ!(buf::AbstractVector{<:Complex}, x₁::Real, x₂::Real)
        @inbounds begin
            idx = 1
            for j in 1:nK
                κ₂ = K2[j]
                @simd for i in 1:nK
                    κ₁ = K1[i]
                    idx = (j - 1) * nK + i
                    buf[idx] = norm_cst * cis(ω * (κ₁ * x₁ + κ₂ * x₂))
                end
            end
        end
        buf
    end
    ϕ!(buf::AbstractVector{<:Complex}, x::AbstractVector{<:Real}) = (@assert length(x) == 2; ϕ!(buf, x[1], x[2]))

    function ϕ(x₁::Real, x₂::Real)
        buf = Vector{ComplexF64}(undef, nF)
        ϕ!(buf, x₁, x₂)
    end
    ϕ(x::AbstractVector{<:Real}) = (@assert length(x) == 2; ϕ(x[1], x[2]))

    function ∇ϕ!(gx::AbstractVector{<:Complex}, gy::AbstractVector{<:Complex}, x₁::Real, x₂::Real)
        @inbounds begin
            for j in 1:nK
                κ₂ = K2[j]
                @simd for i in 1:nK
                    κ₁ = K1[i]
                    idx = (j - 1) * nK + i
                    φ = norm_cst * cis(ω * (κ₁ * x₁ + κ₂ * x₂))
                    gx[idx] = (im * ω * κ₁) * φ
                    gy[idx] = (im * ω * κ₂) * φ
                end
            end
        end
        return gx, gy
    end
    ∇ϕ!(gx::AbstractVector{<:Complex}, gy::AbstractVector{<:Complex}, x::AbstractVector{<:Real}) =
        (@assert length(x) == 2; ∇ϕ!(gx, gy, x[1], x[2]))

    function ∇ϕ(x₁::Real, x₂::Real)
        gx = Vector{ComplexF64}(undef, nF)
        gy = Vector{ComplexF64}(undef, nF)
        ∇ϕ!(gx, gy, x₁, x₂)
    end
    ∇ϕ(x::AbstractVector{<:Real}) = (@assert length(x) == 2; ∇ϕ(x[1], x[2]))

    function Δϕ!(buf::AbstractVector{<:Complex}, x₁::Real, x₂::Real)
        @inbounds begin
            for j in 1:nK
                κ₂ = K2[j]
                @simd for i in 1:nK
                    κ₁ = K1[i]
                    idx = (j - 1) * nK + i
                    φ = norm_cst * cis(ω * (κ₁ * x₁ + κ₂ * x₂))
                    buf[idx] = -(ω^2) * (κ₁^2 + κ₂^2) * φ
                end
            end
        end
        buf
    end
    Δϕ!(buf::AbstractVector{<:Complex}, x::AbstractVector{<:Real}) = (@assert length(x) == 2; Δϕ!(buf, x[1], x[2]))
    function Δϕ(x₁::Real, x₂::Real)
        buf = Vector{ComplexF64}(undef, nF)
        Δϕ!(buf, x₁, x₂)
    end
    Δϕ(x::AbstractVector{<:Real}) = (@assert length(x) == 2; Δϕ(x[1], x[2]))

    @inline function _D(Δ::Real)
        Δw = wrap01(Δ)
        s = sinpi(Δw)
        if abs(s) < 1e-12
            return 1.0                         # lim_{Δ→0} D(Δ) = 1
        else
            return sinpi(nK * Δw) / (nK * s)
        end
    end

    @inline function _Dp(Δ::Real)
        Δw = wrap01(Δ)
        s = sinpi(Δw)
        if abs(s) < 1e-8
            return 0.0                         # odd derivative, zero at integers
        end
        c = cospi(Δw)
        S = sinpi(nK * Δw)
        C = cospi(nK * Δw)
        return π * (nK * C * s - S * c) / (nK * s^2)
    end

    @inline function _Dpp(Δ::Real)
        Δw = wrap01(Δ)
        s = sinpi(Δw)
        if abs(s) < 1e-6
            return (π^2) * (1 - nK^2) / 3      # series: D(Δ) ≈ 1 + ((1-nK^2)π^2/6)Δ^2
        end
        c = cospi(Δw)
        S = sinpi(nK * Δw)
        C = cospi(nK * Δw)
        F = nK * C * s - S * c
        return (π^2 / nK) * ((1 - nK^2) * (S / s) - 2 * F * c / (s^3))
    end

    @inline cϕ(Δ₁::Real, Δ₂::Real) = _D(Δ₁) * _D(Δ₂)

    @inline function cϕ(x₁::Real, x₂::Real, y₁::Real, y₂::Real)
        cϕ(wrap01(y₁ - x₁), wrap01(y₂ - x₂))
    end
    cϕ(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}) = cϕ(x[1], x[2], y[1], y[2])

    @inline function ∇cϕ(Δ₁::Real, Δ₂::Real)
        (_Dp(Δ₁) * _D(Δ₂), _D(Δ₁) * _Dp(Δ₂))
    end
    @inline function ∇cϕ(x₁::Real, x₂::Real, y₁::Real, y₂::Real)
        ∇cϕ(wrap01(y₁ - x₁), wrap01(y₂ - x₂))
    end

    @inline function Δcϕ(Δ₁::Real, Δ₂::Real)
        _Dpp(Δ₁) * _D(Δ₂) + _D(Δ₁) * _Dpp(Δ₂)
    end
    @inline function Δcϕ(x₁::Real, x₂::Real, y₁::Real, y₂::Real)
        Δcϕ(wrap01(y₁ - x₁), wrap01(y₂ - x₂))
    end

    function Φₓ!(out::AbstractVector{<:Complex},
        x₁::AbstractVector{<:Real}, x₂::AbstractVector{<:Real},
        a::AbstractVector{<:Real})
        @assert length(x₁) == length(x₂) == length(a)
        fill!(out, 0)
        @inbounds for jx in eachindex(a)
            x1 = x₁[jx]
            x2 = x₂[jx]
            α = a[jx]
            for j in 1:nK
                κ₂ = K2[j]
                @simd for i in 1:nK
                    κ₁ = K1[i]
                    idx = (j - 1) * nK + i
                    out[idx] += α * (norm_cst * cis(ω * (κ₁ * x1 + κ₂ * x2)))
                end
            end
        end
        out
    end
    Φₓ!(out::AbstractVector{<:Complex}, x::AbstractVector{<:AbstractVector{<:Real}}, a::AbstractVector{<:Real}) =
        (@assert length(x[1]) == length(x[2]); Φₓ!(out, x[1], x[2], a))

    function Φₓ(x₁::AbstractVector{<:Real}, x₂::AbstractVector{<:Real}, a::AbstractVector{<:Real})
        out = zeros(ComplexF64, nF)
        Φₓ!(out, x₁, x₂, a)
    end
    Φₓ(x::AbstractVector{<:AbstractVector{<:Real}}, a::AbstractVector{<:Real}) =
        (@assert length(x[1]) == length(x[2]); Φₓ(x[1], x[2], a))

    function Φₓᴴ!(out::AbstractArray{<:Real}, k::AbstractVector{<:Complex}; grid=plt_grids)
        Xg, Yg = aspair(grid)
        gx, gy = gridvecs(Xg, Yg)
        M = length(gx)
        @assert length(gy) == M
        @assert length(k) == nF
        @inbounds for t in 1:M
            x = gx[t]
            y = gy[t]
            s = zero(ComplexF64)
            for j in 1:nK
                κ₂ = K2[j]
                @simd for i in 1:nK
                    κ₁ = K1[i]
                    idx = (j - 1) * nK + i
                    s += (norm_cst * cis(-ω * (κ₁ * x + κ₂ * y))) * k[idx]
                end
            end
            out[t] = real(s)
        end
        isa(Xg, AbstractMatrix) ? reshape(out, gridshape(grid)) : out
    end
    function Φₓᴴ(k::AbstractVector{<:Complex}; grid=plt_grids)
        out = similar(first(aspair(grid)))
        Φₓᴴ!(out, k; grid=grid)
    end

    function Γₓ!(out::AbstractVector{<:Complex},
        x₁::AbstractVector{<:Real}, x₂::AbstractVector{<:Real},
        a::AbstractVector{<:Real}, b₁::AbstractVector{<:Real}, b₂::AbstractVector{<:Real})
        @assert length(x₁) == length(x₂) == length(a) == length(b₁) == length(b₂)
        fill!(out, 0)
        @inbounds for jx in eachindex(a)
            x1 = x₁[jx]
            x2 = x₂[jx]
            α = a[jx]
            β1 = b₁[jx]
            β2 = b₂[jx]
            for j in 1:nK
                κ₂ = K2[j]
                @simd for i in 1:nK
                    κ₁ = K1[i]
                    idx = (j - 1) * nK + i
                    φ = norm_cst * cis(ω * (κ₁ * x1 + κ₂ * x2))
                    out[idx] += φ * (α + (im * ω * κ₁) * β1 + (im * ω * κ₂) * β2)
                end
            end
        end
        out
    end

    function ηλ(x₁::AbstractVector{<:Real}, x₂::AbstractVector{<:Real},
        a::AbstractVector{<:Real}, y::AbstractVector{<:Complex}, λ::Real)
        pλ = (y - Φₓ(x₁, x₂, a)) / λ
        grid -> Φₓᴴ(pλ; grid=grid)
    end
    ηλ(x::AbstractVector{<:AbstractVector{<:Real}}, a::AbstractVector{<:Real},
        y::AbstractVector{<:Complex}, λ::Real) =
        (@assert length(x[1]) == length(x[2]); ηλ(x[1], x[2], a, y, λ))

    function ηᵥ(x₁::AbstractVector{<:Real}, x₂::AbstractVector{<:Real}, a::AbstractVector{<:Real})
        Nₓ = length(x₁)
        @assert length(x₂) == Nₓ == length(a)
        T = ComplexF64

        # rhs: [sign(a); 0; 0]
        b = zeros(T, 3Nₓ)
        b[1:Nₓ] .= sign.(a)

        # analytic Gram via cϕ, ∇cϕ, Δcϕ  (blocks: a, b₁, b₂)
        G = zeros(T, 3Nₓ, 3Nₓ)
        @inbounds for i in 1:Nₓ, j in 1:Nₓ
            Δ₁ = wrap01(x₁[j] - x₁[i])
            Δ₂ = wrap01(x₂[j] - x₂[i])

            c = _D(Δ₁) * _D(Δ₂)
            g1, g2 = _Dp(Δ₁) * _D(Δ₂), _D(Δ₁) * _Dp(Δ₂)
            C11 = -_Dpp(Δ₁) * _D(Δ₂)
            C22 = -_D(Δ₁) * _Dpp(Δ₂)
            C12 = -_Dp(Δ₁) * _Dp(Δ₂)

            # fill symmetric 3×3 block
            G[i, j] = c
            G[i, Nₓ+j] = g1
            G[i, 2Nₓ+j] = g2

            G[Nₓ+i, j] = -g1   # ∂_{x_i} c = -∂_{x_j} c
            G[Nₓ+i, Nₓ+j] = C11
            G[Nₓ+i, 2Nₓ+j] = C12

            G[2Nₓ+i, j] = -g2
            G[2Nₓ+i, Nₓ+j] = C12
            G[2Nₓ+i, 2Nₓ+j] = C22
        end

        λ = Symmetric(G) \ b
        pᵥ = zeros(ComplexF64, nF)
        λa = @view λ[1:Nₓ]
        λb1 = @view λ[Nₓ+1:2Nₓ]
        λb2 = @view λ[2Nₓ+1:3Nₓ]
        Γₓ!(pᵥ, x₁, x₂, real.(λa), real.(λb1), real.(λb2))
        grid -> Φₓᴴ(pᵥ; grid=grid)
    end
    ηᵥ(x::AbstractVector{<:AbstractVector{<:Real}}, a::AbstractVector{<:Real}) =
        (@assert length(x[1]) == length(x[2]); ηᵥ(x[1], x[2], a))

    return Operators(ϕ!, ϕ, ∇ϕ!, ∇ϕ, Δϕ!, Δϕ, cϕ, ∇cϕ, Δcϕ,
        Φₓ!, Φₓ, Φₓᴴ!, Φₓᴴ, Γₓ!, ηλ, ηᵥ, :fourier)
end


"""
Forward operator using a discretised Gaussian convolution kernel with standard deviation `σ` on grid `meas_grid`.
"""
function gaussian_operators_1D(σ::Real, meas_grid::AbstractVector{<:Real})::Operators
    N = length(meas_grid)
    invσ2 = inv(σ^2)
    inv2σ2 = inv(2σ^2)
    inv4σ2 = inv(4σ^2)
    inv4σ4 = inv(4σ^4)
    norm_cst = sqrt(inv(√π * σ))
    sqrth = sqrt(step(meas_grid))

    @inline wrap01(Δ) = Δ - round(Δ)

    function ϕ!(buf::AbstractVector{<:Real}, x::Real)
        @inbounds @simd for i in 1:N
            Δ = wrap01(meas_grid[i] - x)
            buf[i] = sqrth * norm_cst * exp(-Δ^2 * inv2σ2)
        end
        buf
    end

    function ϕ(x::Real)
        T = promote_type(typeof(x), eltype(meas_grid))
        buf = similar(meas_grid, T)
        ϕ!(buf, x)
    end

    function ∇ϕ!(buf::AbstractVector{<:Real}, x::Real)
        @inbounds @simd for i in 1:N
            Δ = wrap01(meas_grid[i] - x)
            buf[i] = sqrth * norm_cst * (Δ * invσ2) * exp(-Δ^2 * inv2σ2)
        end
        buf
    end

    function ∇ϕ(x::Real)
        T = promote_type(typeof(x), eltype(meas_grid))
        buf = similar(meas_grid, T)
        ∇ϕ!(buf, x)
    end

    function Δϕ!(buf::AbstractVector{<:Real}, x::Real)
        @inbounds @simd for i in 1:N
            Δ = wrap01(meas_grid[i] - x)
            buf[i] = sqrth * norm_cst * ((Δ^2 * (invσ2^2)) - invσ2) * exp(-Δ^2 * inv2σ2)
        end
        buf
    end

    function Δϕ(x::Real)
        T = promote_type(typeof(x), eltype(meas_grid))
        buf = similar(meas_grid, T)
        Δϕ!(buf, x)
    end

    function cϕ(Δ::Real)
        exp(-Δ^2 * inv4σ2)
    end

    function cϕ(x₁::Real, x₂::Real)
        cϕ(wrap01(x₁ - x₂)) # Δ = wrap01(x₁ - x₂)
    end

    function ∇cϕ(Δ::Real)
        -(Δ * inv2σ2) * exp(-Δ^2 * inv4σ2)
    end

    function ∇cϕ(x₁::Real, x₂::Real)
        ∇cϕ(wrap01(x₁ - x₂))
    end

    function Δcϕ(Δ::Real)
        ((Δ^2) * (invσ2^2) / 4 - inv2σ2) * exp(-Δ^2 * inv4σ2)
    end

    function Δcϕ(x₁::Real, x₂::Real)
        Δcϕ(wrap01(x₁ - x₂))
    end

    function Φₓ!(out::AbstractVector{<:Real}, x::AbstractVector{<:Real}, a::AbstractVector{<:Real})
        tmp = similar(out)
        fill!(out, 0)
        @inbounds for j in eachindex(a)
            ϕ!(tmp, x[j])
            @. out += a[j] * tmp
        end
        out
    end

    function Φₓ(x::AbstractVector{<:Real}, a::AbstractVector{<:Real})
        out = zeros(promote_type(eltype(x), eltype(a), eltype(meas_grid)), N)
        Φₓ!(out, x, a)
    end

    function Φₓᴴ!(out::AbstractVector{<:Real}, k::AbstractVector{<:Real}; grid::AbstractVector{<:Real}=meas_grid)
        M = length(grid)
        tmp = similar(k, eltype(grid))
        @inbounds for i in 1:M
            ϕ!(tmp, grid[i])
            out[i] = dot(tmp, k)
        end
        out
    end

    function Φₓᴴ(k::AbstractVector{<:Real}; grid::AbstractVector{<:Real}=meas_grid)
        out = similar(grid)
        Φₓᴴ!(out, k; grid=grid)
    end

    function Γₓ!(out::AbstractVector{<:Real}, x::AbstractVector{<:Real}, a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
        @assert length(x) == length(a) == length(b)
        fill!(out, 0)
        @inbounds for j in eachindex(a)
            xj = x[j]
            α = a[j]
            β = b[j] * invσ2
            @simd for i in eachindex(out)  # i iterates measurement grid
                Δ = wrap01(meas_grid[i] - xj)
                φ = sqrth * norm_cst * exp(-(Δ * Δ) * inv2σ2)
                out[i] += φ * (α + β * Δ)     # a*ϕ + b*∂ϕ
            end
        end
        return out
    end

    function ηλ(x::AbstractVector{<:Real}, a::AbstractVector{<:Real}, y::AbstractVector{<:Real}, λ::Real)
        pλ = (y - Φₓ(x, a)) / λ
        grid -> Φₓᴴ(pλ; grid=grid)
    end

    function ηᵥ(x::AbstractVector{<:Real}, a::AbstractVector{<:Real})
        Nₓ = length(x)
        @assert length(a) == Nₓ
        T = promote_type(eltype(x), eltype(a))

        A = [cϕ(x[i], x[j]) for i = 1:Nₓ, j = 1:Nₓ]
        B = [-∇cϕ(x[i], x[j]) for i = 1:Nₓ, j = 1:Nₓ]
        C = [-Δcϕ(x[i], x[j]) for i = 1:Nₓ, j = 1:Nₓ]

        G = [A B; B' C]
        rhs = vcat(sign.(a), zeros(T, Nₓ))
        λ = G \ rhs
        α = @view λ[1:Nₓ]
        β = @view λ[Nₓ+1:2Nₓ]

        pᵥ = zeros(T, N)
        Γₓ!(pᵥ, x, α, β)
        return grid -> Φₓᴴ(pᵥ; grid=grid)
    end

    return Operators(ϕ!, ϕ, ∇ϕ!, ∇ϕ, Δϕ!, Δϕ, cϕ, ∇cϕ, Δcϕ, Φₓ!, Φₓ, Φₓᴴ!, Φₓᴴ, Γₓ!, ηλ, ηᵥ, :gaussian)
end

"""
Forward operator using a discretised Gaussian convolution kernel with standard deviation `σ`, sampled on grids `meas_grids`.
"""
function gaussian_operators_2D(σ::Real, meas_grids::AbstractVector{<:AbstractArray{<:Real}})::Operators
    @inline aspair(g) = isa(g, Tuple) ? g : (g[1], g[2])
    @inline gridvecs(h₁::AbstractMatrix, h₂::AbstractMatrix) = (vec(h₁), vec(h₂))
    @inline gridvecs(h₁::AbstractVector, h₂::AbstractVector) = (h₁, h₂)
    @inline gridshape(g) = isa(first(aspair(g)), AbstractMatrix) ? size(first(aspair(g))) :
                           (length(aspair(g)[1]), length(aspair(g)[2]))

    function gridsteps(g₁, g₂)
        if isa(g₁, AbstractMatrix) && isa(g₂, AbstractMatrix)
            hx = abs(g₁[min(end, 2), 1] - g₁[1, 1])
            hy = abs(g₂[1, min(end, 2)] - g₂[1, 1])
        else
            hx = step(g₁)
            hy = step(g₂)
        end
        return (hx, hy)
    end

    G₁, G₂ = aspair(meas_grids)
    g₁, g₂ = gridvecs(G₁, G₂)
    N = length(g₁)
    @assert length(g₂) == N

    invσ2 = inv(σ^2)
    inv2σ2 = inv(2σ^2)
    inv4σ2 = inv(4σ^2)
    inv4σ4 = inv(4σ^4)
    norm_cst = sqrt(inv(π * σ^2))
    hx, hy = gridsteps(G₁, G₂)
    sqrth = sqrt(hx * hy)

    @inline wrap01(Δ) = Δ - round(Δ)

    function ϕ!(buf::AbstractVector{<:Real}, x₁::Real, x₂::Real)
        @inbounds @simd for i in 1:N
            Δ₁ = wrap01(g₁[i] - x₁)
            Δ₂ = wrap01(g₂[i] - x₂)
            buf[i] = sqrth * norm_cst * exp(-(Δ₁ * Δ₁ + Δ₂ * Δ₂) * inv2σ2)
        end
        buf
    end
    ϕ!(buf, x::AbstractVector{<:Real}) = (@assert length(x) == 2; ϕ!(buf, x[1], x[2]))

    function ϕ(x₁::Real, x₂::Real)
        buf = similar(g₁)
        ϕ!(buf, x₁, x₂)
    end
    ϕ(x::AbstractVector{<:Real}) = (@assert length(x) == 2; ϕ(x[1], x[2]))

    function ∇ϕ₁₂!(buf₁::AbstractVector{<:Real}, buf₂::AbstractVector{<:Real}, x₁::Real, x₂::Real)
        @inbounds @simd for i in 1:N
            Δ₁ = wrap01(g₁[i] - x₁)
            Δ₂ = wrap01(g₂[i] - x₂)
            φ = sqrth * norm_cst * exp(-(Δ₁ * Δ₁ + Δ₂ * Δ₂) * inv2σ2)
            buf₁[i] = (Δ₁ * invσ2) * φ
            buf₂[i] = (Δ₂ * invσ2) * φ
        end
        return buf₁, buf₂
    end
    ∇ϕ!(buf₁::AbstractVector{<:Real}, buf₂::AbstractVector{<:Real}, x₁::Real, x₂::Real) = ∇ϕ₁₂!(buf₁, buf₂, x₁, x₂)

    function ∇ϕ(x₁::Real, x₂::Real)
        gx = similar(g₁)
        gy = similar(g₂)
        ∇ϕ!(gx, gy, x₁, x₂)
        return gx, gy
    end
    ∇ϕ(x::AbstractVector{<:Real}) = (@assert length(x) == 2; ∇ϕ(x[1], x[2]))

    function Δϕ!(buf::AbstractVector{<:Real}, x₁::Real, x₂::Real)
        @inbounds @simd for i in 1:N
            Δ₁ = wrap01(g₁[i] - x₁)
            Δ₂ = wrap01(g₂[i] - x₂)
            r2 = Δ₁ * Δ₁ + Δ₂ * Δ₂
            φ = sqrth * norm_cst * exp(-r2 * (0.5 * invσ2))
            buf[i] = ((r2 * invσ2^2) - (2 * invσ2)) * φ
        end
        buf
    end
    function Δϕ(x₁::Real, x₂::Real)
        buf = similar(g₁)
        Δϕ!(buf, x₁, x₂)
    end
    Δϕ(x::AbstractVector{<:Real}) = (@assert length(x) == 2; Δϕ(x[1], x[2]))

    cϕ(Δ₁::Real, Δ₂::Real) = exp(-(Δ₁ * Δ₁ + Δ₂ * Δ₂) * inv4σ2)
    function cϕ(x₁::Real, x₂::Real, y₁::Real, y₂::Real)
        Δ₁ = wrap01(x₁ - y₁)
        Δ₂ = wrap01(x₂ - y₂)
        cϕ(Δ₁, Δ₂)
    end
    cϕ(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}) = cϕ(x[1], x[2], y[1], y[2])

    # note that ∇₁cϕ = -∇₂cϕ
    function ∇cϕ(Δ₁::Real, Δ₂::Real)
        c = exp(-(Δ₁ * Δ₁ + Δ₂ * Δ₂) * inv4σ2)
        return (-Δ₁ * inv2σ2 * c, -Δ₂ * inv2σ2 * c)
    end
    function ∇cϕ(x₁::Real, x₂::Real, y₁::Real, y₂::Real)
        ∇cϕ(wrap01(y₁ - x₁), wrap01(y₂ - x₂))
    end

    # note that Δ₁cϕ = -Δ₂cϕ 
    function Δcϕ(Δ₁::Real, Δ₂::Real)
        r2 = Δ₁ * Δ₁ + Δ₂ * Δ₂
        c = exp(-r2 * inv4σ2)
        return ((r2 * (invσ2^2) / 4) - (2 * inv2σ2)) * c
    end
    function Δcϕ(x₁::Real, x₂::Real, y₁::Real, y₂::Real)
        Δcϕ(wrap01(y₁ - x₁), wrap01(y₂ - x₂))
    end

    function Φₓ!(out::AbstractVector{<:Real},
        x₁::AbstractVector{<:Real}, x₂::AbstractVector{<:Real},
        a::AbstractVector{<:Real})
        @assert length(x₁) == length(x₂) == length(a)
        tmp = similar(out)
        fill!(out, 0)
        @inbounds for j in eachindex(a)
            ϕ!(tmp, x₁[j], x₂[j])
            @. out += a[j] * tmp
        end
        out
    end
    Φₓ!(out::AbstractVector{<:Real}, x::AbstractVector{<:AbstractVector{<:Real}}, a::AbstractVector{<:Real}) =
        (@assert length(x[1]) == length(x[2]); Φₓ!(out, x[1], x[2], a))

    function Φₓ(x₁::AbstractVector{<:Real}, x₂::AbstractVector{<:Real}, a::AbstractVector{<:Real})
        out = zeros(promote_type(eltype(x₁), eltype(x₂), eltype(a), eltype(g₁)), N)
        Φₓ!(out, x₁, x₂, a)
    end
    Φₓ(x::AbstractVector{<:AbstractVector{<:Real}}, a::AbstractVector{<:Real}) =
        (@assert length(x[1]) == length(x[2]); Φₓ(x[1], x[2], a))

    function Φₓᴴ!(out::AbstractArray{<:Real}, k::AbstractVector{<:Real}; grid=meas_grids)
        eg₁, eg₂ = gridvecs(aspair(grid)...) # evaluation grids
        @assert length(k) == N
        M = length(eg₁)
        tmp = similar(k, promote_type(eltype(eg₁), eltype(eg₂)))
        @inbounds for i in 1:M
            ϕ!(tmp, eg₁[i], eg₂[i])
            out[i] = dot(tmp, k)
        end
        if isa(first(aspair(grid)), AbstractMatrix)
            reshape(out, gridshape(grid))
        else
            out
        end
    end
    function Φₓᴴ(k::AbstractVector{<:Real}; grid=meas_grids)
        out = similar(first(aspair(grid)))
        Φₓᴴ!(out, k; grid=grid)
    end

    function Γₓ!(out::AbstractVector{<:Real},
        x₁::AbstractVector{<:Real}, x₂::AbstractVector{<:Real},
        a::AbstractVector{<:Real}, b₁::AbstractVector{<:Real}, b₂::AbstractVector{<:Real})
        @assert length(x₁) == length(x₂) == length(a) == length(b₁) == length(b₂)
        fill!(out, 0)

        @inbounds for j in eachindex(a)
            x1 = x₁[j]
            x2 = x₂[j]
            α = a[j]
            β1 = b₁[j] * invσ2
            β2 = b₂[j] * invσ2

            @simd for i in 1:N
                Δ₁ = wrap01(g₁[i] - x1)
                Δ₂ = wrap01(g₂[i] - x2)
                r2 = Δ₁ * Δ₁ + Δ₂ * Δ₂
                φ = sqrth * norm_cst * exp(-r2 * inv2σ2)
                out[i] += φ * (α + β1 * Δ₁ + β2 * Δ₂)
            end
        end
        return out
    end

    function ηλ(x₁::AbstractVector{<:Real}, x₂::AbstractVector{<:Real},
        a::AbstractVector{<:Real}, y::AbstractVector{<:Real}, λ::Real)
        pλ = (y - Φₓ(x₁, x₂, a)) / λ
        grid -> Φₓᴴ(pλ; grid=grid)
    end
    ηλ(x::AbstractVector{<:AbstractVector{<:Real}}, a::AbstractVector{<:Real},
        y::AbstractVector{<:Real}, λ::Real) =
        (@assert length(x[1]) == length(x[2]); ηλ(x[1], x[2], a, y, λ))

    function ηᵥ(x₁::AbstractVector{<:Real}, x₂::AbstractVector{<:Real}, a::AbstractVector{<:Real})
        Nₓ = length(x₁)
        @assert length(x₂) == Nₓ == length(a)
        T = promote_type(eltype(x₁), eltype(x₂), eltype(a))

        A = [cϕ(x₁[i], x₂[i], x₁[j], x₂[j]) for i = 1:Nₓ, j = 1:Nₓ]
        B₁ = Matrix{T}(undef, Nₓ, Nₓ)
        B₂ = Matrix{T}(undef, Nₓ, Nₓ)
        C₁₁ = Matrix{T}(undef, Nₓ, Nₓ)
        C₂₂ = Matrix{T}(undef, Nₓ, Nₓ)
        C₁₂ = Matrix{T}(undef, Nₓ, Nₓ)

        @inbounds for i in 1:Nₓ, j in 1:Nₓ
            # First derivatives
            g1, g2 = ∇cϕ(x₁[i], x₂[i], x₁[j], x₂[j])
            B₁[i, j] = -g1
            B₂[i, j] = -g2

            # Second derivatives: C = -∂²_{ΔΔ} c. Use Δ = (x_j - x_i), c = cϕ(Δ)
            Δ₁ = wrap01(x₁[j] - x₁[i])
            Δ₂ = wrap01(x₂[j] - x₂[i])
            c = cϕ(x₁[i], x₂[i], x₁[j], x₂[j])
            C₁₁[i, j] = (inv2σ2 - Δ₁ * Δ₁ * inv4σ4) * c
            C₂₂[i, j] = (inv2σ2 - Δ₂ * Δ₂ * inv4σ4) * c
            C₁₂[i, j] = (-(Δ₁ * Δ₂) * inv4σ4) * c
        end

        # Assemble G = [A B₁ B₂;-B₁ᵗ C₁₁ C₁₂;-B₂ᵗ C₁₂ C₂₂]
        G = zeros(T, 3Nₓ, 3Nₓ)
        G[1:Nₓ, 1:Nₓ] .= A
        G[1:Nₓ, Nₓ+1:2Nₓ] .= B₁
        G[1:Nₓ, 2Nₓ+1:3Nₓ] .= B₂
        G[Nₓ+1:2Nₓ, 1:Nₓ] .= -transpose(B₁)
        G[Nₓ+1:2Nₓ, Nₓ+1:2Nₓ] .= C₁₁
        G[Nₓ+1:2Nₓ, 2Nₓ+1:3Nₓ] .= C₁₂
        G[2Nₓ+1:3Nₓ, 1:Nₓ] .= -transpose(B₂)
        G[2Nₓ+1:3Nₓ, Nₓ+1:2Nₓ] .= C₁₂
        G[2Nₓ+1:3Nₓ, 2Nₓ+1:3Nₓ] .= C₂₂

        rhs = vcat(sign.(a), zeros(T, 2Nₓ))
        λ = G \ rhs
        α = @view λ[1:Nₓ]
        β₁ = @view λ[Nₓ+1:2Nₓ]
        β₂ = @view λ[2Nₓ+1:3Nₓ]

        pᵥ = zeros(T, N)
        Γₓ!(pᵥ, x₁, x₂, α, β₁, β₂)
        return grid -> Φₓᴴ(pᵥ; grid=grid)
    end
    ηᵥ(x::AbstractVector{<:AbstractVector{<:Real}}, a::AbstractVector{<:Real}) =
        (@assert length(x[1]) == length(x[2]); ηᵥ(x[1], x[2], a))

    return Operators(ϕ!, ϕ, ∇ϕ!, ∇ϕ, Δϕ!, Δϕ, cϕ, ∇cϕ, Δcϕ, Φₓ!, Φₓ, Φₓᴴ!, Φₓᴴ, Γₓ!, ηλ, ηᵥ, :gaussian)
end


function Base.iterate(ops::Operators, state::Int=1)
    if state == 1
        return (ops.ϕ, 2)
    elseif state == 2
        return (ops.Φₓ, 3)
    elseif state == 3
        return (ops.Φₓᴴ, 4)
    else
        return nothing
    end
end