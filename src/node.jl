using Optim, LineSearches
using LinearAlgebra: norm, dot
using Mooncake
using DifferentiationInterface
using ForwardDiff: jacobian
using LinearAlgebra: cond

export NODE!

mutable struct MBuffer{T}
    s::Int
    d::Int
    M::Matrix{T}
end

function MBuffer(s::Integer, d::Integer; T::Type=Float64)
    n = s * (d + 1)
    MBuffer{T}(s, d,
        zeros(T, n, n),
    )
end

# RHS that reuses your MBuffer and v0
@views function rhs(z, _MB, v0, ops, y, d, s)
    a = z[1:s]
    x = reshape(z[s+1:end], d, s)  # (d × s)
    fill_M!(_MB, a, x, ops, y)        # fills MB.M in-place
    return (_MB.M \ v0)             # dz
end

# Forward Euler
function forward_euler_step!(z, h, _MB, v0, ops, y, d, s)
    k1 = rhs(z, _MB, v0, ops, y, d, s)
    z .-= h .* k1
    return z
end

# Heun's method (RK2)
function heun_step!(z, h, _MB, v0, ops, y, d, s)
    k1 = rhs(z, _MB, v0, ops, y, d, s)
    k2 = rhs(z .- h .* k1, _MB, v0, ops, y, d, s)
    z .-= (h / 2) .* (k1 .+ k2)
    return z
end

# RK4
function rk4_step!(z, h, _MB, v0, ops, y, d, s)
    k1 = rhs(z, _MB, v0, ops, y, d, s)
    k2 = rhs(z .- 0.5 * h .* k1, _MB, v0, ops, y, d, s)
    k3 = rhs(z .- 0.5 * h .* k2, _MB, v0, ops, y, d, s)
    k4 = rhs(z .- h .* k3, _MB, v0, ops, y, d, s)
    z .-= (h / 6) .* (k1 .+ 2 * k2 .+ 2 * k3 .+ k4)
    return z
end

function implicit_euler_step!(z, h, _MB, v0, ops, y, d, s; tol=1e-8, maxiter=100)
    z0 = copy(z)

    # # Prediction (explicit Euler) as initial guess
    k1 = rhs(z, _MB, v0, ops, y, d, s)
    w = z0 .- h .* k1

    # Closure to build rhs with proper element type (for Dual numbers)
    function rhs_typed(wvec)
        T = eltype(wvec)
        MBt = MBuffer(s, d; T=T)                 # fresh buffer with correct eltype
        a = wvec[1:s]
        x = reshape(wvec[s+1:end], d, s)
        fill_M!(MBt, a, x, ops, y)
        vt = convert.(T, v0)                     # promote v
        # display(MBt.M)
        # println("cond(M) ≈ ", cond(ForwardDiff.value.(MBt.M)))
        return (MBt.M \ vt)
    end

    G(wvec) = wvec .- z0 .+ h .* rhs_typed(wvec)
    # G(w) = w .- z0 .+ h .* rhs_typed(w)
    # w = copy(z)

    converged = false
    for _ in 1:maxiter
        Gw = G(w)
        if norm(Gw, Inf) < tol * (1 + norm(z0, Inf))
            z .= w
            return z
            # converged = true
            # break
        end
        J = jacobian(G, w)
        δ = J \ (-Gw)
        w .+= δ
        if norm(δ) < tol * (1 + norm(w))
            z .= w
            return z
            # converged = true
            # break
        end
    end
    # if !converged
    #     @warn "Implicit Euler step did not converge within $maxiter iterations."
    # end
    z .= w
    return z
end

function update_μ!(μ, z, d, s)
    μ.a .= z[1:s]
    flat_x = z[s+1:end]
    if d == 1
        μ.x .= flat_x
    else
        xmat = reshape(flat_x, s, d)'
        @inbounds for i in 1:d
            μ.x[i] .= xmat[i, :]
        end
    end
end

"""
Assemble the BLASSO path ODE block matrix.

Arguments:
- MB
- a
- x
- ϕ
- y
- xref :: reference location for building K(Δ) = ⟨ϕ(xref+Δ), ϕ(xref)⟩  (default zeros(d))

Blocks (s spikes, spatial dimension d):
  M_aa (s×s), M_ax (s×(s d)), M_xa ((s d)×s), M_xx ((s d)×(s d))

Convention: M(a,x) * [ȧ; ẋ] = [-v; 0]
with
  M_aa[k,j]   = K(xk - xj)
  M_ax[k,(j)] =  j==k ? ∑_ℓ a_ℓ ∇K(xk - xℓ)^T - ∇f(xk)^T :  -a_j ∇K(xk - xj)^T
  M_xa[(k),j] =  ∇K(xk - xj)
  M_xx[(k),(j)] =  j==k ? ∑_ℓ a_ℓ ∇²K(xk - xℓ) - ∇²f(xk) :  -a_j ∇²K(xk - xj)
where g(x)=⟨φ(x),y⟩.
"""
function fill_M!(MB, a, x, ops, y)
    s, d = MB.s, MB.d
    fill!(MB.M, 0)

    if isa(x, AbstractVector) && (eltype(x) <: AbstractVector)
        x = hcat(x...)'
    elseif isa(x, AbstractVector) && !(eltype(x) <: AbstractVector)
        x = reshape(x, 1, :)
    end
    @assert size(x, 1) == d "x must be a (d×s) position matrix"

    M_aa = @view MB.M[1:s, 1:s]
    M_ax = @view MB.M[1:s, s+1:end]
    M_xa = @view MB.M[s+1:end, 1:s]
    M_xx = @view MB.M[s+1:end, s+1:end]

    cϕ, ∇cϕ, Δcϕ = ops.cϕ, ops.∇cϕ, ops.Δcϕ
    g(z::Real) = dot(ops.ϕ(z), y)
    ∇g(z::Real) = dot(ops.∇ϕ(z), y)
    Δg(z::Real) = dot(ops.Δϕ(z), y)

    @inbounds for k in 1:s
        xk = @view x[:, k]

        # g_k = Σ_l a_l ∇K(xk - xl) - ∇g(xk)
        ∇K_jk = zeros(eltype(a), d)
        for l in 1:s
            Δkl = xk .- @view(x[:, l])
            ∇K_jk .+= a[l] .* ∇cϕ(Δkl[1])
        end
        ∇K_jk .-= ∇g(xk[1])

        for j in 1:s
            Δkj = xk .- @view(x[:, j])
            M_aa[k, j] = cϕ(Δkj[1])

            max_blk = @view M_ax[k, (j-1)*d+1:j*d]
            if j == k
                @. max_blk = ∇K_jk
            else
                max_blk .= -a[j] .* ∇cϕ(Δkj[1])
            end

            mxa_col = @view M_xa[(k-1)*d+1:k*d, j]
            mxa_col .= ∇cϕ(Δkj[1])
            mxx_blk = @view M_xx[(k-1)*d+1:k*d, (j-1)*d+1:j*d]
            if j == k
                fill!(mxx_blk, 0)
                for l in 1:s
                    Δkl = xk .- @view(x[:, l])
                    mxx_blk .+= a[l] .* Δcϕ(Δkl[1])
                end
                mxx_blk .-= Δg(xk[1])
            else
                mxx_blk .= -a[j] .* Δcϕ(Δkj[1])
            end
        end
    end
    return nothing
end

"""
Solve the BLASSO problem using by solving an ODE numerically to approximate the regularisation path.

# Arguments
- `b::BLASSO`: BLASSO problem to solve.
- `options::Dict()`: Options for the solver. Relevant keys:
    - `:maxits::Int` (default: `50`): Maximum number of iterations.
    - `:progress::Bool` (default: `true`): Whether to show progress bar.
    - `:base_solver::Symbol` (default: `:SFW`): Underlying solver to use.
    - `:verbose::Bool` (default: `false`): Whether to print debugging information.
    - `:store_reg_path::Bool` (default: `false`): Whether to store the regularisation path.
    - `:kink_tol::Real` (default: `1e-4`): Tolerance for detecting kinks in the regularisation path.
    - `:τδ::Real` (required): Noise level x τ

# Returns
- `b::BLASSO`: BLASSO problem with the recovered measure in the `μ` field. If `:SDP` is used, the dual solution is also stored in the `p` field.
"""
function NODE!(b::BLASSO,
    options::Dict{Symbol,<:Any}=Dict{Symbol,Any}())

    if b.ops.kind != :gaussian
        throw(ArgumentError("NODE! only supports Gaussian convolution measurements."))
    end

    maxits = get(options, :maxits, 50)
    options[:maxits] = get(options, :inner_maxits, 50)
    progress = get(options, :progress, true)
    options[:progress] = get(options, :progress, false)
    base_solver = get(options, :base_solver, :SFW)
    verbose = get(options, :verbose, false)
    store_reg_path = get(options, :store_reg_path, false)
    η_tol = get(options, :η_tol, 1e-5)
    λ₀_opt = get(options, :λ₀, Inf)
    λₖ = get(options, :λₖ, Inf)
    sₖ = get(options, :sₖ, Inf)
    h = get(options, :h, 1e-2) # step size
    τδ = get(options, :τδ, eps())

    if τδ ≤ 0
        throw(ArgumentError("τδ must be strictly greater than 0"))
    end

    b.reg_path = Dict(
        :λs => Float64[],
        :μs => DiscreteMeasure[],
    )

    xgrid = build_grid(b.domain, b.n_coarse_grid)

    y, ops = b.y, b.ops
    b.λ = min(λ₀_opt, λ₀(xgrid, ops.ϕ, y, η_tol) - eps(Float64))
    solve!(b, base_solver; options=options)
    μ_pred = deepcopy(b.μ)
    η = b.η
    r = norm(y - ops.Φₓ(b.μ...))
    z = vcat(b.μ.a, vcat(b.μ.x...))
    v0 = vcat(-sign.(b.μ.a), zeros(b.μ.s * b.d))
    MB = MBuffer(b.μ.s, b.d)

    prev_s = 0

    prog = ProgressUnknown(desc="NODE iterations: ")

    corrections = 0
    total = 0

    for _ in 1:maxits
        # println("$total")
        progress && next!(prog)

        verbose && println("λ = $(b.λ), |I| = $(sparsity(b.μ))")

        if b.λ - h < 0
            break
        elseif b.λ - h > λₖ && h < 0
            break
        elseif b.μ.s == sₖ
            break
        end
        b.λ -= h

        forward_euler_step!(z, h, MB, v0, ops, y, b.d, b.μ.s)
        # heun_step!(z, h, MB, v0, ops, y, b.d, b.μ.s)
        # rk4_step!(z, h, MB, v0, ops, y, b.d, b.μ.s)
        # implicit_euler_step!(z, h, MB, v0, ops, y, b.d, b.μ.s)
        update_μ!(μ_pred, z, b.d, b.μ.s)
        a_sign_changed = any(μ_pred.a .* b.μ.a .<= 0)
        
        η = ops.ηλ(μ_pred..., y, b.λ)
        η_max = compute_η_max(η, xgrid)[2]

        if η_max < 1 + η_tol && !a_sign_changed
            copy!(b.μ.x, μ_pred.x)
            copy!(b.μ.a, μ_pred.a)
            h = min(h * 2, 1e-3)
        else
            solve!(b, base_solver; options=options)
            z = vcat(b.μ.a, vcat(b.μ.x...))
            if b.μ.s != prev_s
                v0 = vcat(-sign.(b.μ.a), zeros(b.μ.s * b.d))
                MB = MBuffer(b.μ.s, b.d)
                μ_pred = deepcopy(b.μ)
                prev_s = b.μ.s
            end
            corrections += 1
            h = max(h / 2, 1e-10)
        end

        r = norm(y - ops.Φₓ(b.μ...))

        push!(b.reg_path[:λs], b.λ)
        push!(b.reg_path[:μs], deepcopy(b.μ))

        verbose && println("r = $r, τδ = $(τδ)")
        verbose && println("μ = $(b.μ)")

        if r < τδ
            break
        end

        total += 1
    end

    display("NODE! completed with $(corrections) corrections over $(total) iterations.")

    if !store_reg_path
        b.reg_path = nothing
    end
    return b
end