using JuMP, MosekTools, Polynomials
using DSP: conv
using LinearAlgebra: tr, pinv, diag, Hermitian

export SDP!, solve_dual, solve_primal

"""
Solve the BLASSO problem using semidefinite programming (SDP).

# Arguments
- `b`: BLASSO instance.
- `options`: Options for the solver.
"""
function SDP!(b::BLASSO, options::Dict{Symbol,<:Any}=Dict{Symbol,Any}())
    if b.d != 1 || b.ops.kind != :fourier
        throw(ArgumentError("SDP! only supports 1D problems using Fourier measurements."))
    end
    verbose = get(options, :verbose, false)
    solve_dual!(b, verbose)
    solve_primal!(b)
end

"""
Solve the dual optimisation problem.

# Arguments
- `b`: BLASSO instance.
- `verbose`: Verbosity flag.
"""
function solve_dual!(b, verbose)
    n = length(b.y)
    model = Model(Mosek.Optimizer)
    !verbose && set_silent(model)

    @variable(model, p[1:n] in ComplexPlane())
    @variable(model, Q[1:n, 1:n] in HermitianMatrixSpace())
    @constraint(model, Hermitian([Q p; p' 1]) in HermitianPSDCone())
    @constraint(model, sum(diag(Q, 0)) == 1)
    @constraint(model, [k = 1:n-1], sum(diag(Q, k)) == 0)

    # Objective (λ>0): min || y/λ - p ||_2^2 ; (λ=0): max Re⟨p,y⟩
    if b.λ == 0
        @objective(model, Max, real(p' * b.y))
    else
        @objective(model, Min, sum(abs2.(b.y / b.λ .- p)))
    end
    optimize!(model)
    b.p = value.(p) * sqrt(n) # unnormalise
end

"""
Solve the primal problem given the solution to the dual problem pλ.

# Arguments
- `b`: BLASSO instance.
"""
function solve_primal!(b)
    n = length(b.y)
    p′ = b.p / sqrt(n) # normalise
    c = -conv(p′, reverse(conj(p′)))
    c[n] += 1
    r = roots(Polynomial(c))
    rU = r[abs.(1 .- abs.(r)).<1e-6] # unit roots
    θ = sort(-angle.(rU))[1:2:end]
    x = sort(mod.(θ / (2π), 1))
    Φₓ = hcat((b.ops.ϕ(xⱼ) for xⱼ in x)...)
    s = sign.(Φₓ' * b.p)
    a = real(Φₓ \ b.y - b.λ * pinv(Φₓ' * Φₓ) * s)
    b.μ = DiscreteMeasure(x, a)
end