using JuMP, MosekTools, Polynomials
using DSP: conv

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
    @variable(model, X[1:n+1, 1:n+1] in HermitianPSDCone())
    @constraint(model, tr(X) == 2)
    @constraint(model, X[n+1, n+1] == 1)
    @constraint(model, X[1:n, n+1] .== p)
    @constraint(model, [j = 1:(n-1)], sum(diag(X, j)) == X[n+1-j, n+1])
    if λ == 0
        @objective(model, Max, real(p' * y))
    else
        @objective(model, Min, sum(abs2.(y / b.λ .- p)))
    end
    optimize!(model)
    b.p = value.(p)
end

"""
Solve the primal problem given the solution to the dual problem pλ.

# Arguments
- `b`: BLASSO instance.
"""
function solve_primal!(b)
    n = length(b.y)
    c = -conv(b.p, reverse(conj(b.p)))
    c[n] += 1
    P = Polynomial(c)
    r = roots(P)
    r0 = r[abs.(1 .- abs.(r)).<1e-10]
    r0 = sort(r0, by=angle)[1:2:end]
    x = angle.(r0) / (2π)
    x = sort(mod.(x, 1))
    ϕx = b.ops.ϕ(x)
    s = sign.(real.(ϕx' * b.p))
    a = real(pinv(ϕx) * b.y - b.λ * pinv(ϕx' * ϕx) * s)
    b.μ = DiscreteMeasure(x, a)
end