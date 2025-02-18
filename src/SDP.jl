using JuMP, MosekTools, LinearAlgebra, Polynomials, LaTeXStrings, Plots, DSP

export SDP!, solve_dual, solve_primal

"""
    SDP!(blasso::BLASSO)

Solve the BLASSO problem using semidefinite programming (SDP).

# Arguments
- `blasso`: BLASSO instance.

# Returns
- `blasso`: Updated BLASSO instance with the solution.
"""
function SDP!(blasso::BLASSO)
    blasso.p = solve_dual(blasso.y, blasso.λ)
    blasso.μ = solve_primal(blasso.p, blasso.y, blasso.operators.ϕ, blasso.λ)
    return blasso
end

"""
    solve_dual(y, λ) -> Vector{Complex{Float64}}

Solve the dual optimisation problem.

# Arguments
- `y`: Observation vector.
- `λ`: Regularisation parameter.

# Returns
- `pλ`: Solution vector of the dual problem.
"""
function solve_dual(y, λ)
    n = length(y)
    model = Model(Mosek.Optimizer)
    @variable(model, p[1:n] in ComplexPlane())
    @variable(model, X[1:n+1, 1:n+1] in HermitianPSDCone())
    @constraint(model, tr(X) == 2)
    @constraint(model, X[n+1, n+1] == 1)
    @constraint(model, X[1:n, n+1] .== p)
    @constraint(model, [j = 1:(n-1)], sum(diag(X, j)) == X[n+1-j, n+1])
    if λ == 0
        @objective(model, Max, real(p' * y))
    else
        @objective(model, Min, sum(abs2.(y / λ .- p)))
    end
    optimize!(model)
    return value.(p)
end

"""
    solve_primal(pλ, y, ϕ, λ) -> DiscreteMeasure

Solve the primal problem given the solution to the dual problem pλ.

# Arguments
- `pλ`: Solution vector of the dual problem.
- `y`: Observation vector.
- `ϕ`: Fourier operator.
- `λ`: Regularisation parameter.

# Returns
- `μ`: Recovered measure.
"""
function solve_primal(pλ, y, ϕ, λ)
    n = length(y)
    c = -conv(pλ, reverse(conj(pλ)))
    c[n] += 1
    P = Polynomial(c)
    r = roots(P)
    r0 = r[abs.(1 .- abs.(r)).<1e-10]
    r0 = sort(r0, by=angle)[1:2:end]
    x = angle.(r0) / (2π)
    x = sort(mod.(x, 1))
    ϕx = ϕ(x)
    s = sign.(real.(ϕx' * pλ))
    a = real(pinv(ϕx) * y - λ * pinv(ϕx' * ϕx) * s)
    return DiscreteMeasure(x, a)
end