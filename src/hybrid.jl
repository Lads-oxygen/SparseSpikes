using Optim, LineSearches, Printf
using LinearAlgebra: norm, dot

export Hybrid!

const _TOP = "┌───────────┬────────────┬───────────────────┬───────────────────┬──────────────────────────┬─────────────┬──────────┬──────┐"
const _HEAD = "│ iteration │ λ          │ current sparsity  │ previous sparsity │ previous kink sparsity   │ direction   │ next λ   │ kink │"
const _SEP = "├───────────┼────────────┼───────────────────┼───────────────────┼──────────────────────────┼─────────────┼──────────┼──────┤"
const _BOTTOM = "└───────────┴────────────┴───────────────────┴───────────────────┴──────────────────────────┴─────────────┴──────────┴──────┘"

@inline function _row(iteration, λ, current_sparsity, previous_sparsity, previous_kink_sparsity, forward, next_λ, kink::Bool)
    dir =
        forward ? "↓" : "↑"
    kink_str = kink ? "x" : ""
    if isnan(next_λ)
        next_λ_str = "   —        "
    else
        next_λ_str = @sprintf("%8.4f", next_λ)
    end
    return @sprintf("│ %9d │ %10.4f │ %17d │ %17d │ %24d │  %8s   │ %s │  %3s │",
        iteration, λ, current_sparsity, previous_sparsity, previous_kink_sparsity, dir, next_λ_str, kink_str)
end

prev_kink_μ(reg_path::Dict) = isempty(reg_path[:μs]) ? nothing : reg_path[:μs][end]

function next_kink!(b, ϕ, xgrid, η_tol)
    τ_seed = 0.1
    x = b.μ.x
    a = b.μ.a
    ϕₓ = ϕ(x)
    α = ϕₓ \ b.y
    β = (ϕₓ' * ϕₓ) \ sign.(a)
    u = b.y .- ϕₓ * α
    v = ϕₓ * β

    λₑ(xᵢ) = begin
        ϕᵢ = ϕ(xᵢ)
        g = dot(ϕᵢ, u)
        h = dot(ϕᵢ, v)
        tol = 1e-10
        new_λ = 0.0
        for σ in (1.0, -1.0)                  # try both branches
            denom = (1 + η_tol) - σ * h
            # Feasible without dividing: 0<λ<λk  ⇔  sign(σg)=sign(denom) and |g| ≤ λk|denom|
            if sign(σ * g) == sign(denom) && abs(g) <= b.λ * abs(denom)
                cand_λ = denom > tol ? (σ * g) / denom : 0.0
                new_λ = max(new_λ, cand_λ)
            end
        end
        return new_λ
    end

    λₑₛ(xᵢ) = λₑ([xᵢ])
    λₑ′(xᵢ) = nothing#ForwardDiff.gradient(λₑ, xᵢ)[1] #TODO
    λₑ′′(xᵢ) = nothing#ForwardDiff.hessian(λₑ, xᵢ)[1, 1]

    Δ = step(xgrid)
    domain = extrema(xgrid)

    function λₑmax(xᵢ)
        try
            lb = max(domain[1], xᵢ - Δ)
            ub = min(domain[2], xᵢ + Δ)
            xᵢ = clamp_strict(xᵢ, lb, ub)
            result = optimize(
                xᵢ -> -λₑ(xᵢ),
                [lb],
                [ub],
                [xᵢ],
                Fminbox(LBFGS(linesearch=LineSearches.BackTracking())),
                Optim.Options(show_warnings=false); autodiff=:forward)
            if (abs(λₑ′(Optim.minimizer(result))) < 1e-8 && abs(λₑ′′(Optim.minimizer(result))) < 1e4)
                return (-Optim.minimum(result), Optim.minimizer(result)[1])
            else
                return (λₑ([xᵢ]), xᵢ)
            end
        catch e
            @warn "Local ascent failed, using grid result" error = e
            (λₑ([xᵢ]), xᵢ)
        end
    end

    λₑ_grid = λₑₛ.(xgrid)
    thresh = τ_seed * maximum(λₑ_grid)
    idxs = findall(>=(thresh), λₑ_grid)
    x0s = xgrid[idxs]

    cands = [λₑmax(x0)[1] for x0 in x0s]
    idx_max = argmax(cands)

    b.λ = cands[idx_max]
end

"""
Solve the BLASSO problem using Hybrid (LARS-ODE) Algorithm.

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
function Hybrid!(b::BLASSO,
    options::Dict{Symbol,<:Any}=Dict{Symbol,Any}())

    maxits = get(options, :maxits, 50)
    options[:maxits] = get(options, :inner_maxits, 50)
    progress = get(options, :progress, true)
    options[:progress] = get(options, :progress, false)
    base_solver = get(options, :base_solver, :SFW)
    verbose = get(options, :verbose, false)
    store_reg_path = get(options, :store_reg_path, false)
    η_tol = get(options, :η_tol, 1e-5)
    kink_tol = 1e-3
    τδ = get(options, :τδ, nothing)

    if isnothing(τδ)
        throw(ArgumentError("τδ must be provided in options."))
    elseif τδ ≤ 0
        throw(ArgumentError("τδ must be strictly greater than 0"))
    end

    b.reg_path = Dict(
        :λs => Float64[],
        :μs => DiscreteMeasure[],
    )

    xgrid = build_grid(b.domain, b.n_coarse_grid)

    ϕ, Φ = b.ops.ϕ, b.ops.Φ

    y = b.y
    b.λ = λ₀(xgrid, ϕ, y, η_tol) + eps(Float16)
    b_prev = deepcopy(b)
    b_prev.λ = Inf

    forward = true

    prog = ProgressUnknown(desc="Hybrid iterations: ")

    kink_found = false
    display(_TOP)
    display(_HEAD)
    display(_SEP)

    for _ in 1:maxits
        progress && next!(prog)

        verbose && println("λ = $(b.λ)")

        solve!(b, base_solver; options=options)

        prev_kink_sparsity = sparsity(prev_kink_μ(b.reg_path))

        r = norm(y - Φ(b.μ...))

        if r < τδ
            push!(b.reg_path[:λs], b.λ)
            push!(b.reg_path[:μs], deepcopy(b.μ))
            break
        elseif (abs(b.λ - b_prev.λ) < kink_tol) &&
               (sparsity(b.μ) > prev_kink_sparsity) # hit a kink

            push!(b.reg_path[:λs], b.λ)
            push!(b.reg_path[:μs], deepcopy(b.μ))
            r = norm(Φ(b.μ...) - y)

            verbose && println("r = $r, τδ = $(τδ)")
            verbose && println("μ = $(b.μ)")

            kink_found = true
            forward = true
        elseif sparsity(b.μ) > prev_kink_sparsity # overshot
            reg_path = deepcopy(b.reg_path)
            λs = reg_path[:λs]
            λₖ = isempty(λs) ? Inf : λs[end]
            solve!(b, :NODE, options=Dict(:λ₀ => b.λ, :λₖ => λₖ, :sₖ => prev_kink_sparsity, :h => -(0.01 * kink_tol), :maxits => 100, :inner_maxits => sparsity(b.μ), :progress => false, :base_solver => base_solver, :verbose => false, :η_tol => η_tol, :store_reg_path => false))
            b.reg_path = reg_path
            kink_found = false
            forward = false
        else # undershot
            forward = true
            kink_found = false
        end

        curr_sparsity = sparsity(b.μ)
        prev_sparsity = sparsity(b_prev.μ)

        b_prev = deepcopy(b)
        if forward
            next_kink!(b, ϕ, xgrid, η_tol)
        end

        μs = b.reg_path[:μs]
        if isempty(μs) || length(μs) < 2 || (b.λ - b_prev.λ < 10 * kink_tol)
            b.μ = nothing
        else
            b.μ = μs[end-1]
        end

        if b.λ == b_prev.λ
            if forward
                b.λ -= eps(Float32)
            else
                b.λ += eps(Float32)
            end
        end
        
        display(_row(prog.core.counter, b_prev.λ, curr_sparsity, prev_sparsity, prev_kink_sparsity, forward, b.λ, kink_found))

        if b.λ ≤ 0
            @warn "λ has become non-positive, stopping."
            break
        end
    end

    display(_BOTTOM)

    if !store_reg_path
        b.reg_path = nothing
    end
    return b
end