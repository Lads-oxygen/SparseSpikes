using LinearAlgebra: norm

export build_reg_paths

"""
Greedy match under threshold T on a cost matrix of size n_prev × n_curr.
"""
function greedy_match(cost::AbstractMatrix{<:Real}, T::Real)
    n_prev, n_curr = size(cost)
    matched_prev = falses(n_prev)
    matched_curr = falses(n_curr)
    matches = Vector{Tuple{Int,Int}}()

    while true
        best_cost = Inf
        best_i = 0
        best_j = 0


        # Find the smallest-cost unmatched pair
        for i in 1:n_prev
            !matched_prev[i] || continue
            for j in 1:n_curr
                !matched_curr[j] || continue
                c = cost[i, j]
                if c < best_cost
                    best_cost = c
                    best_i, best_j = i, j
                end
            end
        end

        # Stop if none acceptable or exhausted
        if best_cost > T || best_i == 0
            break
        end

        matched_prev[best_i] = true
        matched_curr[best_j] = true
        push!(matches, (best_i, best_j))

        # if one side is exhausted, we’re done
        if all(matched_prev) || all(matched_curr)
            break
        end
    end

    return matches
end

position_distance(p, q) = (isa(p, Number) && isa(q, Number)) ? abs(p - q) : norm(p .- q)

"""
Construct spike paths from a sequence of DiscreteMeasure BLASSO solutions.

Positional arguments
- λs :: Vector{<:Real}        : regularisation levels in descending order
- μs :: Vector                : BLASSO measures at each λ, each with fields `x` (positions) and `a` (amplitudes)

Keyword arguments:
- C  :: Real = 1.0            : threshold scale (Tₖ = C * |Δλ|)

Returns a NamedTuple with
- x: Vector of position-paths
- a: Vector of amplitude-paths
"""
function build_reg_paths(λs::Vector{<:Real}, μs; C::Real=1.0)
    K = length(λs)
    spike_lists = Vector{Vector{NamedTuple{(:x, :a),Tuple{Any,Any}}}}(undef, K)
    for k in 1:K
        μ = μs[k]
        if μ === nothing || isempty(μ.a)
            spike_lists[k] = NamedTuple{(:x, :a),Tuple{Any,Any}}[]
        else
            if isa(μ.x, AbstractMatrix)
                d, s = size(μ.x)
                xs = d == 1 ? vec(μ.x) : [μ.x[:, j] for j in 1:s]
            elseif isa(μ.x, AbstractVector)
                xs = μ.x
            else
                error("Unsupported position container at λ index $k")
            end
            @assert length(μ.a) == length(xs)
            spike_lists[k] = [(x=xs[j], a=μ.a[j]) for j in eachindex(xs)]
        end
    end

    x_paths = Vector{Vector{Any}}()
    a_paths = Vector{Vector{Float64}}()
    path_to_idx = Union{Int,Nothing}[]

    # Seed (FIX: Vector{Any} instead of fill(Any,...))
    for sp in spike_lists[1]
        xv = Vector{Any}(undef, K); fill!(xv, nothing)
        av = fill(NaN, K)
        xv[1] = sp.x
        av[1] = sp.a
        push!(x_paths, xv)
        push!(a_paths, av)
        push!(path_to_idx, findfirst(t -> t.x === sp.x && t.a === sp.a, spike_lists[1]))
    end

    # Iterate levels
    for k in 1:(K-1)
        Δλ = abs(λs[k+1] - λs[k])
        T = C * Δλ
        S_prev, S_curr = spike_lists[k], spike_lists[k+1]
        n_prev, n_curr = length(S_prev), length(S_curr)

        cost = zeros(n_prev, n_curr)
        for i in 1:n_prev, j in 1:n_curr
            cost[i,j] = position_distance(S_prev[i].x, S_curr[j].x)
        end

        matches = greedy_match(cost, T)
        matched_prev = Set(m[1] for m in matches)
        matched_curr = Set(m[2] for m in matches)

        old_map = copy(path_to_idx)

        for p in 1:length(x_paths)
            i = old_map[p]
            if i isa Int && (i in matched_prev)
                j = first(m[2] for m in matches if m[1] == i)
                x_paths[p][k+1] = S_curr[j].x
                a_paths[p][k+1] = S_curr[j].a
                path_to_idx[p] = j
            else
                x_paths[p][k+1] = nothing       # FIX: was Any
                a_paths[p][k+1] = NaN
                path_to_idx[p] = nothing
            end
        end

        for j in 1:n_curr
            if !(j in matched_curr)
                xv = Vector{Any}(undef, K); fill!(xv, nothing)   # FIX here too
                av = fill(NaN, K)
                xv[k+1] = S_curr[j].x
                av[k+1] = S_curr[j].a
                push!(x_paths, xv)
                push!(a_paths, av)
                push!(path_to_idx, j)
            end
        end
    end

    for i in 1:length(x_paths)
        path = x_paths[i]
        if all(p -> (p === nothing) || isa(p, Number), path)
            x_paths[i] = [p === nothing ? NaN : float(p) for p in path]
        end
    end

    return (x=x_paths, a=a_paths)
end