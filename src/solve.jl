using ..SparseSpikes

export solve!

"""
Solve the BLASSO problem using the specified solver (`:SDP` or `:SFW`).

# Arguments
- `blasso::BLASSO`: BLASSO problem to solve.
- `solver::Symbol=:SFW`: Solver to use.
- `options::Dict()`: Options for the solver.

# Returns
- `blasso::BLASSO`: BLASSO problem with the recovered measure in the `μ` field. If `:SDP` is used, the dual solution is also stored in the `p` field.
"""
function solve!(blasso::BLASSO,
    solver::Symbol=:SFW;
    options::Dict{Symbol,<:Any}=Dict{Symbol,Any}())

    # If λ is not provided, use Morozov's Discrepancy rule
    if isnothing(blasso.λ) && solver != :MDP
        @warn "λ not provided; using Morozov's Discrepancy rule."
        if !haskey(options, :base_solver)
            options = Dict{Symbol,Any}(options)
            options[:base_solver] = solver
        end
        MDP!(blasso, options)
    else
        if blasso.λ == 0 && (solver == :SFW || solver == :BSFW)
            throw(ArgumentError("SFW and BSFW require λ > 0. Use SDP for unregularised problems."))
        end

        solver_map = Dict(
            :SFW => () -> SFW!(blasso, options),
            :BSFW => () -> BSFW!(blasso, options),
            :SDP => () -> SDP!(blasso),
            :MDP => () -> MDP!(blasso, options)
        )

        if haskey(solver_map, solver)
            return solver_map[solver]()
        else
            throw(ArgumentError("Solver must be either :SFW, :BSFW, :SDP or :MDP."))
        end
    end
    return blasso
end