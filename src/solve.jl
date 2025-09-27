export solve!

"""
Solve the BLASSO problem using the specified solver.

# Arguments
- `b::BLASSO`: BLASSO problem to solve.
- `solver::Symbol=:SFW`: Solver to use.
- `options::Dict()`: Options for the solver.

# Returns
- `b::BLASSO`: BLASSO problem with the recovered measure in the `μ` field. If `:SDP` is used, the dual solution is also stored in the `p` field.
"""
function solve!(b::BLASSO, solver::Symbol=:SFW;
    options::Dict{Symbol,<:Any}=Dict{Symbol,Any}())

    # If λ is not provided for a base solver, use a homotopy method
    if isnothing(b.λ) && !(solver in (:MDP, :FH, :NODE, :Hybrid))
        @warn "λ not provided; using MDP."
        if !haskey(options, :base_solver)
            options = Dict{Symbol,Any}(options)
            options[:base_solver] = solver
        end
        MDP!(b, options)
    else
        if b.λ == 0 && (solver in (:FW, :SFW, :BSFW))
            throw(ArgumentError("SFW and BSFW require λ > 0. Use SDP for unregularised problems."))
        end

        solver_map = Dict(
            :FW => () -> FW!(b, options),
            :SFW => () -> SFW!(b, options),
            :BSFW => () -> BSFW!(b, options),
            :SDP => () -> SDP!(b, options),
            :MDP => () -> MDP!(b, options),
            :FH => () -> FH!(b, options),
            :NODE => () -> NODE!(b, options)
            :Hybrid => () -> Hybrid!(b, options),
        )

        if haskey(solver_map, solver)
            return solver_map[solver]()
        else
            throw(ArgumentError("Solver $solver is not implemented. Available solvers: $(keys(solver_map))."))
        end
    end
    return b
end