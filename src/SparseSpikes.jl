module SparseSpikes

# Dev
using Revise

includet("operators.jl")
includet("utils.jl")
includet("blasso.jl")
includet("regularisation_paths.jl")
includet("plots.jl")
includet("semidefinite_programming.jl")
includet("sliding_frank_wolfe.jl")
includet("morozov_discrepancy_principle.jl")
includet("fast_homotopy.jl")
includet("node.jl")
includet("hybrid.jl")
includet("solve.jl")

export solve!

end
