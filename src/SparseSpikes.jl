module SparseSpikes

include("operators.jl")
include("utils.jl")
include("blasso.jl")
include("regularisation_paths.jl")
include("plots.jl")
include("semidefinite_programming.jl")
include("sliding_frank_wolfe.jl")
include("morozov_discrepancy_principle.jl")
include("fast_homotopy.jl")
include("node.jl")
include("hybrid.jl")
include("solve.jl")

export solve!

end
