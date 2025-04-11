using Revise, Plots, LinearAlgebra, Images
include("../src/SparseSpikes.jl")
using .SparseSpikes

# Define domain and regularization parameter
FOV = 6400
domain = [[0, 1], [0, 1]]

# Define the plot grid
num_points = 600
plt_grid_x1 = [domain[1][1] + i * (domain[1][2] - domain[1][1]) / (num_points - 1) for j in 0:num_points-1, i in 0:num_points-1]
plt_grid_x2 = [domain[2][1] + j * (domain[2][2] - domain[2][1]) / (num_points - 1) for j in 0:num_points-1, i in 0:num_points-1]
grid = range(0, stop=1, length=num_points)
plot_size = (400, 400) .* 2

# Calculate sigma for Gaussian PSF
const σ2 = let
    λ = 723.0 # Wavelength
    NA = 1.4  # Numerical aperture
    FWHM = λ / (2 * NA) # Full width at half maximum i.e. diffraction limit
    σ = FWHM / (2 * log(2.0))
    (σ / (FOV))^2
end
σ = sqrt(σ2) / 40

function readSourcesCSV(filename)
    results = []

    open(filename, "r") do file
        readline(file)  # Skip header

        for line in eachline(file)
            parts = split(line, ",")
            frame = parse(Int, strip(parts[2]))
            xnano = parse(Float64, strip(parts[3])) / FOV
            ynano = parse(Float64, strip(parts[4])) / FOV
            znano = parse(Float64, strip(parts[5])) / 1030

            # If frame already exists in results, add to it; otherwise create new
            if length(results) < frame
                push!(results, DiscreteMeasure([[xnano], [ynano]], [znano]))
            else
                μ = results[frame]
                push!(μ.x[1], xnano)
                push!(μ.x[2], ynano)
                push!(μ.a, znano)
            end
        end
    end

    return results
end

# After reading the results
function mergeDiscreteMeasures(measures)
    x1 = Vector{Float32}(vcat([m.x[1] for m in measures]...))
    x2 = Vector{Float32}(vcat([m.x[2] for m in measures]...))
    a = Vector{Float32}(vcat([m.a for m in measures]...))
    return DiscreteMeasure([x1, x2], a)
end

# Read the results and create discrete measures
results = readSourcesCSV("high_density/fluorophores/activation.csv")

# results = readSourcesCSV("results/SMLM_low_density_results.csv")

ops = gaussian_operators_2D(σ, plt_grid_x1, plt_grid_x2)

# Apply the PSF operator to the discrete measure
μ = mergeDiscreteMeasures(results)

reconstructed_image = reshape(ops.Φ(μ.x..., μ.a), num_points, num_points)

using Plots.Measures
# Display the result
heatmap(grid, grid, reconstructed_image, c=:viridis, ratio=:equal, size=plot_size, ticks=false, xlabel="", ylabel="", colorbar=false, margins=-2mm)

savefig("figures/SMLM/SMLM_sources_high.svg")

# worse results near ends where psfs are most likely to overlap