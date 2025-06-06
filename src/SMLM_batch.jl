using Revise, Plots, LinearAlgebra, LaTeXStrings, Images, Printf
using Base.Threads, ProgressMeter
include("../src/SparseSpikes.jl")
using .SparseSpikes

# Define domain and regularization parameter
domain = [[0, 1], [0, 1]]

# Define the plot grid
num_points = 64
plt_grid_x1 = [domain[1][1] + i * (domain[1][2] - domain[1][1]) / (num_points - 1) for j in 0:num_points-1, i in 0:num_points-1]
plt_grid_x2 = [domain[2][1] + j * (domain[2][2] - domain[2][1]) / (num_points - 1) for j in 0:num_points-1, i in 0:num_points-1]
grid = range(0, stop=1, length=num_points)
plot_size = (400, 250) .* 2

# Calculate sigma for Gaussian PSF
const σ2 = let
    λ = 723.0 # Wavelength
    NA = 1.4  # Numerical aperture
    FWHM = λ / (2 * NA) # Full width at half maximum i.e. diffraction limit
    σ = FWHM / (2 * log(2.0))
    (σ / (64 * 100.0))^2
end
σ = sqrt(σ2)

# Create operators
ops = gaussian_operators_2D(σ, plt_grid_x1, plt_grid_x2)

# Read TIFF image
function readImage(imageDir, frameNum, n_pixels_x)
    imageId = @sprintf("%05d", frameNum)
    img = load("$imageDir/$imageId.tif")
    return channelview(img)
end

# Process a single frame
function runSFW(image, ops, λ, domain, options=Dict())
    y = vec(image)
    prob = BLASSO(y, ops, domain, λ)
    options = merge(Dict(:maxits => 100, :positivity => true), options)
    solve!(prob, :BSFW, options=options)
    return prob.μ
end

# Write a single frame result to CSV, appending to existing file
function writeToCSV(filename, μ, frame)
    # Create file with header if it doesn't exist
    if !isfile(filename)
        open(filename, "w") do csvfile
            write(csvfile, "Ground-truth,frame,xnano,ynano,znano,intensity\n")
        end
    end

    # Append results
    open(filename, "a") do csvfile
        if μ.dims == 2
            for i in eachindex(μ.a)
                xPos = @sprintf("%.2f", μ.x[1][i] * 6400)
                yPos = @sprintf("%.2f", μ.x[2][i] * 6400)
                zPos = @sprintf("%.2f", μ.a[i] * 1030)
                write(csvfile, "$i, $frame, $xPos, $yPos, $zPos\n")
            end
        else
            error("Unsupported dimensions")
        end
    end
end

# Find a unique filename by appending a number if needed
function get_unique_filename(basepath)
    path, ext = splitext(basepath)
    i = 1
    filename = basepath
    while isfile(filename)
        filename = "$(path)_$(i)$(ext)"
        i += 1
    end
    return filename
end

λ = 0.0025
nImages = 361

results = Vector{Any}(undef, nImages)
input_file = "SMLM/high_density_data/sequence"
output_file = get_unique_filename("SMLM/results/high_density_results.csv")

@time @showprogress desc = "Frame: " Threads.@threads for frameNum in 1:nImages
    image = readImage(input_file, frameNum, 64)
    results[frameNum] = runSFW(image, ops, λ, domain, Dict(:descent => :BFGS))
end

for frameNum in 1:nImages
    writeToCSV(output_file, results[frameNum], frameNum)
end

# 5115.790710 seconds sparse sfw

# 4985.769319 seconds high density bsfw (only lasso basically)

# 28474.787277 seconds high density bsfw (50% gc)