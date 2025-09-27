using Revise  # Development tool
using Images: load, channelview
using Printf: @sprintf
using Base.Threads: @threads
using ProgressMeter: Progress, next!

include("../src/SparseSpikes.jl")
using .SparseSpikes

# Define domain and regularisation parameter
domain = [[0, 1], [0, 1]]

n_coarse_grid = 21
n_plt_grid = 64

plt_grids = grid(domain, n_plt_grid)
hm_grid = grid(domain[1, :], n_plt_grid)
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
ops = gaussian_operators_2D(σ, plt_grids)

"""
    readImage!(buffer, imageDir, frameNum)

Load a TIFF into `buffer` in-place and return it.
"""
function readImage!(
    buffer::AbstractMatrix{T},
    imageDir::AbstractString,
    frameNum::Integer
) where T
    imageId = @sprintf("%05d", frameNum)
    img = load("$imageDir/$imageId.tif")
    cv = channelview(img)             # this allocates once per frame
    copyto!(buffer, cv)               # re-use your big buffer
    return buffer
end

# Process a single frame
function runSFW(image, ops, λ, domain, solver=:BSFW, options=Dict())
    y = vec(image)
    prob = BLASSO(y, ops, domain, n_coarse_grid, λ=λ)
    options = merge(Dict(:maxits => 100, :positivity => true, :progress => false), options)
    @time solve!(prob, solver, options=options)
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
        if μ.d == 2
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

λ = 0.06
nImages = 32

results = Vector{Any}(undef, nImages)
input_file = "SMLM/low_density_data/sequence"
output_file = get_unique_filename("SMLM/results/low_density_results.csv")

# ————————————————
# INITIALISE img_buffer BY LOADING FRAME #1 ONCE
first_id = @sprintf("%05d", 1)
first_path = "$(input_file)/$first_id.tif"
first_img = load(first_path)
first_cv = channelview(first_img)          # allocate once
const img_buffer = similar(first_cv)       # same eltype & size
# ————————————————


p = Progress(nImages; desc="Frame: ")
@time Threads.@threads for frameNum in 1:nImages
    image = readImage!(img_buffer, input_file, frameNum)
    results[frameNum] = runSFW(image, ops, λ, domain, :BSFW, Dict(:descent => :BFGS))
    next!(p)
end

for frameNum in 1:nImages
    writeToCSV(output_file, results[frameNum], frameNum)
end