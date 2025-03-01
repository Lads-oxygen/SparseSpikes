ENV["GKSwstype"] = "100"

using Revise, Plots, Plots.Measures, FFTW, LinearAlgebra, LaTeXStrings, Random, Distributions, ColorSchemes#, Colors
includet("../src/SparseSpikes.jl")
using .SparseSpikes

# Constants
markersize = 5
markerstrokewidth = 1

zoom = 1000
bbox_size = 0.2


domain = [[0, 1], [0, 1]]

# Define the plot
num_points = 100

plt_grid_x1 = [domain[1][1] + i * (domain[1][2] - domain[1][1]) / num_points for j in 0:num_points, i in 0:num_points]
plt_grid_x2 = [domain[2][1] + j * (domain[2][2] - domain[2][1]) / num_points for j in 0:num_points, i in 0:num_points]

grid = range(0, stop=1, length=(num_points + 1))
plot_size = (100, 100) .* 5
plt = heatmap(xlims=domain[1], ylims=domain[2],
    xaxis=false, yaxis=false, colorbar=false,
    top_margin=-1.94mm, left_margin=-1.94mm,
    bottom_margin=-1.97mm, right_margin=-1.97mm,
    ticks=false,
    framestyle=:none,
    color=:viridis, size=plot_size, ratio=:equal, grid=false)

ops = gaussian_operators_2D(0.2, plt_grid_x1, plt_grid_x2)

x0 = [[0.2, 0.5, 0.8], [0.3, 0.8, 0.2]]
a0 = [1.0, 1.0, 1.0]

μ0 = DiscreteMeasure(x0, a0)

ηV = pre_certificate(μ0, ops)

plt_cert = deepcopy(plt)

heatmap!(plt_cert, grid, grid, ηV, color=:viridis)
plot!(plt_cert, legend_background_color=:white)

# scatter!(plt_cert, μ0.x..., label=L"μ_0", zcolor=μ0.a, color=:viridis, marker=:square, markersize=markersize, markerstrokewidth=markerstrokewidth)

lens_size = (domain[1][2] - domain[1][1]) / zoom
lens_marker_scale = 0.75(bbox_size / lens_size)


inset_pos_adj = [[0, 0.15], [0, -0.15], [-0, 0.15]]


yellow = ColorSchemes.viridis[end]

for i in 1:length(μ0.a)
    μx1 = μ0.x[1][i]
    μx2 = μ0.x[2][i]
    inset_pos = (μx1, μx2) .+ inset_pos_adj[i]

    inset_pos_bl = inset_pos .- (0.5bbox_size, 0.5bbox_size)

    # Draw inset square
    # plot!(plt_cert, Shape([inset_pos[1] - 0.5bbox_size, inset_pos[1] + 0.5bbox_size, inset_pos[1] + 0.5bbox_size, inset_pos[1] - 0.5bbox_size, inset_pos[1] - 0.5bbox_size],
    #         [inset_pos[2] - 0.5bbox_size, inset_pos[2] - 0.5bbox_size, inset_pos[2] + 0.5bbox_size, inset_pos[2] + 0.5bbox_size, inset_pos[2] - 0.5bbox_size]),
    #     linecolor=:black, colour=yellow, linestyle=:dash, label="", subplot=1)

    # Draw square around zoomed area
    # plot!(plt_cert,
    #     [μx1 - lens_size, μx1 + lens_size, μx1 + lens_size, μx1 - lens_size, μx1 - lens_size],
    #     [μx2 - lens_size, μx2 - lens_size, μx2 + lens_size, μx2 + lens_size, μx2 - lens_size],
    #     color=:black, label=nothing, linestyle=:dash, lw=0.01)

    # Draw connection lines
    if i != 2
        lcorner = inset_pos .- (0.5bbox_size, 0.5bbox_size)
        rcorner = inset_pos .- (-0.5bbox_size, 0.5bbox_size)
    else
        lcorner = inset_pos .+ (-0.5bbox_size, 0.5bbox_size)
        rcorner = inset_pos .+ (0.5bbox_size, 0.5bbox_size)
    end
    plot!(plt_cert,
        [μx1 - lens_size, lcorner[1]],
        [μx2 - lens_size, lcorner[2]],
        color=:black, label=nothing, linestyle=:dash, lw=1)
    plot!(plt_cert,
        [μx1 + lens_size, rcorner[1]],
        [μx2 - lens_size, rcorner[2]],
        color=:black, label=nothing, linestyle=:dash, lw=1)


    plot!(plt_cert,
        inset=(bbox(inset_pos_bl..., bbox_size, bbox_size, :bottom)),
        xlims=(μx1 - lens_size, μx1 + lens_size),
        ylims=(μx2 - lens_size, μx2 + lens_size),
        framestyle=:none, bg_colour=yellow,
        margins=0mm, ticks=false,
        subplot=(i + 1))

    # Overlay rectangle on the main plot
    INSET_SCALE = 0.507
    plot!(plt_cert, Shape([inset_pos[1] - INSET_SCALE * bbox_size, inset_pos[1] + INSET_SCALE * bbox_size, inset_pos[1] + INSET_SCALE * bbox_size, inset_pos[1] - INSET_SCALE * bbox_size, inset_pos[1] - INSET_SCALE * bbox_size],
            [inset_pos[2] - INSET_SCALE * bbox_size, inset_pos[2] - INSET_SCALE * bbox_size, inset_pos[2] + INSET_SCALE * bbox_size, inset_pos[2] + INSET_SCALE * bbox_size, inset_pos[2] - INSET_SCALE * bbox_size]),
        linestyle=:dash, fillalpha=0, label="", subplot=1)
end

display(plt_cert)


# for i in 2:4
#     scatter!(plt_cert[i], μ0.x..., label="", zcolor=μ0.a, color=:viridis, marker=:square, markersize=markersize * lens_marker_scale, markerstrokewidth=markerstrokewidth * lens_marker_scale)
# end


y0 = ops.Φ(μ0...)

function add_noise(y0, noise_level)
    sigma = noise_level * norm(y0)
    w = randn(length(y0))
    w = w / norm(w) * sigma
    y = y0 + w
    return y
end

prob = BLASSO(y0, ops, domain)

noise_levels = 0.001 .+ (0.1 - 0.001) * (range(0, stop=1, length=101) .^ 4)
λs = 100 * noise_levels

markersizes = 0.1noise_levels

for (i, noise_level) in enumerate(noise_levels)
    Random.seed!(1)
    prob.y = add_noise(y0, noise_level)
    prob.λ = λs[i]
    solve!(prob, :SFW, options=Dict(:maxits => 4))
    scatter!(plt_cert[1], prob.μ.x..., color=:black, marker=:circle, markersize=markersizes[i], markerstrokewidth=0.2 * markersizes[i], label=nothing)
    for j in 2:4
        scatter!(plt_cert[j], prob.μ.x..., color=:black, marker=:circle, markersize=markersizes[i] * lens_marker_scale, markerstrokewidth=0.2 * markersizes[i] * lens_marker_scale, label=nothing)
    end
    println("Noise level: $(noise_level), λ: $(prob.λ), μ: $(prob.μ.x)")
end

display(plt_cert)

savefig(plt_cert, "precert.svg")