ENV["GKSwstype"] = "100"

using Revise, Plots, Plots.Measures, FFTW, LinearAlgebra, LaTeXStrings, Random, Distributions, ColorSchemes
includet("../../src/SparseSpikes.jl")
using .SparseSpikes

# Constants
markersize = 5
markerstrokewidth = 1

zoom = 100
bbox_size = 0.2

domain = [[0, 1], [0, 1]]

n_coarse_grid = 21
n_plt_grid = 101

coarse_grids = grid(domain, n_coarse_grid)
plt_grids = grid(domain, n_plt_grid)
hm_grid = grid(domain[1, :], n_plt_grid)

plot_size = (100, 100) .* 5
plt = heatmap(xlims=domain[1], ylims=domain[2],
    xaxis=false, yaxis=false, colorbar=false,
    top_margin=-1.94mm, left_margin=-1.94mm,
    bottom_margin=-1.97mm, right_margin=-1.97mm,
    ticks=false,
    framestyle=:none,
    color=:viridis, size=plot_size, ratio=:equal, grid=false)

ops = gaussian_operators_2D(0.05, coarse_grids)

x0 = [[0.2, 0.5, 0.8], [0.3, 0.8, 0.2]]
a0 = [1.0, 1.0, 1.0]

μ0 = DiscreteMeasure(x0, a0)

ηV = build_ηV(μ0, ops)

plt_cert = deepcopy(plt)

heatmap!(plt_cert, hm_grid, hm_grid, ηV(plt_grids), color=:viridis)

lens_size = (domain[1][2] - domain[1][1]) / zoom
lens_marker_scale = 0.75(bbox_size / lens_size)

inset_pos_adj = [[0, 0.15], [0, -0.15], [-0, 0.15]]

yellow = ColorSchemes.viridis[end]

for i in 1:length(μ0.a)
    μx1 = μ0.x[1][i]
    μx2 = μ0.x[2][i]
    inset_pos = (μx1, μx2) .+ inset_pos_adj[i]

    inset_pos_bl = inset_pos .- (0.5bbox_size, 0.5bbox_size)

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

y0 = ops.Φ(μ0...)

function add_noise(y0, noise_level)
    sigma = noise_level * norm(y0)
    w = randn(length(y0))
    w = w / norm(w) * sigma
    y = y0 + w
    return y
end

prob = BLASSO(y0, ops, domain, n_coarse_grid)

num_ns_lvls = 100

noise_levels = 0.01 .+ (1.001 - 0.001) * (range(0, stop=1, length=num_ns_lvls) .^ 2)
λs = 0.005noise_levels

markersizes = 0.1noise_levels

colors = get(get_cmap(:black_red), 1:num_ns_lvls, :extrema)

for (i, noise_level) in enumerate(noise_levels)
    Random.seed!(1)
    prob.y = add_noise(y0, noise_level)
    prob.λ = λs[i]
    solve!(prob, :SFW, options=Dict(:maxits => 5, :positivity => true))
    scatter!(plt_cert[1], prob.μ.x..., color=:black, marker=:circle, markersize=markersizes[i], markerstrokewidth=0.2 * markersizes[i], label=nothing)
    for j in 2:4
        scatter!(plt_cert[j], prob.μ.x..., color=colors[i], marker=:circle, markersize=markersizes[i] * lens_marker_scale, markerstrokewidth=0, label=nothing)
    end
    println("Noise level: $(noise_level), λ: $(prob.λ), μ: $(prob.μ.x)")
end

display(plt_cert)

savefig(plt_cert, "../../figures/pre_certificates/noise_robustness.svg")