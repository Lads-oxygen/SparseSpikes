using Plots, LaTeXStrings, Printf, Plots.Measures

export get_cmap, plot_spikes!, plot_roots!, plot_reg_paths, animate_reg_path_2D

const CUSTOM_COLORMAPS = Dict{Symbol,Any}(
    :red_black => cgrad([RGB(1, 0, 0), RGB(0, 0, 0)]),
    :black_red => cgrad([RGB(0, 0, 0), RGB(1, 0, 0)])
)

"""
Get a colormap by symbol name. Returns the requested custom colormap if available,
otherwise falls back to built-in colormaps from Plots.
"""
function get_cmap(name::Symbol)
    if haskey(CUSTOM_COLORMAPS, name)
        return CUSTOM_COLORMAPS[name]
    else
        return name
    end
end

"""
Plot the spikes of a discrete measure.

# Arguments
- `plt`: Plot object to mutate.
- `μ`: Discrete measure to plot.
- `color`: Color of the spikes.
- `colorscheme`: Colorscheme of the spikes (2D).
- `label`: Label for the spikes.
- `marker`: Marker style for the spikes.
- `markersize`: Size of the markers.
- `markerstrokewidth`: Width of the marker stroke.
"""
function plot_spikes!(plt::Plots.Plot, μ::DiscreteMeasure; color::Symbol=:red, colorscheme=get_cmap(:viridis), label::Union{String,LaTeXString}="", marker::Symbol=:circle, markersize::Real=5, markerstrokewidth::Real=0)
    if μ.d == 1
        scatter!(plt, vec(μ.x), vec(μ.a); color=color,
            label=label, marker=marker, markersize=markersize, markerstrokewidth=markerstrokewidth)
    elseif μ.d == 2
        scatter!(plt, inset=(bbox(0, 0, 1, 1, :bottom)), bgcolor=:transparent,
            μ.x[1, :], μ.x[2, :], zcolor=μ.a, color=colorscheme, label=label, marker=marker, colorbar=false, markersize=markersize, markerstrokewidth=markerstrokewidth,
            grid=false, ticks=:none)
    end
end
function plot_spikes!(μ::DiscreteMeasure; color::Symbol=:red, colorscheme=get_cmap(:red_black), label::Union{String,LaTeXString}="", marker::Symbol=:circle, markersize::Real=5, markerstrokewidth::Real=0)
    if μ.d == 1
        scatter!(μ.x, μ.a, color=color, label=label, marker=marker, markersize=markersize, markerstrokewidth=markerstrokewidth)
    elseif μ.d == 2
        scatter!(inset=(bbox(0, 0, 1, 1, :bottom)), bgcolor=:transparent,
            μ.x[1], μ.x[2], zcolor=μ.a, color=colorscheme, label=label, marker=marker, colorbar=false, markersize=markersize, markerstrokewidth=markerstrokewidth,
            grid=false, ticks=:none)
    end
end

"""
Plot the roots of the polynomial on the unit circle. Optionally, plot the support of x.

# Arguments
- `plt`: Plot object to mutate.
- `r`: Roots of the polynomial.
- `x0`: Initial positions (optional).
"""
function plot_roots!(plt::Plots.Plot, r; x0=nothing)
    plot!(plt, cos.(range(0, stop=2π, length=200)), sin.(range(0, stop=2π, length=200)), linestyle=:dash, label="") # Unit circle
    if !isnothing(x0)
        scatter!(plt, cos.(2π .* x0), sin.(2π .* x0), label="Support of x", marker=:o, color=:red) # Support of x
    end
    return scatter!(plt, real(r), imag(r), label="Roots", marker=:x) # Roots
end
function plot_roots!(r; x0=nothing)
    plot!(cos.(range(0, stop=2π, length=200)), sin.(range(0, stop=2π, length=200)), linestyle=:dash, label="") # Unit circle
    if !isnothing(x0)
        scatter!(cos.(2π .* x0), sin.(2π .* x0), label="Support of x", marker=:o, color=:red) # Support of x
    end
    return scatter!(real(r), imag(r), label="Roots", marker=:x) # Roots
end


"""
Plot a path with gaps (NaN values) by connecting only non-NaN segments.

# Arguments
- `plt`: Plot object to mutate
- `x`: x-coordinates
- `y`: y-coordinates (may contain NaN values for gaps)
- `label`: Series label
"""
function plot_path_with_gaps(plt, x, y, label)
    valid_idx = .!isnan.(y)
    if any(valid_idx)
        plot!(plt, x[valid_idx], y[valid_idx], label=label, marker=:circle, markersize=2, markerstrokewidth=0)
    end
end
function plot_path_with_gaps(x, y, label)
    valid_idx = .!isnan.(y)
    if any(valid_idx)
        plot!(x[valid_idx], y[valid_idx], label=label, marker=:circle, markersize=2, markerstrokewidth=0)
    end
end

"""
Plot amplitude and position regularisation paths on a single plot with subplots.

# Arguments
- `λs`: Vector of regularisation parameters
- `x_paths`: Vector of position paths (each path is a vector with possible NaN gaps)
- `a_paths`: Vector of amplitude paths (each path is a vector with possible NaN gaps)

# Keyword Arguments
- `log_scale`: Use log scale for λ axis (default: false)
- `amp_ylims`: Y-axis limits for amplitude plot (default: automatic)
- `pos_ylims`: Y-axis limits for position plot (default: automatic)
- `size`: Plot size (default: (800, 600))
- `legend`: Show legend (default: false)

# Example
```julia
λs, x_paths, a_paths = build_reg_paths(λs, μs)
plot_reg_paths(λs, x_paths, a_paths)
```
"""
function plot_reg_paths(λs, x_paths, a_paths;
    log_scale::Bool=false,
    amp_ylims=nothing,
    pos_ylims=nothing,
    size=(800, 800),
    legend::Bool=false)

    x_vals = log_scale ? log.(λs) : λs
    x_label = log_scale ? L"\log(λ)" : L"λ"

    fmt2 = y -> @sprintf("%.2f", y)

    # Amplitude paths subplot
    plt_amp = plot(xlabel=x_label, ylabel=L"a", yguidefontrotation=-90, yformatter=fmt2, legend=legend)
    for (i, amp_traj) in enumerate(a_paths)
        plot_path_with_gaps(plt_amp, x_vals, amp_traj, "Spike $i")
    end
    if !isnothing(amp_ylims)
        ylims!(plt_amp, amp_ylims)
    end

    # Position paths subplot
    plt_pos = plot(xlabel=x_label, ylabel=L"x", yguidefontrotation=-90, yformatter=fmt2, legend=legend)
    for (i, pos_traj) in enumerate(x_paths)
        plot_path_with_gaps(plt_pos, x_vals, pos_traj, "Spike $i")
    end
    if !isnothing(pos_ylims)
        ylims!(plt_pos, pos_ylims)
    end

    return plot(plt_amp, plt_pos, layout=(1, 2), link=:x, size=size, margins=5mm)
end

"""
Create an animated GIF showing the evolution of 2D spikes over the regularisation path.

Arguments:
- λs: Vector of λ values
- x_paths: Vector of length 2, each entry is a vector of spike positions for x1 and x2
- a_paths: Vector of amplitude paths (each path is a vector)
- domain: 2D domain, e.g. [[0,1],[0,1]]
- filename: Output GIF filename

Keyword arguments:
- marker_scale: scales marker size by amplitude (default: 10)
- fps: frames per second (default: 10)
"""
function animate_reg_path_2D(λs, x_paths, a_paths, domain, filename; colorscheme=get_cmap(:red_black), fps=10, step=1)

    n_steps = length(λs)
    n_spikes = length(a_paths)
    alims = extrema([a for path in a_paths for a in path if !isnan(a)])
    anim = @animate for k in 1:step:n_steps
        plt = plot(xlim=domain[1], ylim=domain[2], aspect_ratio=1,
            xlabel="x₁", ylabel="x₂", title=@sprintf("λ = %.3g", λs[k]))
        x1s = x_paths[1][k]
        x2s = x_paths[2][k]
        if !isnothing(x1s)
            xs = [[x1s[i], x2s[i]] for i in eachindex(x1s)]
            as = [a_paths[i][k] for i in 1:n_spikes]
            scatter!(plt, xs..., zcolor=as, marker=:circle, color=colorscheme, clims=alims, label=false,
            markerstrokewidth=0)
        end
    end
    gif(anim, filename; fps=fps)
end