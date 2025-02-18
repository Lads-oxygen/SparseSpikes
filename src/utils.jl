using ForwardDiff

export DiscreteMeasure, plot_spikes!, pre_certificate, plot_roots!, complex_grad

struct DiscreteMeasure
    x::Union{Vector{Float64},Vector{Vector{Float64}}}
    a::Vector{Float64}
    dims::Int
    n_spikes::Int

    function DiscreteMeasure(x::Vector{Float64}, a::Vector{Float64})
        new(x, a, 1, length(x))
    end

    function DiscreteMeasure(x::Vector{Vector{Float64}}, a::Vector{Float64})
        new(x, a, length(x), length(x[1]))
    end
end

function Base.iterate(μ::DiscreteMeasure, i=1)
    if i > μ.dims + 1
        return nothing
    elseif i == μ.dims + 1
        return μ.a, i + 1
    else
        if μ.dims == 1
            return μ.x, i + 1
        else
            return μ.x[i], i + 1
        end
    end
end

"""
    plot_spikes!(plt, measure; color=:red, label="", marker=:circle)

Plot the spikes of a discrete measure.

# Arguments
- `plt`: Plot object to mutate.
- `μ`: Discrete measure to plot.
- `color`: Color of the spikes.
- `label`: Label for the spikes.
- `marker`: Marker style for the spikes.
"""
function plot_spikes!(plt::Plots.Plot, μ::DiscreteMeasure; color::Symbol=:red, colorscheme=:viridis, label::Union{String,LaTeXString}="", marker::Symbol=:circle)
    if μ.dims == 1
        plot!(plt, μ.x, μ.a, seriestype=:scatter, color=color, label=label, marker=marker)
    elseif μ.dims == 2
        scatter!(plt, μ.x[1], μ.x[2], zcolor=μ.a, color=colorscheme, label=label, marker=marker, colorbar=true)
    end
end

"""
    plot_roots!(plt, r; x0=nothing)

Plot the roots of the polynomial on the unit circle. Optionally, plot the support of x.

# Arguments
- `plt`: Plot object to mutate.
- `r`: Roots of the polynomial.
- `x0`: Initial positions (optional).
"""
function plot_roots!(plt::Plots.Plot, r; x0=nothing)
    plot!(plt, cos.(range(0, stop=2π, length=200)), sin.(range(0, stop=2π, length=200)), linestyle=:dash, label="") # Unit circle
    if !isnothing(x0)
        plot!(plt, cos.(2π .* x0), sin.(2π .* x0), seriestype=:scatter, label="Support of x", marker=:o, color=:red) # Support of x
    end
    plot!(plt, real(r), imag(r), seriestype=:scatter, label="Roots", marker=:x) # Roots
end

"""
    pre_certificate(μ0, ops) -> Vector{Float64}

Compute the pre-certificate for a given measure.

# Arguments
- `μ0`: Discrete measure.
- `ops`: Operators.

# Returns
- `ηV`: Pre-certificate vector.
"""
function pre_certificate(μ0::DiscreteMeasure, ops::Operators)::Union{Vector,Matrix}
    b = [sign.(μ0.a); zeros(μ0.dims * μ0.n_spikes)]
    ϕ = ops.ϕ
    if μ0.dims == 1
        dϕ = x -> complex_grad(ϕ, x)
        Γx = [ϕ(μ0.x)'; dϕ(μ0.x)']
    elseif μ0.dims == 2
        dxϕ = (x1, x2) -> complex_grad(ξ -> ϕ(ξ, x2), x1)
        dyϕ = (x1, x2) -> complex_grad(ξ -> ϕ(x1, ξ), x2)
        Γx = [
            ϕ(μ0.x...)';
            dxϕ(μ0.x...)';
            dyϕ(μ0.x...)'
        ]
    else
        error("Not implemented")
    end
    pV = pinv(Γx) * b
    ηV = ops.adjΦ(pV)
    return ηV
end

"""
    complex_grad(f, x) -> Array{Complex{Float64}, 1}

Compute the gradient of a complex-valued function.

# Arguments
- `f`: Complex-valued function.
- `x`: Point at which to evaluate the gradient.

# Returns
- `grad`: Gradient of the function at `x`.
"""
function complex_grad(f, x)
    fx = f(x)# TODO: avoid function call
    if eltype(fx) <: Complex
        real_part = ForwardDiff.jacobian(ξ -> real(f(ξ)), x)
        imag_part = ForwardDiff.jacobian(ξ -> imag(f(ξ)), x)
        jac = real_part + im * imag_part
    else
        jac = ForwardDiff.jacobian(f, x)
    end
    grad = reshape(sum(jac, dims=2), size(fx))
    return grad
end