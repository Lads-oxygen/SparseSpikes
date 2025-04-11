using Revise, Plots, LinearAlgebra, LaTeXStrings, Random, BenchmarkTools

includet("../../src/SparseSpikes.jl")
using .SparseSpikes

domain = [[0, 1], [0, 1]]

# Define the plot
num_points = 64

plt_grid_x1 = [domain[1][1] + i * (domain[1][2] - domain[1][1]) / num_points for j in 0:num_points, i in 0:num_points]
plt_grid_x2 = [domain[2][1] + j * (domain[2][2] - domain[2][1]) / num_points for j in 0:num_points, i in 0:num_points]

grid = range(0, stop=1, length=(num_points + 1))

ops = gaussian_operators_2D(0.02, plt_grid_x1, plt_grid_x2)

max_n_spikes = 20
SFW_times = zeros(max_n_spikes)
BSFW_times = zeros(max_n_spikes)

num_trials = 20

for n_spikes = 1:max_n_spikes

    sfw_trial_times = zeros(num_trials)
    bsfw_trial_times = zeros(num_trials)

    for trial = 1:num_trials
        # Use trial-dependent seed for reproducibility but different initializations
        Random.seed!(42 + n_spikes * 100 + trial)

        x0 = [rand(n_spikes), rand(n_spikes)]
        x0 = [round.(x0[1], digits=3), round.(x0[2], digits=3)]

        a0 = clamp.(abs.(round.(rand(Normal(0.9, 0.1), n_spikes), digits=3)), 0, 1)

        μ0 = DiscreteMeasure(x0, a0)

        y0 = ops.Φ(μ0...)

        function add_noise(y0, noise_level)
            sigma = noise_level * norm(y0)
            w = randn(length(y0))
            w = w / norm(w) * sigma
            y = y0 + w
            return y
        end

        # Add noise to the observation y = y0 + w
        noise_level = 0.1
        y = add_noise(y0, noise_level)

        λ = 0.01noise_level

        prob = BLASSO(y, ops, domain, λ)
        sfwtime = @timed solve!(prob, :SFW, options=Dict(:maxits => length(a0) + 10, :positivity => true))
        bsfwtime = @timed solve!(prob, :BSFW, options=Dict(:maxits => length(a0) + 10, :positivity => true))

        sfw_trial_times[trial] = sfwtime.time
        bsfw_trial_times[trial] = bsfwtime.time

    end

    SFW_times[n_spikes] = mean(sfw_trial_times)
    BSFW_times[n_spikes] = mean(bsfw_trial_times)
end


# Create the comparison plot
p = plot(1:15, SFW_times[1:15], label="SFW", linewidth=2, marker=:circle,
    xlabel="Number of Spikes", ylabel="Time (seconds)",
)
plot!(p, 1:15, BSFW_times[1:15], label="BSFW", linewidth=2, marker=:square)

# display(p)
savefig(p, "SFW_BSFW.svg")