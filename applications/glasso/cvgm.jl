cd(@__DIR__)
cd("../..")
pwd()
using Statistics
using JuMP
using LinearAlgebra
using ConeProgramDiff
using Random
using SCS

include("glasso_conic.jl")

function generate_partitions(samples; K=10, ρ=0.7)
    nsamples = size(samples, 1)
    num_training = Int64(ρ*nsamples)

    shuffled_datasets = [samples[shuffle(1:end), :] for i=1:K]
    training_datasets = [d[1:num_training, :] for d in shuffled_datasets]
    test_datasets = [d[(num_training+1):end, :] for d in shuffled_datasets]
    return training_datasets, test_datasets
end


function grid_search_cv(samples, alphas, alg, loss; K=10, ρ=0.7)
    training_datasets, test_datasets = generate_partitions(samples, K=K, ρ=ρ)

    test_losses = Array{Float64}(undef, length(alphas), K)
    for (fold_ix, (training_data, test_data)) in enumerate(zip(training_datasets, test_datasets))
        for (alpha_ix, alpha) in enumerate(alphas)
            p = alg(training_data, alpha)
            test_losses[alpha_ix, fold_ix] = loss(test_data, p)
        end
    end

    avg_losses = mean(test_losses, dims=2)
    best_ix = argmin(avg_losses)
    best_alpha = alphas[best_ix]

    p = alg(samples, best_alpha)
    return p, best_alpha, test_losses
end


function cvgm(samples, α0, alg, loss, loss_deriv; K=10, ρ=0.7, max_iters=50, η=0.1, tol=1e-4)
    num_iter = 0
    αt = α0

    alpha_path = []
    gradient_path = []
    loss_path = []
    diff = Inf
    println("==================")
    while (num_iter < max_iters) & (diff > tol)
        num_iter += 1
        println("-----")
        println("αt ", αt)

        gradients = []
        losses = []
        for (training_data, test_data) in zip(generate_partitions(samples, K=K, ρ=ρ)...)
            θt, D = alg(training_data, αt)
            dθ_dα = D(αt)
            dloss_dθ = loss_deriv(test_data, θt)
            dloss_dα = sum(dθ_dα .* dloss_dθ)
            push!(gradients, dloss_dα)
            push!(losses, loss(training_data, θt))
        end
        avg_gradient = median(gradients)
        avg_loss = median(losses)
        println("grad ", avg_gradient)
        println("loss ", avg_loss)

        push!(gradient_path, avg_gradient)
        push!(alpha_path, αt)
        push!(loss_path, avg_loss)
        α_new = αt - η * avg_gradient / sqrt(num_iter)
        α_new = max(α_new, 0)
        diff = norm(α_new - αt)
        println("diff ", diff)
        αt = α_new
    end
    θ = alg(samples, αt)
    return θ, αt, alpha_path, gradient_path, loss_path
end


function glasso(samples, α)
    Σ = cor(samples)

    A, b, c, cone_dict = write_glasso_cone_program(Σ, α)

    x, y, s, D, DT = solve_and_diff(A, b, c, cone_dict, eps=1e-9)

    p = size(Σ, 1)
    θ_size = Int64(p*(p+1)/2)
    function derivative(dα)
        # Changing α changes the input parameter c
        dc = zeros(size(c))
        m_offset = 2*θ_size + p + 1
        for (i, j) in tril_indices(Σ)
            dc[vectorized_index(i, j) + m_offset] = (i == j ? 1 : 2)
        end
        dA = zeros(size(A))
        db = zeros(size(b))
        dx, dy, ds = D(dA, db, dc)

        # Extract just the change in θ
        dθ = dx[1:θ_size]
        return vec2sym(dθ, p)
    end

    sol = value.(x)
    p = size(Σ, 1)
    θ = sol[1:Int64(p*(p+1)/2)]
    return vec2sym(θ, p), derivative
end


function vec2sym(a, dim)
    A = Array{Float64}(undef, dim, dim)
    A[triu(trues(size(A)))] = a
    A = A + A'
    A[diagind(A)] /= 2
    return A
end

function nll(samples, θ)
    test_cov = cov(samples)
    d = det(θ)
    if d < 0
        return -Inf
    end
    return sum(test_cov .* θ) - log(d)
end

function nll_deriv(samples, θ)
    test_cov = cov(samples)
    return test_cov - inv(θ)
end

using Distributions
using Random
using Plots

Random.seed!(1231)

p = 3
num_samples = 60
B = randn(p, p)
B[triu(trues(size(B)))] .= 0
for i=1:p
    for j=1:i
        if rand() < .95
            B[i, j] = 0
        end
    end
end
K = (I - B)' * (I - B)
K = I(p)
Σ = inv(K)
Σ = (Σ + Σ')/2
min(eigvals(K)...)
samples = rand(MultivariateNormal(zeros(p), Σ), num_samples)'
heldout_samples = rand(MultivariateNormal(zeros(p), Σ), 1000)'

α_grid = .01:.01:.4

θs = []
derivs = []
for α in α_grid
    θ, D = glasso(samples, α)
    dθ_dα = D(α)
    dloss_dθ = nll_deriv(heldout_samples, θ)
    dloss_dα = sum(dθ_dα .* dloss_dθ)
    println(max(abs.(dloss_dθ)...))
    println(max(abs.(dθ_dα)...))
    push!(θs, θ)
    push!(derivs, dloss_dα)
end
median(derivs)
min(derivs...)
max(derivs...)

dets = [det(θ) for θ in θs]
validation_loss = [nll(heldout_samples, θ) for θ in θs]

# θ, best_alpha, test_losses = grid_search_cv(samples, alphas, glasso, nll, K=2)
plot(α_grid, validation_loss, legend=false)
xlabel!("λ")
ylabel!("Validation loss")
arrow_len = .02
x_steps = sign.(derivs)
y_steps = abs.(derivs)
quiver!(α_grid, validation_loss, quiver=(arrow_len*x_steps, arrow_len*y_steps))
savefig("/home/csquires/Desktop/cvgm-glasso.png")

θ, alpha_final, alpha_path, gradient_path, loss_path = cvgm(samples, 0.04, glasso, nll, nll_deriv, K=20)
θ_path = [glasso(samples, α)[1] for α in alpha_path]
nll_path = [nll(heldout_samples, θ) for θ in θ_path]
xlabel!("λ")
ylabel!("Validation loss")
arrow_len = .02
quiver!(alpha_path, nll_path, quiver=(arrow_len*ones(length(alpha_path)), arrow_len*gradient_path))
savefig("/home/csquires/Desktop/cvgm-glasso.png")
