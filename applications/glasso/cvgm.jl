using Statistics
using JuMP
using LinearAlgebra
using ConeProgramDiff
using Random
using SCS

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


function cvgm(samples, α0, alg, loss, loss_deriv; K=10, ρ=0.7, max_iters=1000)
    num_iter = 0
    αt = α0

    alpha_path = []
    gradient_path = []
    while num_iter < max_iters
        num_iter += 1

        gradients = []
        for (training_data, test_data) in generate_partitions(samples, K=K, ρ=ρ)
            θt, D = alg(training_data, αt)
            dθ_dα = D(αt)
            dloss_dθ = loss_deriv(test_data, θt)
            dloss_dα = sum(dθ_dα .* dloss_dθ)
            push!(gradients, dloss_dα)
        end
        avg_gradient = mean(gradients)
        push!(gradient_path, avg_gradient)
        push!(alpha_path, αt)
    end
    θ = alg(training_data, αt)
    return θ, αt, alpha_path, gradient_path
end


function glasso(samples, α)
    Σ = cov(samples)
    println(Σ)

    println("Writing as cone program")
    A, b, c, cone_dict = write_glasso_cone_program(Σ, α)

    println("Optimizing")
    x, y, s, D, DT = solve_and_diff(A, b, c, cone_dict)

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
        println(size(A), size(b), size(c), size(dA), size(db), size(dc))
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
        println("negative determinant", d)
    end
    return sum(test_cov .* θ) - log(d)
end

function nll_deriv(samples, θ)
    test_cov = cov(samples)
    return test_cov - inv(θ)
end

samples = randn(100, 3)
alphas = [.1, .2, .3]
# θ = glasso(samples, 0)
# nll(samples, θ)
# θ, best_alpha, test_losses = grid_search_cv(samples, alphas, glasso, nll, K=2)


θ, best_alpha, test_losses = cvgm(samples, 0.1, glasso, nll, nll_deriv, K=2)
