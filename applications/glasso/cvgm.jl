using Statistics
using JuMP
using LinearAlgebra
using ConeProgramDiff

function generate_partitions(samples; K=10, ρ=0.7)
    nsamples = size(samples, 1)
    num_training = Int64(ρ*nsamples)

    training_datasets = [samples[1:50, :] for i=1:K]
    test_datasets = [samples[51:100, :] for i=1:K]
    return training_datasets, test_datasets
end


function grid_search_cv(samples, alphas, alg, loss; K=10, ρ=0.7)
    training_datasets, test_datasets = generate_partitions(samples, K=K, ρ=ρ)

    test_losses = Array{Float64}(undef, length(alphas), K)
    for (fold_ix, (training_data, test_data)) in enumerate(zip(training_datasets, test_datasets))
        for (alpha_ix, alpha) in enumerate(alphas)
            p = alg(training_data, alpha)
            println(det(p))
            test_losses[alpha_ix, fold_ix] = loss(test_data, p)
        end
    end
end


function glasso(samples, α)
    Σ = cov(samples)
    A, b, c, cone_dict = write_glasso_cone_program(Σ, α)
    m,n = size(A)
    model = Model()
    set_optimizer(model, optimizer_with_attributes(SCS.Optimizer, "eps" => 1e-10, "max_iters" => 100000, "verbose" => 0))
    @variable(model, x[1:n])
    @variable(model, s[1:m])
    @objective(model, Min, c'*x)
    con = @constraint(model, A*x + s .== b)
    curr = 1
    for cone in cones
        dimension = MOI.dimension(cone)
        @constraint(model, s[curr:curr+dimension-1] in cone)
        curr += dimension
    end
    optimize!(model)

    sol = value.(x)
    p = size(Σ, 1)
    θ = sol[1:Int64(p*(p+1)/2)]
    return vec2sym(θ, p)
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
    return sum(test_cov .* θ) - log(det(θ))
end

samples = randn(100, 3)
alphas = [1., 2., 3.]
θ = glasso(samples, 1.)
nll(samples, θ)
grid_search_cv(samples, alphas, glasso, nll)
