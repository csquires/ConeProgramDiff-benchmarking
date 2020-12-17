using LinearAlgebra
using MathOptInterface
const MOI = MathOptInterface
using SymEngine
using IterTools


function vectorized_index(i, j)
    if i <= j
        k = div((j-1)*j, 2) + i
    else
        k = div((i-1)*i, 2) + j
    end
    return k
end


function triu_indices(A::Array)
    p = size(A, 1)
    return [(i, j) for i=1:p for j=i:p]
end


function triu_indices(p::Int)
    return [(i, j) for i=1:p for j=i:p]
end


function tril_indices(A::Array)
    p = size(A, 1)
    return [(i, j) for i=1:p for j=1:i]
end


function tril_indices(p::Int)
    return [(i, j) for i=1:p for j=1:i]
end


function _merge_psd_diag(θ_size, z_size, p)
    s_size = θ_size + 3*z_size - p
    A = zeros(s_size, θ_size+z_size)
    lower_triangle_indices = [(i, j) for i=1:p for j=1:i]
    upper_triangle_indices = [(i, j) for i=1:p for j=i:p]

    # === FILL UPPER LEFT BLOCK WITH Θ
    for (orig_i, orig_j) in tril_indices(p)
        old_index = vectorized_index(orig_i, orig_j)
        new_index = vectorized_index(orig_i, orig_j)
        A[new_index, old_index] = 1
    end

    # === FILL LOWER LEFT BLOCK WITH Z
    for (orig_i, orig_j) in triu_indices(p)
        old_index = vectorized_index(orig_i, orig_j) + θ_size
        new_index = vectorized_index(orig_i+p, orig_j)
        A[new_index, old_index] = 1
    end

    # === FILL LOWER RIGHT BLOCK WITH DIAG(Z)
    for orig_i in 1:p
        old_index = vectorized_index(orig_i, orig_i) + θ_size
        new_index = vectorized_index(orig_i+p, orig_i+p)
        A[new_index, old_index] = 1
    end

    return A
end


function write_glasso_cone_program(S, λ)
    p = size(S, 1)
    θ_size = Int64(p*(p+1)/2)
    z_size = θ_size
    m_size = θ_size

    A1 = -_merge_psd_diag(θ_size, z_size, p)

    # === EXPONENTIAL CONE CONSTRAINTS ON t_i's
    A2 = zeros(3p, z_size+p)
    b2 = zeros(3p)
    for d=1:p
        z_ix = vectorized_index(d, d)
        t_ix = z_size+d
        A2[d*3, z_ix] = -1
        b2[d*3-1] = 1
        A2[d*3-2, t_ix] = -1
    end

    # === EQUALITY (ZERO-CONE) CONSTRAINT ON t
    A3 = zeros(1, p+1)
    A3[1, 1:(end-1)] .= -1
    A3[1, end] = 1

    # === NONNEGATIVITY CONSTRAINT ON M
    A4 = zeros(2θ_size, θ_size + z_size + p + 1 + m_size)
    A4[1:θ_size, 1:θ_size] .= -I(θ_size)
    A4[1:θ_size, (end-θ_size+1):end] .= I(θ_size)
    A4[(θ_size+1):end, 1:θ_size] .= I(θ_size)
    A4[(θ_size+1):end, (end-θ_size+1):end] .= I(θ_size)

    # === STICK ALL CONSTRAINTS INTO A SINGLE MATRIX
    B1 = hcat(A1, zeros(size(A1, 1), p+1+m_size))
    B2 = hcat(zeros(size(A2, 1), θ_size), A2, zeros(size(A2, 1), 1+m_size))
    B3 = hcat(zeros(1, θ_size+z_size), A3, zeros(1, m_size))
    A = vcat(B1, B2, B3, -A4)

    b = zeros(size(A, 1))
    b[(size(A1, 1)+1):(size(A1, 1)+length(b2))] .= b2

    # === COST FUNCTION
    c = zeros(size(A, 2))
    m_offset = θ_size + z_size + p + 1
    for (i, j) in tril_indices(S)
        println((i, j))
        c[vectorized_index(i, j)] = (i == j ? S[i, j] : 2*S[i, j])
        c[vectorized_index(i, j) + m_offset] = (i == j ? λ : 2λ)
    end
    t_ix = θ_size + z_size + p + 1
    c[t_ix] = -1

    cones = [
        MOI.PositiveSemidefiniteConeTriangle(2p),
        [MOI.ExponentialCone() for i=1:p]...,
        MOI.Zeros(1),
        MOI.Nonnegatives(2θ_size)
    ]

    return A, b, c, cones
end

a = randn(100, 3)
S = cov(a)
λ = .1

A, b, c, cones = write_glasso_cone_program(S, λ)
x_symbolic = [symbols(a) for a in
    ["θ11", "θ21", "θ22", "θ31", "θ32", "θ33",
    "z11", "z21", "z22", "z31", "z32", "z33",
    "t1", "t2", "t3", "t",
    "m11", "m21", "m22", "m31", "m32", "m33"
    ]
]
s_symbolic = b - A * x_symbolic
s_symbolic[1:6]
s_symbolic[7:10]
s_symbolic[11:15]
s_symbolic[16:21]
s_symbolic[22:24]
s_symbolic[25:27]
s_symbolic[28:30]
s_symbolic[31]
s_symbolic[32:37]
s_symbolic[38:43]

c' * x_symbolic
using SCS
using JuMP

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

K = inv(S)
log(det(K))
sol = value.(x)
t = sol[16]
θ_est = sol[1:6]
θmat = vec2sym(θ_est, 3)
log(det(θmat))

using SCS
using Convex

X = Semidefinite(3)
problem = minimize(sum(X .* S) - logdet(X) + λ*norm(X, 1))
solve!(problem, SCS.Optimizer)
a = X.value
log(det(a))
