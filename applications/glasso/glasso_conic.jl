using LinearAlgebra
using MathOptInterface
const MOI = MathOptInterface


function _merge_psd_diag(θ_size, z_size, p)
    s_size = θ_size + 3*z_size - p
    A = zeros(s_size, θ_size+z_size)

    # === FILL UPPER LEFT BLOCK WITH Θ
    i, j = 1, 1
    for k=p:-1:1
        for k_=1:k
            A[i, j] = 1
            i += 1
            j += 1
        end
        i += p
    end

    # === FILL BOTTOM LEFT BLOCK WITH Z
    i, j = p+1, θ_size+p
    for d=1:p
        zd_ix = θ_size + d
        A[i, zd_ix] = 1
        for d_=1:(d-1)
            A[i-d_, j] = 1
            j += 1
        end
        i += (2p - d + 1)
    end

    # === FILL BOTTOM RIGHT BLOCK WITH DIAGONAL OF Z
    i = Int64(p^2 + p*(p-1)/2) + 1
    for d=1:p
        z_ix = θ_size + d
        A[i, z_ix] = 1
        i += (p-d)
    end

    return A
end


function write_glasso_cone_program(S, λ)
    p = size(S, 1)
    θ_size = Int64(p*(p+1)/2)
    z_size = θ_size
    m_size = θ_size

    A1 = _merge_psd_diag(θ_size, z_size, p)

    # === EXPONENTIAL CONE CONSTRAINTS ON t_i's
    A2 = zeros(3p, z_size+p)
    b2 = zeros(3p)
    for d=1:p
        z_ix = d
        t_ix = z_size+d
        A2[d*3, z_ix] = -1
        b2[d*3-1] = 1
        A2[d*3-2, t_ix] = -1
    end

    # === EQUALITY (ZERO-CONE) CONSTRAINT ON t
    A3 = zeros(1, p+1)
    A3[1, 1:(end-1)] .= -1
    A3[1, end-1] = 1

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
    c[1:θ_size] .= S[tril(trues(size(S)))]
    c[(end-m_size+1):end] .= λ
    c[θ_size + z_size + p] = -1

    cones = [
        MOI.PositiveSemidefiniteConeTriangle(2p),
        [MOI.ExponentialCone() for i=1:p]...,
        MOI.Zeros(1),
        MOI.Nonnegatives(2θ_size)
    ]

    return A, b, c, cones
end

a = randn(100, 3)
S = a' * a

A, b, c, cones = write_glasso_cone_program(S, 1)

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
    if (typeof(cone) <: MOI.PositiveSemidefiniteConeTriangle)
        dimension = Int64(cone.side_dimension * (cone.side_dimension + 1) / 2)
    elseif (typeof(cone) <: MOI.ExponentialCone)
        dimension = 3
    else
        dimension = cone.dimension
    end
    println(dimension)
    @constraint(model, s[curr:curr+dimension-1] in cone)
    curr += dimension
end
optimize!(model)

sol = value.(x)
θ_est = sol[1:6]
