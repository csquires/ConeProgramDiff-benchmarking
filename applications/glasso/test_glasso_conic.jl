include("glasso_conic.jl")
using SymEngine

S = cov(randn(100, 3))
λ = .1

A, b, c, cones = write_glasso_cone_program(S, λ)

# === CHECK CONSTRAINTS AND COST FUNCTION SYMBOLICALLY
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

c'
c' * x_symbolic


# === COMPARE SOLUTION TO (1) inverse, when no penalty, and (2) non-standard form of the program
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

# True inverse
true_K = inv(S)
log(det(true_K))

# === Estimated solution
sol = value.(x)
t = sol[16]  # should equal log det
θ_est = sol[1:6]
θmat = vec2sym(θ_est, 3)
log(det(θmat))

# === Program in original (non-standard) form
using Convex

X = Semidefinite(3)
problem = minimize(sum(X .* S) - logdet(X) + λ*norm(X, 1))
solve!(problem, SCS.Optimizer)
a = X.value
log(det(a))
