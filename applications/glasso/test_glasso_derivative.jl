include("glasso_conic.jl")

samples = randn(100, 3)
Σ = cov(samples)
α0 = .1
A, b, c, cone_dict = write_glasso_cone_program(Σ, α0)
x, y, s, D, DT = solve_and_diff(A, b, c, cone_dict)

dA = zeros(size(A))
db = zeros(size(b))
dc = zeros(size(c))

D(dA, db, dc)
