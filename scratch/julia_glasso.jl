using SCS
using Convex
using LinearAlgebra

x = Semidefinite(3)
A = randn(100, 3)
C = A' * A / 100
problem = minimize(sum(x .* C) - logdet(x))
solve!(problem, SCS.Optimizer)
xval = evaluate(x)
inv(C)
