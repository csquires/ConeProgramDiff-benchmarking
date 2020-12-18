using Pkg
Pkg.activate("ConeProgramDiff")
include("glasso_conic.jl")
using ConeProgramDiff
using Statistics

samples = randn(100, 3)
Σ = cov(samples)
α0 = .1
A, b, c, cone_dict = write_glasso_cone_program(Σ, α0)
x, y, s, D, DT = solve_and_diff(A, b, c, cone_dict)
ConeProgramDiff.d_project_onto_cone((y-s)[1:21], [cone_dict[1]])


A_ = copy(A)
b_ = copy(b)
c_ = copy(c)
cone_dict_ = copy(cone_dict)

dA = zeros(size(A))
db = zeros(size(b))
dc = zeros(size(c))

cone_dict
size(A)
length(x)
length(s)
length(y)
ConeProgramDiff.dpi_z(x, y-s, 1.0, cone_dict)
D(dA, db, dc)

using MathOptSetDistances

const MOSD = MathOptSetDistances

MOSD.projection_gradient_on_set(MOSD.DefaultDistance(), (y-s)[1:21], MOI.PositiveSemidefiniteConeTriangle(6))
