using ConeProgramDiff
using JuMP
using MathOptInterface
using Random
const MOI = MathOptInterface

function test_l1_problem(tol=1e-6)
    Random.seed!(0)
    l1_prob = ConeProgramDiff.l1_minimization_program
    m,n = 20, 10
    A, b, c, cones, model = l1_prob((m,n); solve=true)
    xs = value.(all_variables(model))

    x, y, s, pf, pb  = solve_and_diff(A, b, c, cones)

    isapprox(x, xs, atol=tol)
    # Objective
    @assert isapprox(dot(c,x), -dot(b,y), atol=tol)
    # Check Primal
    @assert isapprox(b - A*x, s, atol=tol)
    s_proj = ConeProgramDiff.project_onto_cone(s, cones)
    @assert isapprox(s, s_proj, atol=tol)
    # Check dual
    y_proj = ConeProgramDiff.project_onto_cone(y, [MOI.dual_set(c) for c in cones])
    @assert isapprox(y, y_proj, atol=tol)
end

test_l1_problem(1e-8)
