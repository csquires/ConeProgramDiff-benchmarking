using ConeProgramDiff


using MathOptInterface
using SCS
using Random

const MOI = MathOptInterface


function check_kkt(A, b, c, cone_prod, x, y, s; tol=1e-8)
    @assert isapprox(A*x + s, b,atol=tol)
    @assert isapprox(A'*y + c, zeros(length(c)),atol=tol)
    @assert isapprox(s, project_onto_cone(s, cone_prod),atol=tol)
    @assert isapprox(y, project_onto_cone(y, [MOI.dual_set(c) for c in cone_prod]),atol=tol)
    @assert isapprox(s'*y, 0, atol=tol)
end


function check_adjoint()
    Random.seed!(0)
    dims = (15, 10)
    A, b, c, cone_prod = ConeProgramDiff.l1_minimization_program(dims; solve=false)
    x, y, s, pf, pb = solve_and_diff(A, b, c, cone_prod)
    @assert check_kkt(A, b, c, cone_prod, x, y, s, tol=1e-6)


    # Check adjoint
    pstar = dot(x, c)
    dA, db, dc = pb(c)
    del = 1e-6
    x_pert, _, _, _, _ = solve_and_diff(A+del*dA, b+del*db, c+del*dc, cone_prod)
    pstar_pert = dot(x_pert, c)
    df = pstar_pert - pstar
    dp = del * (sum(dA .* dA) + db'*db + dc'*dc)
    @assert isapprox(df, dp, atol=1e-6)
end


function check_derivative()
    Random.seed!(0)
    dims = (15, 10)
    A, b, c, cone_prod = ConeProgramDiff.l1_minimization_program(dims; solve=false)
    x, y, s, pf, pb = solve_and_diff(A, b, c, cone_prod)
    check_kkt(A, b, c, cone_prod, x, y, s, tol=1e-6)

    # Check derivative
    del = 1e-8
    dA, db, dc = del*randn(size(A)), del*randn(size(b)), del*randn(size(c))
    dx, dy, ds = pf(dA, db, dc)
    x_pert, y_pert, s_pert, _, _ = solve_and_diff(A+dA, b+db, c+dc, cone_prod)
    @assert isapprox(x_pert - x, dx, atol=1e-4)
    @assert isapprox(y_pert - y, dy, atol=1e-4)
    @assert isapprox(s_pert - s, ds, atol=2e-4)
end


check_adjoint()
check_derivative()
