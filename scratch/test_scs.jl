cd(@__DIR__)
cd("..")
pwd()
using ConeProgramDiff
using MathOptInterface
const MOI = MathOptInterface
using Profile
using SCS

Profile.clear()
program = ConeProgramDiff.cp_from_file("benchmarking/programs/soc-small/0.txt", dense=false)
cones = [
    MOI.Zeros(3),
    MOI.Nonnegatives(3),
    MOI.SecondOrderCone(3)
]
A, b, c = program[:A], program[:b], program[:c]
A_, b_, cones2 = ConeProgramDiff.reorder_opt_problem_scs(A, b, cones)
m, n = size(A)
solver = SCS.DirectSolver()
SCS_solve(solver, m, n, A_, b_, c, f=3, l=3, q=[3])
SCS_solve(solver, m, n, A_, reshape(b_, 9, 1), reshape(c, 5, 1), f=3, l=3, q=[3])
SCS_solve(solver, m, n, A_, reshape(b_, 9, 1), reshape(c, 5, 1), f=3, l=3, q=[3], s=[], ep=0, ed=0, p=[])
SCS_solve(solver, m, n, A_, b_, c, 3, 3, [3], [], 0, 0, [])
x, y, s, pushforward, pullback, sol = @profile ConeProgramDiff.solve_and_diff(A, b, c, cones)
Juno.profiler()
