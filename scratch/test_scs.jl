cd(@__DIR__)
cd("..")
pwd()
using ConeProgramDiff
using MathOptInterface
const MOI = MathOptInterface
using Profile
using SCS

program = ConeProgramDiff.cp_from_file("benchmarking/programs/soc-small/0.txt", dense=false)
cones = [
    MOI.Zeros(3),
    MOI.Nonnegatives(3),
    MOI.SecondOrderCone(3)
]
A, b, c = program[:A], program[:b], program[:c]

x, y, s, D, DT = ConeProgramDiff.solve_and_diff(A, b, c, cones)
dA = zeros(size(A))
db = zeros(size(b))
dc = ones(size(c))
Profile.clear()
@profile for i=1:300; D(dA, db, dc); end
# @profile for i=1:100; ConeProgramDiff.solve_and_diff(A, b, c, cones); end
Juno.profiler()
