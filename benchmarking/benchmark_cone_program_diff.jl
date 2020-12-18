cd(@__DIR__)
cd("..")
using ConeProgramDiff
using MathOptInterface
const MOI = MathOptInterface
using DelimitedFiles


function rewrite_sdp_constraint(A, b, dim)
    # TODO: shuffle rows of A and b so that their order matches
end

function solve_and_time(folder, cones, num_programs)
    solve_times = []
    deriv_times = []
    adjoint_times = []
    for program_num in 0:(num_programs-1)
        program = ConeProgramDiff.cp_from_file("benchmarking/programs/$folder/$program_num.txt", dense=false)
        A, b, c = program[:A], program[:b], program[:c]

        start = time()
        x, y, s, pushforward, pullback = ConeProgramDiff.solve_and_diff(A, b, c, cones)
        push!(solve_times, time() - start)

        start = time()
        pushforward(zeros(size(A)), zeros(size(b)), ones(size(c)))
        push!(deriv_times, time() - start)

        start = time()
        pullback(ones(size(x)))
        push!(adjoint_times, time() - start)
    end
    writedlm("benchmarking/programs/$(folder)_cpd_solve_times.txt", solve_times)
    writedlm("benchmarking/programs/$(folder)_cpd_deriv_times.txt", deriv_times)
    writedlm("benchmarking/programs/$(folder)_cpd_adjoint_times.txt", adjoint_times)
end

println("=== BENCHMARKING ConeProgramDiff on SOC-small")
name = "soc-small"
cones = [
    MOI.Zeros(3),
    MOI.Nonnegatives(3),
    MOI.SecondOrderCone(3)
]
num_programs = 30
times = solve_and_time(name, cones, num_programs)

println("=== BENCHMARKING ConeProgramDiff on SOC-large")
name = "soc-large"
cones = [
    MOI.Zeros(3),
    MOI.Nonnegatives(3),
    MOI.SecondOrderCone(20)
]
num_programs = 2
solve_and_time(name, cones, num_programs)

# println("=== BENCHMARKING ConeProgramDiff on SDP-small")
# name = "sdp-small"
# cones = [
#     MOI.Zeros(5),
#     MOI.PositiveSemidefiniteConeTriangle(10)
# ]
# num_programs = 2
# solve_and_time(name, cones, num_programs)

# println("=== BENCHMARKING ConeProgramDiff on SDP-large")
# name = "sdp-large"
# cones = [
#     MOI.Zeros(25),
#     MOI.PositiveSemidefiniteConeTriangle(50)
# ]
# num_programs = 2
# solve_and_time(name, cones, num_programs)

println("=== BENCHMARKING ConeProgramDiff on exp-small")
name = "exponential-small"
cones = [
    [MOI.ExponentialCone() for i=1:2]...
]
num_programs = 2
solve_and_time(name, cones, num_programs)

println("=== BENCHMARKING ConeProgramDiff on exp-large")
name = "exponential-large"
cones = [
    [MOI.ExponentialCone() for i=1:20]...
]
num_programs = 2
solve_and_time(name, cones, num_programs)
