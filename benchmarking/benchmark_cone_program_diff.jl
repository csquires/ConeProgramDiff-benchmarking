cd(@__DIR__)
cd("..")
pwd()
using ConeProgramDiff
using MathOptInterface
const MOI = MathOptInterface
using DelimitedFiles
using ProgressMeter
using Profile


function solve_and_time(folder, cones, num_programs; psd_indices=nothing, psd_dim=nothing)
    solve_times = []
    deriv_times = []
    adjoint_times = []

    burn_in = 2
    @showprogress for program_num in 0:(num_programs-1)
        program = ConeProgramDiff.cp_from_file("benchmarking/programs/$folder/$program_num.txt", dense=false)
        A, b, c = program[:A], program[:b], program[:c]
        ConeProgramDiff.rewrite_sdps_to_col_major!(A, b, cones)

        start = time()
        x, y, s, pushforward, pullback = ConeProgramDiff.solve_and_diff(A, b, c, cones)
        if program_num >= burn_in
            push!(solve_times, time() - start)
        end

        start = time()
        pushforward(zeros(size(A)), zeros(size(b)), ones(size(c)))
        if program_num >= burn_in
            push!(deriv_times, time() - start)
        end

        start = time()
        pullback(ones(size(x)))
        if program_num >= burn_in
            push!(adjoint_times, time() - start)
        end
    end
    writedlm("benchmarking/programs/$(folder)_cpd_solve_times.txt", solve_times)
    writedlm("benchmarking/programs/$(folder)_cpd_deriv_times.txt", deriv_times)
    writedlm("benchmarking/programs/$(folder)_cpd_adjoint_times.txt", adjoint_times)
end

# === SOC SMALL ===
println("=== BENCHMARKING ConeProgramDiff on SOC-small")
name = "soc-small"
cones = [
    MOI.Zeros(3),
    MOI.Nonnegatives(3),
    MOI.SecondOrderCone(3)
]
num_programs = 30
times = solve_and_time(name, cones, num_programs)


# === SOC LARGE ===
println("=== BENCHMARKING ConeProgramDiff on SOC-large")
name = "soc-large"
cones = [
    MOI.Zeros(3),
    MOI.Nonnegatives(3),
    MOI.SecondOrderCone(20)
]
num_programs = 30
solve_and_time(name, cones, num_programs)

# === SDP SMALL ===
println("=== BENCHMARKING ConeProgramDiff on SDP-small")
name = "sdp-small"
cones = [
    MOI.Zeros(5),
    MOI.PositiveSemidefiniteConeTriangle(10)
]
num_programs = 30
k = Int64(10*11/2)
solve_and_time(name, cones, num_programs, psd_indices=6:(5+k), psd_dim=10)

# === SDP LARGE ===
println("=== BENCHMARKING ConeProgramDiff on SDP-large")
name = "sdp-large"
n = 20
p = 10
cones = [
    MOI.Zeros(p),
    MOI.PositiveSemidefiniteConeTriangle(n)
]
num_programs = 30
k = Int64(n*(n+1)/2)
solve_and_time(name, cones, num_programs, psd_indices=(p+1):(p+k), psd_dim=n)
#
# === EXPONENTIAL SMALL ===
println("=== BENCHMARKING ConeProgramDiff on exp-small")
name = "exponential-small"
cones = [
    [MOI.ExponentialCone() for i=1:2]...
]
num_programs = 30
solve_and_time(name, cones, num_programs)

# # === EXPONENTIAL LARGE ===
# println("=== BENCHMARKING ConeProgramDiff on exp-large")
# name = "exponential-large"
# cones = [
#     [MOI.ExponentialCone() for i=1:4]...
# ]
# num_programs = 30
# solve_and_time(name, cones, num_programs)
