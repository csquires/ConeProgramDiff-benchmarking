cd(@__DIR__)
cd("..")
pwd()
using ConeProgramDiff
using MathOptInterface
const MOI = MathOptInterface
using DelimitedFiles
using ProgressMeter
using Profile

function vectorized_index(i, j)
    if i <= j
        k = div((j-1)*j, 2) + i
    else
        k = div((i-1)*i, 2) + j
    end
    return k
end

function rewrite_sdp_constraint(A, b, dim)
    println("Rewriting constraint")
    new_ixs = []
    for i=1:dim
        for j=i:dim
            k = vectorized_index(i, j)
            push!(new_ixs, k)
        end
    end
    new_ixs = sortperm(new_ixs)
    return A[new_ixs, :], b[new_ixs]
end


function solve_and_time(folder, cones, num_programs; psd_indices=nothing, psd_dim=nothing)
    solve_times = []
    deriv_times = []
    adjoint_times = []
    @showprogress for program_num in 0:(num_programs-1)
        program = ConeProgramDiff.cp_from_file("benchmarking/programs/$folder/$program_num.txt", dense=false)
        A, b, c = program[:A], program[:b], program[:c]
        if ~isnothing(psd_indices)
            A_rewrite, b_rewrite = rewrite_sdp_constraint(A[psd_indices, :], b[psd_indices], psd_dim)
            A[psd_indices, :] .= A_rewrite
            b[psd_indices] .= b_rewrite
        end

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

# === SOC SMALL ===
Profile.clear()
println("=== BENCHMARKING ConeProgramDiff on SOC-small")
name = "soc-small"
cones = [
    MOI.Zeros(3),
    MOI.Nonnegatives(3),
    MOI.SecondOrderCone(3)
]
num_programs = 30
times = @profile solve_and_time(name, cones, num_programs)
Juno.profiler()


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

# === EXPONENTIAL SMALL ===
println("=== BENCHMARKING ConeProgramDiff on exp-small")
name = "exponential-small"
cones = [
    [MOI.ExponentialCone() for i=1:2]...
]
num_programs = 30
solve_and_time(name, cones, num_programs)

# === EXPONENTIAL LARGE ===
println("=== BENCHMARKING ConeProgramDiff on exp-large")
name = "exponential-large"
cones = [
    [MOI.ExponentialCone() for i=1:4]...
]
num_programs = 30
solve_and_time(name, cones, num_programs)
