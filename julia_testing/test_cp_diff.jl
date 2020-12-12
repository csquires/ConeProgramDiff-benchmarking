using ConeProgramDiff


using MathOptInterface
using Random
Random.seed!(0)
const MOI = MathOptInterface
cone_prod = [
    MOI.Zeros(3),
    MOI.Nonnegatives(3),
    MOI.SecondOrderCone(5)
]

m,n = sum([MOI.dimension(c) for c in cone_prod]),5
cp = random_cone_program((m,n), cone_prod)


x, y, s, pf, pb = solve_and_diff(cp[:A], cp[:b], cp[:c], cone_prod)
@time pb(cp[:c], zeros(m), zeros(m))
@time pf(cp[:A], cp[:b], cp[:c])
ConeProgramDiff._d_proj(ones(2), MOI.Reals(2))
ConeProgramDiff.d_project_onto_cone(ones(5), [MOI.Reals(2), MOI.SecondOrderCone(3)])
x ≈ cp[:x_star]
y
z



all([typeof(c) <: ConeProgramDiff.SUPPORTED_INPUT_SETS for c in cone_prod])
typeof(cone_prod[1]) <: ConeProgramDiff.SUPPORTED_INPUT_SETS
for cone in cone_prod
    println(MOI.dimension(cone)-1)
end
# pb(cp[:c], zeros(m), zeros(m))

curr = 1
cone = cone_prod[3]
curr += MOI.dimension(cone)
@view(y[curr:curr+MOI.dimension(cone)-1])
for cone in cone_prod
    println(cone)
    x_curr = @view()
    pi_x[curr:curr+MOI.dimension(cone)-1] .= _proj(x_curr, cone)
    curr += MOI.dimension(cone)
end
cone_prod

# m, n = 3, 4
# deriv = Dict(
#     :dA => randn(m,n),
#     :db => randn(m),
#     :dc => randn(n),
#     :dx => randn(n),
#     :dy => randn(m),
#     :ds => randn(m)
# )
# adjoint = deepcopy(deriv)
# derivatives_to_file("deriv.txt", deriv, adjoint)
# d, a = derivatives_from_file("deriv.txt")
# all([d[k] == deriv[k] for k in keys(d)])

# A = sparse(randn(4,3))
# b = randn(4)
# c = randn(3)

# cp_to_file("test.txt", (A, b, c), dense=false)
# folder = "/home/csquires/Desktop"
# program_sparse = cp_from_file("$folder/test_sparse_py.txt", dense=false)
# program_dense = cp_from_file("$folder/test_dense_py.txt", dense=true)
# sparse_params = (program_sparse[:A], program_sparse[:b], program_sparse[:c])
# sparse_optvals = (program_sparse[:x_star], program_sparse[:y_star], program_sparse[:s_star])
# cp_to_file("$folder/test_sparse_jl.txt", sparse_params, opt_vals=sparse_optvals, dense=false)
# dense_params = (program_dense[:A], program_dense[:b], program_dense[:c])
# dense_optvals = (program_dense[:x_star], program_dense[:y_star], program_dense[:s_star])
# cp_to_file("$folder/test_dense_jl.txt", dense_params, opt_vals=dense_optvals, dense=true)

# _cp_to_file("nothing.txt", (A, b, c), opt_vals=(nothing, nothing, nothing), dense=false)
# # cp_from_file("test.txt", dense=false)
# iterate((nothing, nothing, nothing))
