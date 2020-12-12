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
pb(cp[:c])
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
