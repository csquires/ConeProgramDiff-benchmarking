using ConeProgramDiff


using MathOptInterface
using SCS
using Random
Random.seed!(0)
const MOI = MathOptInterface

## Projection Test Functions
function test_proj_zero()
    Random.seed!(0)
    n = 100
    for _ in 1:10
        x = randn(n)
        @assert zeros(n) ≈ ConeProgramDiff._proj(x, MOI.Zeros(n))
        @assert x ≈ ConeProgramDiff._proj(x, MOI.dual_set(MOI.Zeros(n)))
    end
end

x
ConeProgramDiff._proj(x, MOI.dual_set(MOI.Zeros(n)))

function test_proj_pos()
    Random.seed!(0)
    n = 100
    for _ in 1:10
        x = randn(n)
        @assert max.(x,zeros(n)) ≈ ConeProgramDiff._proj(x, MOI.Nonnegatives(n))
        @assert max.(x,zeros(n)) ≈ ConeProgramDiff._proj(x, MOI.dual_set(MOI.Nonnegatives(n)))
    end
end


function test_proj_soc()
    Random.seed!(0)
    n = 100
    for _ in 1:10
        x = randn(n)
        model = Model()
        set_optimizer(model, optimizer_with_attributes(SCS.Optimizer, "eps" => 1e-8, "max_iters" => 10000, "verbose" => 0))
        @variable(model, z[1:n])
        @variable(model, t)
        @objective(model, Min, t)
        @constraint(model, sum((x-z).^2) <= t)
        @constraint(model, z in MOI.SecondOrderCone(n))
        optimize!(model)

        p = ConeProgramDiff._proj(x, MOI.SecondOrderCone(n))
        @assert p ≈ value.(z)
        @assert p ≈ ConeProgramDiff._proj(x, MOI.dual_set(MOI.SecondOrderCone(n)))
    end
end


# TODO: Fix PSD cone -- Triangle vs square matrix
function test_proj_psd()
    Random.seed!(0)
    n = 10
    for _ in 1
        x = randn(n,n)
        x = x + x'
        x_vec = ConeProgramDiff.vec_symm(x)
        model = Model()
        set_optimizer(model, optimizer_with_attributes(SCS.Optimizer, "eps" => 1e-8, "max_iters" => 10000, "verbose" => 0))
        @variable(model, z[1:n,1:n])
        @variable(model, t)
        @objective(model, Min, t)
        @constraint(model, sum((x-z).^2) <= t)
        @constraint(model, z in PSDCone())
        optimize!(model)

        z_star = ConeProgramDiff.vec_symm(value.(z))
        p = ConeProgramDiff._proj(x_vec, MOI.PositiveSemidefiniteConeTriangle(n))
        @assert p ≈ z_star
        @assert p ≈ ConeProgramDiff._proj(x_vec, MOI.dual_set(MOI.PositiveSemidefiniteConeTriangle(n)))
    end
end

# TODO: if pass matrix intro MOSD.projection for PSD cone, get StackOverflowError

## dProjection Test Functions

function _test_d_proj(cone; tol=1e-8)
    n = MOI.dimension(cone)
    x = randn(n)
    dx = 1e-6 * randn(n)
    proj_x = ConeProgramDiff._proj(x, cone)
    proj_xdx = ConeProgramDiff._proj(x+dx, cone)
    dproj_finite_diff = proj_xdx - proj_x

    dproj_test = ConeProgramDiff._d_proj(x, cone) * dx
    @assert isapprox(dproj_finite_diff, dproj_test, atol=tol)

end


function test_d_proj(cone::MOI.AbstractVectorSet)
    Random.seed!(0)
    for _ in 1:10
        _test_d_proj(cone)
        _test_d_proj(MOI.dual_set(cone))
    end
end


function test_d_proj(n::Int)
    d_proj_cones = [
        MOI.Zeros(n),
        MOI.Nonnegatives(n),
        MOI.SecondOrderCone(n),
        MOI.PositiveSemidefiniteConeTriangle(n)
    ]
    for cone in cones
        test_d_proj(cone)
    end
end


## Testing Projections
test_proj_zero()
test_proj_pos()
test_proj_soc()
test_proj_psd()
test_d_proj(10)


## Test pi and project_onto_cone functions
# project_onto_cone
# d_project_onto_cone

function get_random_product_cone(n)
    zero_dim = rand(1:n)
    nonneg_dim = rand(1:n)
    soc_dim = rand(1:n)
    psd_dim = rand(1:n)
    cone_prod = [MOI.Zeros(zero_dim), MOI.Nonnegatives(nonneg_dim),
             MOI.SecondOrderCone(soc_dim), MOI.PositiveSemidefiniteConeTriangle(psd_dim)]
    return cone_prod
end


function test_project_onto_cone()
    function _test_proj_all_cones(x, cone_prod; dual=false)
        cones = dual ? [MOI.dual_set(c) for c in cone_prod] : cone_prod
        xp = project_onto_cone(x, cones)
        offset = 0
        for cone in cones
            inds = 1+offset:offset+MOI.dimension(cone)
            @assert xp[inds] ≈ ConeProgramDiff._proj(x[inds], cone)
            offset += MOI.dimension(cone)
        end
    end

    for _ in 1:10
        cone_prod = get_random_product_cone(10)
        N = sum([MOI.dimension(c) for c in cone_prod])
        local x = randn(N)
        _test_proj_all_cones(x, cone_prod; dual=false)
        _test_proj_all_cones(x, cone_prod; dual=true)
    end
end


function test_d_project_onto_cone()
    function _test_d_proj_all_cones(x, cone_prod; dual=false)
        cones = dual ? [MOI.dual_set(c) for c in cone_prod] : cone_prod
        dx = 1e-7 * randn(length(x))
        Dpi = d_project_onto_cone(x, cones)
        proj_x = ConeProgramDiff.project_onto_cone(x, cones)
        proj_xdx = ConeProgramDiff.project_onto_cone(x+dx, cones)
        dproj_finite_diff = proj_xdx - proj_x
        println(diag(Dpi))
        dproj_test = Dpi * dx
        @assert isapprox(dproj_finite_diff, dproj_test, atol=tol)
        # if ~isapprox(dproj_finite_diff, dproj_test, atol=tol)
        #     println("Test failed:")
        #     println("determinant: $(det(Dpi))")
        #     inds = abs.(dproj_finite_diff - dproj_test) .>= tol
        #     println((dproj_finite_diff - dproj_test)[inds])
        #     println()
        # end
    end

    for _ in 1:10
        cone_prod = get_random_product_cone(10)
        N = sum([MOI.dimension(c) for c in cone_prod])
        local x = randn(N)
        _test_d_proj_all_cones(x, cone_prod; dual=false)
        _test_d_proj_all_cones(x, cone_prod; dual=true)
    end
end


## Testing projection and derivative onto cone product
tol
test_project_onto_cone()
test_d_project_onto_cone()


## Test Solve and Derivative function
