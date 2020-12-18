using ConeProgramDiff
using MathOptInterface
using SCS, JuMP
using Random
using BenchmarkTools
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

function _test_d_proj(cone, tol)
    n = MOI.dimension(cone)
    x = randn(n)
    dx = 1e-6 * randn(n)
    proj_x = ConeProgramDiff._proj(x, cone)
    proj_xdx = ConeProgramDiff._proj(x+dx, cone)
    dproj_finite_diff = proj_xdx - proj_x

    dproj_test = ConeProgramDiff._d_proj(x, cone) * dx
    if !isapprox(dproj_finite_diff, dproj_test, atol=tol)
        println("Error!")
        error("Error for x = $x, dx = $dx, $cone")
    end
end


function test_d_proj(cone::MOI.AbstractVectorSet; tol=1e-8)
    Random.seed!(0)
    for _ in 1:100
        _test_d_proj(cone, tol)
        _test_d_proj(MOI.dual_set(cone), tol)
    end
end


function test_d_proj(n::Int; tol=tol)
    cones = [
        MOI.Zeros(n),
        MOI.Nonnegatives(n),
        MOI.SecondOrderCone(n),
        MOI.PositiveSemidefiniteConeTriangle(n)
    ]
    for cone in cones
        println("Testing $cone")
        test_d_proj(cone; tol=tol)
    end
end


function test_d_proj_exp(tol)
    function _helper(x, dx, tol; dual)
        proj_x = ConeProgramDiff._proj_exp_cone(x,  dual=dual)
        proj_xdx = ConeProgramDiff._proj_exp_cone(x+dx,  dual=dual)
        dproj_finite_diff = proj_xdx - proj_x
        dproj_test = ConeProgramDiff._d_proj_exp_cone(x; dual=dual) * dx
        # println(dproj_finite_diff)
        # println(dproj_test)
        @assert isapprox(dproj_finite_diff, dproj_test, atol=tol)
    end

    Random.seed!(0)
    case_p = zeros(4)
    case_d = zeros(4)
    for _ in 1:100
        x = randn(3)
        dx = 1e-6 * randn(3)
        case_p[det_case_exp_cone(x; dual=false)] += 1
        # println(det_case_exp_cone(x; dual=false))
        _helper(x, dx, tol; dual=false)

        case_d[det_case_exp_cone(x; dual=true)] += 1
        # println(det_case_exp_cone(x; dual=true))
        # println(x)
        _helper(x, dx, tol; dual=true)
    end
    @assert all(case_p .> 0) && all(case_d .> 0)
end



## Testing Projections
test_proj_zero()
test_proj_pos()
test_proj_soc()
test_proj_psd()
test_d_proj(10; tol=1e-10)
test_d_proj_exp(1e-5)


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


## Test Exponential Cone
function det_case_exp_cone(v; dual=false)
    v = dual ? -v : v
    if ConeProgramDiff.in_exp_cone(v)
        return 1
    elseif ConeProgramDiff.in_exp_cone_dual(v)
        return 2
    elseif v[1] <= 0 && v[2] <= 0 #TODO: threshold here??
        return 3
    else
        return 4
    end
end

function test_proj_exp(tol)
    function _test_proj_exp_cone_help(x, tol; dual=false)
        cone = dual ? MOI.DualExponentialCone() : MOI.ExponentialCone()
        model = Model()
        set_optimizer(model, optimizer_with_attributes(SCS.Optimizer, "eps" => 1e-8, "max_iters" => 10000, "verbose" => 0))
        @variable(model, z[1:3])
        @variable(model, t)
        @objective(model, Min, t)
        @constraint(model, sum((x-z).^2) <= t)
        @constraint(model, z in cone)
        optimize!(model)
        z_star = value.(z)
        @assert isapprox(ConeProgramDiff._proj_exp_cone(x,  dual=dual), z_star, atol=tol)
    end

    Random.seed!(0)
    n = 3
    case_p = zeros(4)
    case_d = zeros(4)
    for _ in 1:100
        # x = ConeProgramDiff._proj_exp_cone(x) + randn(3)
        x = randn(3)
        # println(x)
        case_p[det_case_exp_cone(x; dual=false)] += 1
        _test_proj_exp_cone_help(x, tol; dual=false)

        case_d[det_case_exp_cone(x; dual=true)] += 1
        _test_proj_exp_cone_help(x, tol; dual=true)
    end
    @assert all(case_p .> 0) && all(case_d .> 0)
end

test_proj_exp(1e-7)


function det_case_pow_cone(v, α; dual=false)
    v = dual ? -v : v
    x, y, z = v

    if ConeProgramDiff.in_pow_cone(x, y, z, α)
        # if in power cone
        return 1
    elseif ConeProgramDiff.in_pow_cone_polar(x, y, z, α)
        # if in polar cone Ko = -K*
        return 2
    elseif abs(z) <= 1e-6
        # if not in K, Ko and z = 0
        return 3
    else
        return 4
    end
end

function test_proj_pow_cone(tol)
    function _test_proj_pow_cone_help(x, α, tol; dual=false)
        cone = dual ? MOI.DualPowerCone(α) : MOI.PowerCone(α)
        model = Model()
        set_optimizer(model, optimizer_with_attributes(
            SCS.Optimizer, "eps" => 1e-8, "max_iters" => 10000,
            "verbose" => 0, "scale" => 10.0))
        @variable(model, z[1:3])
        @variable(model, t)
        @objective(model, Min, t)
        @constraint(model, sum((x-z).^2) <= t)
        @constraint(model, z in cone)
        optimize!(model)
        z_star = value.(z)
        xp = ConeProgramDiff._proj_pow_cone(x, α, dual=dual)
        if !isapprox(xp, z_star, atol=tol)
            error("x = $x,\nz* = $z_star,\nxp = $xp,\ncase = $(det_case_pow_cone(x, α; dual=dual))")
        end
    end
    Random.seed!(0)
    n = 3
    case_p = zeros(4)
    case_d = zeros(4)
    for _ in 1:100
        x = randn(3)
        α = rand()*0.6 + 0.2
        # α = rand()

        # Need to get some into case 3
        if rand(1:10) == 1
            x[3] = 0
        end
        case_p[det_case_pow_cone(x, α; dual=false)] += 1
        _test_proj_pow_cone_help(x, α, tol; dual=false)

        case_d[det_case_pow_cone(x, α; dual=true)] += 1
        _test_proj_pow_cone_help(x, α, tol; dual=true)
    end

    @assert all(case_p .> 0) && all(case_d .> 0)
end


function test_d_proj_pow(tol)
    function _helper(x, dx, tol; dual)
        # println("$x \t $dx \t $(x+dx)")
        proj_x = ConeProgramDiff._proj_pow_cone(x, α; dual=dual)
        proj_xdx = ConeProgramDiff._proj_pow_cone(x+dx, α; dual=dual)
        dproj_finite_diff = proj_xdx - proj_x
        dproj_test = ConeProgramDiff._d_proj_pow_cone(x, α; dual=dual) * dx
        # println(dproj_finite_diff)
        # println(dproj_test)
        @assert isapprox(dproj_finite_diff, dproj_test, atol=tol)
    end

    Random.seed!(0)
    case_p = zeros(4)
    case_d = zeros(4)
    for _ in 1:100
        x = randn(3)
        dx = 1e-6 * randn(3)
        α = rand()*0.6 + 0.2

        # Need to get some into case 3
        if rand(1:10) == 1
            x[3] = 0
        end

        case_p[det_case_pow_cone(x, α; dual=false)] += 1
        _helper(x, dx, tol; dual=false)

        case_d[det_case_pow_cone(x, α; dual=true)] += 1
        _helper(x, dx, tol; dual=true)
    end
    @assert all(case_p .> 0) && all(case_d .> 0)
end


test_proj_pow_cone(1e-6)
test_d_proj_pow(1e-6)

x = rand(3)
α = 0.5
det_case_pow_cone(x, α; dual=false)
@time ConeProgramDiff._d_proj_pow_cone(x, α, dual=false)

@time test_proj_pow_cone(1e-6)
@time test_d_proj_pow(1e-6)

det_case_exp_cone(x; dual=false)
@btime ConeProgramDiff._proj_exp_cone(x, dual=false)
