using ConeProgramDiff
using MathOptInterface
using SCS, JuMP
using Random
using BenchmarkTools
using LinearAlgebra, SparseArrays
const MOI = MathOptInterface

# Robust LP Example
m, n = 2, 3
Random.seed!(0)
A_ = rand(m,n)
b_ = rand(m)
c_ = ones(n)

P = Matrix(1.0I, n, n)

# Construct matrices and vectors
c = [c_; zeros(m+n)]
b = zeros(n+m*(n+1)+m)
b[n+1:n+m] = b_
A = spzeros(n+m*(n+1)+m, n+m+n)
A[1:n, 1:n] = P'
A[1:n, n+m+1:end] = -Matrix(I, n,n)
for ind in 0:m-1
    # Zero cone constraints
    A[n+ind+1, 1:n] = -A_[ind+1,:]
    A[n+ind+1, n+ind+1] = -1.0
    # SOC constraints
    A[n+m+ind*(n+1)+1,n+ind+1] = 1.0
    A[n+m+ind*(n+1)+2:n+m+ind*(n+1)+1+n, n+m+1:end] = Matrix(I, n, n)
end
A = -A
cone_prod = vec([MOI.Zeros(n+m) repeat([MOI.SecondOrderCone(n+1)],1,m)])

# Solve
x_, y_, s_, pf, pb = ConeProgramDiff.solve_and_diff(A, Vector(b), c, cone_prod; eps=1e-6, verbose=true)

# Compute dP/dx
dP = pb([ones(n); zeros(m+n)])[1][1:n,1:n]
dd, vv = eigen(dP)

# Compare norms
dir = vv[:,1]
dA = zeros(size(A))
db = zeros(size(b))
dc = zeros(size(c))
dA[1:n,1:n] = dir * dir'
norm(pf(dA, db, dc)[1][1:n])
dir = vv[:,2]
dA[1:n,1:n] = dir * dir'
norm(pf(dA, db, dc)[1][1:n])


## Some Checks
function check_formulation()
    # Check formulation
    model = Model(SCS.Optimizer)
    @variable(model, x[1:n])
    @objective(model, Min, c_'*x)
    for i in 1:m
        @constraint(model, [b_[i]-dot(A_[i,:],x); P'*x] ∈ MOI.SecondOrderCone(n+1))
    end
    optimize!(model)

    return isapprox(x_[1:n], value.(x), atol=1e-6)
end
check_formulation()


mA, nA = size(A)
f = m+n
l = 0  # the number of linear cones
q = repeat([n+1], m)  # the array of SOC sizes
s = Int[]  # the array of SDC sizes
ep = 0  # the number of exponential cones
ed = 0  # the number of dual exponential cones
p = Float64[]  # the array of power cone parameters

scs_sol = SCS.SCS_solve(SCS.IndirectSolver, mA, nA, A, b, c, f, l, q, s, ep, ed, p,
    eps=1e-6, verbose=true)
scs_sol.x
x_



#
# MOI.status
# error("Info: $(,scs_sol.info))")
# fieldnames(structure(scs_sol.info))
# nfields(structure(scs_sol.info))
# field
#
# get
# isstructtype(structure(scs_sol.info))
