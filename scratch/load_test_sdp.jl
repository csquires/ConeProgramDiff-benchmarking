using ConeProgramDiff

prog = ConeProgramDiff.cp_from_file("scratch/sdp_test.txt", dense=false)
ConeProgramDiff.cp_to_file("scratch/sdp_test_jl.txt", prog, dense=false)
println(prog[:A][1:5, 1:5])
println(prog[:b][1:5])
println(prog[:c][1:5])

# have to re-shuffle rows of a and b
