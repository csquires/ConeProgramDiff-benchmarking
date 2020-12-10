using NPZ

function load_program(folder)
    A = npzread("$folder/A.npz")
    b = readdlm("$folder/b.txt")
    c = readdlm("$folder/c.txt")
    return A, b, c
end

A, b, c = load_program("random_programs/test")
