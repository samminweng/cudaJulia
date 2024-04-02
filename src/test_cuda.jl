# Install Cuda in Julia
import Pkg
Pkg.add("CUDA") # 
Pkg.test("CUDA")

# Use Cuda
using CUDA
println(string("", CUDA.versioninfo()))

# An example of vector add
const N = 20 # 2 to the power of 20 is 1048576
const THREADS_PER_BLOCK = 16*16

function seq_add(y, x)
    @time begin # Measure the loop time
        for i in eachindex(y, x)
            @inbounds y[i] = y[i] + x[i]  # Remove array bounds checking
        end
    end # End timing
    return y
end # End function 


x = fill(1.0f0, N) # Assing each element in X as 1.0 (Float32)
y = fill(2.0f0, N)
y = seq_add(y, x) # y = y + x
println(y)
using Test
@test all(y .== 3.0f0)
