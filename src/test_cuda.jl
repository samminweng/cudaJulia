# Install Cuda in Julia
import Pkg
Pkg.add("CUDA") # Install Cuda
Pkg.add("BenchmarkTools") # Install benchmark tools
Pkg.add("BenchmarkPlots")
Pkg.add("StatsPlots")
# Use Cuda
using CUDA, Test, BenchmarkTools
using BenchmarkPlots, StatsPlots
println("CUDA information\n", CUDA.versioninfo())

# An example of vector add
const N = 5 # 2 to the power of 20 is 1048576
const THREADS_PER_BLOCK = 16*16

function run_test(y)
    @testset "Test" begin
        @test all(y .== 3.0f0)
    end
end

function cuda_add!(y_d, x_d)
    # Allocate x_d and y_d on GPU device
    CUDA.@sync y_d .= y_d + x_d
    return nothing # Move the device memory to host
end
# Run on GPUs (CUDA)
function gpu_run()
    x_d = CUDA.fill(1.0f0, N) # Assing each element in X as 1.0 (Float32)
    y_d = CUDA.fill(2.0f0, N)
    # # # Use @elapsed macro to measure the execution time
    println("Unit Test: Execution time of cuda_add! ", @elapsed (cuda_add!(y_d, x_d)))
    y = Array(y_d)
    println("y[1] = ", y[1]) # Julia array starts with 1
    run_test(y)
    # Use  @btime macro to benchmark the function
    b = @benchmark cuda_add!(y, x) setup=(y=$y_d; x=$x_d) evals=1 samples=1
    io = IOContext(stdout, :histmin=>1, :histmax=>10, :logbins=>true)
    # using BenchmarkPlots, StatsPlots
    println("Benchmark results for 'cuda_add!' ")
    println("Minimal time of cuda_add! = ", minimum(b))
    println("Mean time of cuda_add! = ", mean(b))
    println("Maximum time of cuda_add! = ", maximum(b))
    show(io, MIME("text/plain"), b) 
end

# Run with sequential execution
function seq_add!(y, x)
    for i in eachindex(y, x)
        @inbounds y[i] = y[i] + x[i]  # Remove array bounds checking
    end
    return nothing
end # End function 'seq_add'

# Run with multiple threads on CPU 
function parallel_add!(y, x)
    Threads.@threads for i in eachindex(y, x)
        @inbounds y[i] = y[i] + x[i]
    end # End of for loop
    return nothing 
end # End function 'parallel_add'

function cpu_run()
    x = fill(1.0f0, N) # Assing each element in X as 1.0 (Float32)
    y = fill(2.0f0, N)
    # Use @elapsed macro to measure the execution time
    println("Execution time of seq_add = ", @elapsed seq_add!(y, x), " seconds") 
    println("y[1] = ", y[1]) # Julia array starts with 1
    run_test(y)
    # Use  @btime macro to benchmark the function
    io = IOContext(stdout, :histmin=>0.001, :histmax=>300, :logbins=>true)
    # Benchmark 'seq_add'
    b = @benchmark seq_add!(y, x) setup=(y=copy($y); x=$x) evals=1 samples=1
    println("Benchmark results for 'seq_add!' ")
    println("Minimal time of seq_add! = ", minimum(b))
    println("Mean time of seq_add! = ", mean(b))
    println("Maximum time of seq_add! = ", maximum(b))
    # plot(b)
    show(io, MIME("text/plain"), b)    
end # End function

print("CPU Version")
cpu_run()
println("\n\n-----------------------------------\n\n")
print("GPU Version")
gpu_run()
println("\n\n-----------------------------------\n\n")
