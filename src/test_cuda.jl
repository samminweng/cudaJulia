# Install Cuda in Julia
import Pkg
Pkg.add("CUDA") # Install Cuda
Pkg.add("BenchmarkTools") # Install benchmark tools
Pkg.add("BenchmarkPlots")
Pkg.add("StatsPlots")
# Use Cuda
using CUDA, Test, BenchmarkTools
using BenchmarkPlots, StatsPlots
# println("CUDA information\n", CUDA.versioninfo())

# An example of vector add
const N = 2^20 # 2 to the power of 20 is 1048576
# const THREADS_PER_BLOCK = 16*4
# const NUM_BLOCKS = ceil(Int, N/THREADS_PER_BLOCK) # Number of blocks = Total number of tasks (N) / THREADS_PER_BLOCK
# println("N = ", N, " Threads per block = ", THREADS_PER_BLOCK, " Total Number of blocks = ", NUM_BLOCKS)

function run_test(y)
    @testset "Test" begin
        @test all(y .== 3.0f0)
    end
end

# Run with sequential execution
function vector_add!(y, x)
    for i = 1:length(y)
        @inbounds y[i] = y[i] + x[i]  # Remove array bounds checking
    end
    return nothing
end # End function 'vector_add'

# Function name with `!` indicates that the function modifies the input vector y in-place.
function vector_add_gpu!(y, x)
    # blockIdx() and threadIdx() are built-in functions 
    # blockIdx(): the index of the current block within the grid
    index = (blockIdx().x -1) * blockDim().x + threadIdx().x # Global thread id
    stride = gridDim().x * blockDim().x # 
    # @cuprintln("thread $index, block $stride")
    for i=index:stride:length(y)
        @inbounds y[i] = y[i] + x[i]
    end
    return nothing
end
# Benchmark the `vector_add_gpu` function
function benchmark_gpu!(y, x)
    kernel = @cuda launch=false vector_add_gpu!(y, x)
    config = launch_configuration(kernel.fun)
    numThreads = min(length(y), config.threads) # Threads per block
    # The smallest integer >= the division of length(y) by threads
    numBlocks = cld(length(y), numThreads) # Number of blocks
    println("Number of blocks = ", numBlocks, " Number of Threads = ", numThreads)
    CUDA.@sync begin
        kernel(y, x; threads=numThreads, blocks=numBlocks)
    end
end

# Run on GPUs (CUDA)
function gpu_run()
    x_d = CUDA.fill(1.0f0, N) # Assing each element in X as 1.0 (Float32)
    y_d = CUDA.fill(2.0f0, N)
    # # # Use @elapsed macro to measure the execution time
    println("Unit Test: Execution time of CUDA 'vector_add!' ", @elapsed benchmark_gpu!(y_d, x_d))
    y = Array(y_d)
    println("y[1] = ", y[1], " Array size = ", length(y)) # Julia array starts with 1
    run_test(y)
    # # # Use @benchmark macro to benchmark the function   
    y_d = CUDA.fill(2.0f0, N)
    b = @benchmark benchmark_gpu!(y, x) setup=(y=$y_d; x=$x_d) evals=1 samples=1
    io = IOContext(stdout, :histmin=>1, :histmax=>2, :logbins=>true)
    # using BenchmarkPlots, StatsPlots
    println("Benchmark results for CUDA 'vector_add_gpu!' ")
    println("Minimal time of CUDA vector_add_gpu! = ", minimum(b))
    println("Mean time of CUDA vector_add_gpu! = ", mean(b))
    println("Maximum time of CUDA vector_add_gpu! = ", maximum(b))
    show(io, MIME("text/plain"), b) 
    # # Profile the code
    # benchmark_gpu!(y_d, x_d)
    # println("\n\n", CUDA.@profile trace=true benchmark_gpu!(y_d, x_d))
end
print("GPU Version")
gpu_run()
println("\n\n-----------------------------------\n\n")

# # Run with multiple threads on CPU 
# function parallel_add!(y, x)
#     Threads.@threads for i = 1:length(y)
#         @inbounds y[i] = y[i] + x[i]
#     end # End of for loop
#     return nothing 
# end # End function 'parallel_add'

function cpu_run()
    x = fill(1.0f0, N) # Assing each element in X as 1.0 (Float32)
    y = fill(2.0f0, N)
    # Use @elapsed macro to measure the execution time
    println("Unit Test: Execution time of 'vector_add!' = ", @elapsed (vector_add!(y, x)), " seconds") 
    println("y[1] = ", y[1], " Array size = ", length(y)) # Julia array starts with 1
    run_test(y)
    # Use  @btime macro to benchmark the function
    io = IOContext(stdout, :histmin=>0.001, :histmax=>300, :logbins=>true)
    # Benchmark 'seq_add'
    y = fill(2.0f0, N)
    b = @benchmark vector_add!(y, x) setup=(y=copy($y); x=$x) evals=1 samples=1
    println("Benchmark results for 'vector_add!' ")
    println("Minimal time of vector_add! = ", minimum(b))
    println("Mean time of vector_add! = ", mean(b))
    println("Maximum time of vector_add! = ", maximum(b))
    # plot(b)
    show(io, MIME("text/plain"), b)    
end # End function

print("CPU Version")
cpu_run()
println("\n\n-----------------------------------\n\n")

