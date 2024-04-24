import Pkg
Pkg.add("CUDA") # Install Cuda 
using CUDA #Use CUDA.jl library
using Test, BenchmarkTools, BenchmarkPlots 
using Plots, Plots.PlotMeasures
# Default Plot options 
default(size=(1200, 400), framestyle=:box, label=false, margin=40px,
    grid=true, linewidth=6.0, thickness_scaling=1,
    labelfontsize=16, tickfontsize=8, titlefontsize=18)
# GPU device info
println(CUDA.versioninfo())
# Select the first GPU device
# println("Device info: $(collect(devices()))")
device!(0) #Select device 0 


# Kernel function
function copy!(A, B)
    # Compute thread ID (ix, iy)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    A[ix,iy] = B[ix,iy] # Copy arrays
    return
end

function bench_gpus()
    array_sizes = Float64[]
    throughtputs = Float64[]
    # Benchmark loop
    for pow = 1:10
        nx = ny = 32 * 2^pow 
        array_size = nx*ny*sizeof(Float64)
        if 3*array_size > 4e9 # > 4Gib
            break
        end
        A = CUDA.zeros(Float64, nx, ny) # Create array A on device memory allocates Float64
        B = CUDA.rand(Float64, nx, ny)
        # @benchmark begin copyto!($A, $B); synchronize() end
        t_it = @belapsed begin copyto!($A, $B); synchronize() end
        # Compute the throughputs 
        numOp = 2 # 2: Read and write operations
        T_tot = numOp * 1/1e9 * nx * ny * sizeof(Float64)/t_it
        ## Display the results
        println("--------------------------------------------------------")
        println("array_size = $(array_size), nx = $(nx), ny = $(ny)")
        println("selected GPUs = $( CUDA.name( CUDA.current_device() ) ) ")
        println("benchmark memory throughtput = $( round(T_tot, sigdigits=3) ) [G/s]")
        # Update the results
        push!(array_sizes, array_size)
        push!(throughtputs, T_tot)
        CUDA.unsafe_free!(A)
        CUDA.unsafe_free!(B)
    end
    throughtputs .= round.(throughtputs, sigdigits=4)
    p = plot(array_sizes, throughtputs;
             ylimit=(0, throughtputs[end]+10),  
             xlabel="array size [log2(nx*ny)]", ylabel="Memory throughput [GB/s]",
             title="Benchmarks on Memory Copy on GPUs", 
             xscale=:log2, grid=false, 
             markersize=5, markershape=:circle)
    display(p)
    png(p, "images/pde/gpu/memcopy_benchmark.png")
    println("array_sizes = $(array_sizes)")
    println("throughtputs = $(throughtputs)")
end


bench_gpus()
