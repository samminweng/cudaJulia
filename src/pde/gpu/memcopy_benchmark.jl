import Pkg
Pkg.add("CUDA") # Install Cuda 
Pkg.add("DataFrames") # Install dataframe
Pkg.add("CSV")

using CUDA #Use CUDA.jl library
using Test, BenchmarkTools, BenchmarkPlots 
using Plots, Plots.PlotMeasures, DataFrames, CSV, Random
# Default Plot options 
default(size=(1200, 400), framestyle=:box, label=false, margin=40px,
    grid=true, linewidth=6.0, thickness_scaling=1,
    labelfontsize=16, tickfontsize=8, titlefontsize=18)
# GPU device info
println(CUDA.versioninfo())
# Select the first GPU device
# println("Device info: $(collect(devices()))")
device!(0) #Select device 0 
@inbounds memcopy_AP!(A, B, C, s) = (A .= B .+ s.*C)
function bench_memcopy_cpus()
    array_sizes = Float64[]
    throughtputs = Float64[]
    # Benchmark loop
    for pow = 0:20
        nx = ny = 32 * 2^pow 
        array_size = nx*ny*sizeof(Float64)
        if 3array_size > 4e9 # > 4Gib
            break
        end
        A = zeros(Float64, nx, ny) # Create array A on device memory allocates Float64
        B = rand(Float64, nx, ny)
        C = rand(Float64, nx, ny)
        s = rand((1, 10))
        t_it = @belapsed begin memcopy_AP!($A, $B, $C, $s); synchronize() end
        numOp = 3 # 3: Two Read and one write operations
        T_tot = numOp * 1/1e9 * nx * ny * sizeof(Float64)/t_it
        ## Display the results
        println("--------------------------------------------------------")
        println("array_size = $(array_size), nx = $(nx), ny = $(ny)")
        println("selected CPUs = $(Sys.cpu_info()[1].model)")
        println("total execution time = $(t_it) [s]")
        println("benchmark memory throughtput = $( round(T_tot, sigdigits=3) ) [G/s]")
        # Update the results
        push!(array_sizes, array_size)
        push!(throughtputs, T_tot)
        GC.gc()
    end
    throughtputs .= round.(throughtputs, sigdigits=4)
    println("Complete CPU benchmarks on $(Sys.cpu_info()[1].model)")
    println("array_sizes = $(array_sizes)")
    println("throughtputs = $(throughtputs)")
    # Write to a dataframe
    df = DataFrame()
    df.array_sizes = array_sizes
    df.throughtputs = throughtputs
    path = joinpath("data", "memorycopy_benchmark_cpu.csv")
    CSV.write(path, df)
    println("Save benchmark results to $(path)")
end


# Kernel function
function memcopy_KP!(A, B, C, s)
    # Compute thread ID (ix, iy)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    A[ix,iy] = B[ix,iy] + s * C[ix, iy]# Copy arrays
    return nothing
end
# Ref: https://github.com/eth-vaw-glaciology/course-101-0250-00/blob/main/slide-notebooks/notebooks/l6_1-gpu-memcopy.ipynb
function bench_memcopy_gpus()
    max_threads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    array_sizes = Float64[]
    throughtputs = Float64[]
    # Benchmark loop
    for pow = 0:20
        nx = ny = 32 * 2^pow 
        array_size = nx*ny*sizeof(Float64)
        if 3*array_size > 4e9 # > 4Gib
            break
        end
        A = CUDA.zeros(Float64, nx, ny) # Create array A on device memory allocates Float64
        B = CUDA.rand(Float64, nx, ny)
        C = CUDA.rand(Float64, nx, ny)
        s = rand((1, 10))
        println("s = $(s)")
        # Use only 32*2 thread per block
        threads = (32, 2)
        blocks =(nx÷threads[1], ny÷threads[2])
        t_it = @belapsed begin @cuda blocks=$blocks threads=$threads memcopy_KP!($A, $B, $C, $s); synchronize() end
        # Compute the throughputs 
        numOp = 3 # 3: Two Read and one write operations
        T_tot = numOp * 1/1e9 * nx * ny * sizeof(Float64)/t_it
        ## Display the results
        println("--------------------------------------------------------")
        println("array_size = $(array_size), nx = $(nx), ny = $(ny), threads = $(threads), blocks=$(blocks)")
        println("selected GPUs = $( CUDA.name( CUDA.current_device() ) ) ")
        println("total execution time = $(t_it) [s]")
        println("benchmark memory throughtput = $( round(T_tot, sigdigits=3) ) [G/s]")
        # Update the results
        push!(array_sizes, array_size)
        push!(throughtputs, T_tot)
        CUDA.unsafe_free!(A)
        CUDA.unsafe_free!(B)
        CUDA.unsafe_free!(C)
    end
    throughtputs .= round.(throughtputs, sigdigits=4)
    println("Complete CUDA benchmarks on $( CUDA.name( CUDA.current_device() ) )")
    println("array_sizes = $(array_sizes)")
    println("throughtputs = $(throughtputs)")
    # Write to a dataframe
    df = DataFrame()
    df.array_sizes = array_sizes
    df.throughtputs = throughtputs
    path = joinpath("data", "memorycopy_benchmark_cuda.csv")
    CSV.write(path, df)
    println("Save benchmark results to $(path)")
end

function display_bench()
    df_cpu = DataFrame(CSV.File(joinpath("data", "memorycopy_benchmark_cpu.csv")))
    df_gpu = DataFrame(CSV.File(joinpath("data", "memorycopy_benchmark_cuda.csv")))
    # Plot the results
    array_sizes = df_gpu.array_sizes
    gpu_throughtputs = df_gpu.throughtputs
    cpu_throughtputs = df_cpu.throughtputs
    throughtputs_max, index = findmax([gpu_throughtputs; cpu_throughtputs])
    xticks_array_sizes = string.(Int.(log2.(array_sizes)))
    xticks_array_sizes = map(x -> string("2^{", x, "}"), xticks_array_sizes)
    println("xticks_array_sizes = $( xticks_array_sizes )")
    nx = ny = array_sizes[index]
    println("nx = $(nx) and ny = $(ny) with max throughputs = $(throughtputs_max)")
    p = plot(array_sizes, [gpu_throughtputs, cpu_throughtputs];
             label=["GPUs" "CPUs"], legend=:outertopright,
             ylimit=(0, throughtputs_max+10),  
             xlabel="array size [log2(nx*ny)]", ylabel="Memory throughput [GB/s]",
             title="Benchmarks on Memory Copy on GPUs and CPUs", 
             xscale=:log2, yticks=0:20:200, 
             markersize=5, markershape=:circle)
    xticks!(p, array_sizes, xticks_array_sizes)
    display(p)
    png(p, "images/pde/gpu/memcopy_benchmark.png")
end

bench_memcopy_cpus() # Benchmark on CPU
bench_memcopy_gpus() # Benchmark on GPUs
# display_bench()


function bench_cuda_threads()
    nx = ny = 4096
    max_threads = attribute(device(),CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    println("max_threads = $(max_threads)")
    thread_counts = []
    throughtputs = []
    for pow = 0: Int(log2(max_threads/32))
        A = CUDA.zeros(Float64, nx, ny) # Create array A on device memory allocates Float64
        B = CUDA.rand(Float64, nx, ny)
        threads = (32, 2^pow)
        blocks =(nx÷threads[1], ny÷threads[2])
        t_it = @belapsed begin @cuda blocks=$blocks threads=$threads memcopy_KP!($A, $B); synchronize() end
        # Compute the throughputs 
        numOp = 2 # 2: Read and write operations
        T_tot = round(numOp * 1/1e9 * nx * ny * sizeof(Float64)/t_it, sigdigits=5)
        thread_count = prod(threads)
        push!(thread_counts, thread_count)
        push!(throughtputs, T_tot)
        println("Thread count = $(thread_count), throughtput = $(T_tot)")
    end
    df = DataFrame()
    df.thread_counts = thread_counts
    df.throughtputs = throughtputs
    path = joinpath("data", "memorycopy_benchmark_cuda_threads.csv")
    CSV.write(path, df)

    throughtputs_max, index = findmax(throughtputs)
    thread_count_max = thread_counts[index]
    println("thread_count = $(thread_count_max), maximal throughputs = $(throughtputs_max)")
end

# bench_cuda_threads()