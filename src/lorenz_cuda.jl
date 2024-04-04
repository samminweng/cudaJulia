import Pkg
Pkg.add("CUDA") # Install Cuda
Pkg.add("BenchmarkTools") # Install benchmark tools
Pkg.add("BenchmarkPlots")
Pkg.add("StatsPlots")
Pkg.add("Plots") # Install `Plots` package


using Plots, Images, FileIO
# Use Cuda
using CUDA, Test, BenchmarkTools, BenchmarkPlots, StatsPlots
# println("CUDA information\n", CUDA.versioninfo())
# Reference: https://pde-on-gpu.vaw.ethz.ch/lecture1/#solving_partial_differential_equations_in_parallel_on_gpus
# CUDA.allowscalar(true)

const DIM = 3 # Specify the dimensions
const STEPS = 100 
# # integrate dx/dt = lorenz(t,x) numerically for 10 steps
function lorenz(x)
    σ = 10
    β = 8/3
    ρ = 28
    return [σ*(x[2]-x[1]),
            x[1]*(ρ-x[3]) - x[2],
            x[1]*x[2] - β*x[3]]
end
# Plot loren results on a plot (png) and an animation (gif)
function plot_loren(out, device)
    # Animate the loren on CPU
    plot(out[1,:], out[2,:], out[3,:])
    path = "images/lorenz_$device.png"
    savefig(path)
    println("Save $device result to $path")
    # initialize a 3D plot with 1 empty series
    plt = plot3d(
        1,
        # xlim = (0, 30),
        # ylim = (0, 30),
        # zlim = (0, 30),
        title = "Lorenz Attractor on $device",
        legend = false,
        marker = 2,
    )

    # build an animated gif by pushing new points to the plot, saving every 10th frame
    anim = @animate for i=1:STEPS        
        push!(plt, out[1,i], out[2,i], out[3,i])
    end every 1
    path = "images/lorenz_$device.gif"
    gif(anim, path, fps=1)
    println("Save $device result to $path")
end

function lorenz_gpu(x, y, z)
    σ = 10
    β = 8/3
    ρ = 28
    return CuArray([σ*(y-x), x*(ρ-z) - y, x*y - β*z])
end

# Animate lorenz system from step 2 upto the final steps
function gpu_loren!(x, lz, dt)
    i = threadIdx().x
    x[i] = x[i] + lz[i] * dt[i]
    return nothing
end

function gpu_step!(out_d)
    dt = CUDA.fill(0.01f0, 3)
    # Simulate each step 
    for step = 2:STEPS
        x = out_d[:, step-1]
        lz = lorenz_gpu(x[1], x[2], x[3])
        @cuda threads=length(x) gpu_loren!(x, lz, dt)
        out_d[:, step] = x
        lz=nothing
    end
    dt = nothing; 
    return nothing
end

function gpu_benchmark()
    x_d = CUDA.fill(0.0f0, 3)
    out_d = CUDA.fill(0.0f0, (3, STEPS))
    x_d[1]= 2.0
    println(" size(out_d) = ", size(out_d))
    # Initial 'out' values
    out_d[:, 1] = x_d    
    b = @benchmark gpu_step!(out_d) setup=(out_d=copy($out_d)) evals=1 samples=1
    io = IOContext(stdout, :histmin=>1, :histmax=>2, :logbins=>true)
    # using BenchmarkPlots, StatsPlots
    println("Benchmark results for CUDA 'gpu_step!' ")
    println("Minimal time of CUDA gpu_step! = ", minimum(b))
    println("Mean time of CUDA gpu_step! = ", mean(b))
    println("Maximum time of CUDA gpu_step! = ", maximum(b))
    show(io, MIME("text/plain"), b) 
end

function gpu_run()
    println("Start running LORENZ on GPUs")
    # Create 'x' and out on GPU device
    x_d = CuArray([2.0, 0.0, 0.0])
    out_d = CUDA.fill(0.0f0, (3, STEPS))
    println(" size(out_d) = ", size(out_d))
    # Initial 'out' values
    out_d[:, 1] = x_d
    gpu_step!(out_d)
    GC.gc(true) # Clear memory
    println(CUDA.memory_status())
    # Run the lorenz on GPUs
    out = Array(out_d)
    println(" Vector size 'out' = ", size(out))
    plot_loren(out, "gpu")
    out = nothing; GC.gc(true) # Clear memory
    println(CUDA.memory_status())
end


# Animate lorenz system from step 2 upto the final steps
function cpu_step!(out)
    dt = 0.01 
    for i = 2: size(out, 2)
        out[:, i] = out[:, i-1] + lorenz(out[:, i-1]) * dt
    end
    return nothing
end
 # Run Benchmark on CPUs
function cpu_benchmarks()   
    x = zeros(DIM)
    x[1] = 2.0
    out = zeros(3, STEPS)
    out[:, 1] = x 
    io = IOContext(stdout, :histmin=>0.001, :histmax=>300, :logbins=>true)
    b = @benchmark cpu_step!(out) setup=(out=copy($out)) evals=1 samples=1
    println("Benchmark results for 'cpu_step!' ")
    println("Minimal time of cpu_step! = ", minimum(b))
    println("Mean time of cpu_step! = ", mean(b))
    println("Maximum time of cpu_step! = ", maximum(b))
    # plot(b)
    show(io, MIME("text/plain"), b)
end

function cpu_run()
    x = zeros(DIM)
    x[1] = 2.0
    out = zeros(3, STEPS) # 3xSTEP arrays
    out[:, 1] = x 
    cpu_step!(out)
    # println("out = ", out)
    plot_loren(out, "cpu")
    out = nothing; GC.gc(true) # Clear memory
    println(CUDA.memory_status())
end

# Perform LORENZ on CPUs
# cpu_run()
# cpu_benchmarks()
# Perform LORENZ on GPUS
gpu_run()
# gpu_benchmark()