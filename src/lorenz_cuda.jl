import Pkg
Pkg.add("CUDA") # Install Cuda
Pkg.add("BenchmarkTools") # Install benchmark tools
Pkg.add("BenchmarkPlots")
Pkg.add("StatsPlots")
Pkg.add("Plots") # Install `Plots` package
Pkg.add("Images") 
Pkg.add("FileIO")

using Plots, Images, FileIO
# Use Cuda
using CUDA, Test, BenchmarkTools, BenchmarkPlots, StatsPlots
# println("CUDA information\n", CUDA.versioninfo())
# Reference: https://pde-on-gpu.vaw.ethz.ch/lecture1/#solving_partial_differential_equations_in_parallel_on_gpus
CUDA.allowscalar(true)

const DIM = 3 # Specify the dimensions
const STEPS = 10 
# # integrate dx/dt = lorenz(t,x) numerically for 10 steps
function lorenz(x)
    σ = 10
    β = 8/3
    ρ = 28
    return [σ*(x[2]-x[1]),
            x[1]*(ρ-x[3]) - x[2],
            x[1]*x[2] - β*x[3]]
end
# Animate lorenz system from step 2 upto the final steps
function gpu_step!(x)
    dt = 0.01
    lz = lorenz(x)
    for i in eachindex(x)
        @inbounds x[i] = x[i] + lz[i] * dt
    end
    # @cuprintln("x $x")
    return nothing
end


function gpu_benchmark()
    println("Start running LORENZ on GPUs")
    # Create 'x' and out on GPU device
    x_d = CUDA.fill(0.0f0, 3)
    out_d = CUDA.fill(0.0f0, (3, STEPS))
    x_d[1]= 2.0
    println(" size(out_d) = ", size(out_d))
    # Initial 'out' values
    out_d[:, 1] = x_d
    println("out_d = ", out_d)    
    for step = 2:STEPS
        x = copy(out_d[:, step-1])
        gpu_step!(x)
        out_d[:, step] = x
    end
    # Run the lorenz on GPUs
    # println("Unit Test: Execution time of CUDA 'gpu_step!' ", )
    out = Array(out_d)
    println("out = ", out, " Vector size 'out' = ", size(out))
    plot(out[1,:], out[2,:], out[3,:])
    path = "images/lorenz_gpu.png"
    savefig(path)
    print("Save GPU results to $path")
    println("Complete LORENZ on GPUs")
end


# Animate lorenz system from step 2 upto the final steps
function cpu_step!(out)
    dt = 0.01 
    for i = 2: size(out, 2)
        out[:, i] = out[:, i-1] + lorenz(out[:, i-1]) * dt
    end
    return nothing
end

function plot_loren(out, device)
    # Animate the loren on CPU
    plot(out[1,:], out[2,:], out[3,:])
    path = "images/lorenz_$device.png"
    savefig(path)
    print("Save $device result to $path")
    # initialize a 3D plot with 1 empty series
    plt = plot3d(
        1,
        # xlim = (0, 30),
        # ylim = (0, 30),
        # zlim = (0, 30),
        title = "Lorenz Attractor",
        legend = false,
        marker = 2,
    )

    # build an animated gif by pushing new points to the plot, saving every 10th frame
    anim = @animate for i=1:STEPS        
        push!(plt, out[1,i], out[2,i], out[3,i])
    end every 1
    path = "images/lorenz_$device.gif"
    gif(anim, path, fps=15*2)
    print("Save $device result to $path")
end

function cpu_benchmark()
    x = zeros(DIM)
    x[1] = 2.0
    out = zeros(3, STEPS) # 3x500 arrays
    println(" size(out) = ", size(out))
    out[:, 1] = x 
    cpu_step!(out)
    println("out = ", out)
    plot_loren(out, "cpu")
end

# Perform LORENZ on CPUs
cpu_benchmark()
# Perform LORENZ on GPUS
# gpu_benchmark()