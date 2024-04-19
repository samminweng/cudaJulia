# # Install Cuda in Julia
import Pkg
# Pkg.add("CUDA") # Install Cuda
Pkg.add("BenchmarkTools") # Install benchmark tools
Pkg.add("Plots") # Install Plots 
Pkg.add("LoopVectorization") # Install loop vectorization
# Pkg.add("BenchmarkPlots")
using Plots, BenchmarkTools, Plots.PlotMeasures, Printf, Test, Base.Threads
using LoopVectorization # using @tturbo: Loop vectorization

println("The number of threads = $(nthreadpools())")
# Default Plot options 
default(size=(1200, 400), framestyle=:box, label=false, margin=40px,
    grid=true, linewidth=6.0, thickness_scaling=1,
    labelfontsize=14, tickfontsize=8, titlefontsize=18)
FPS = 1

# Create macros that computes difference 
macro d_xa(A)
    esc(:($A[ix+1, iy] - $A[ix, iy]))
end
macro d_ya(A)
    esc(:($A[ix, iy+1] - $A[ix, iy]))
end
# array programming-based version of memory copy
function compute_ap!(C2, C, A)
    C2 .= C .+ A 
    return nothing
end
# "kernel programming"-based function uses a loop-based implementation with multi-threading
function compute_kp!(C2, C, A)
    nx, ny = size(C2)
    Threads.@threads for iy = 1:ny
        for ix = 1:nx
            C2[ix, iy] = C[ix, iy] + A[ix, iy]
        end
    end
end

function memcopy(nx, ny, bench="loop")  # the number of grid points
    # numerics 
    # nx, ny = 512, 512
    nt = 2e4
    if nx > 2048
        nt = 2e3
    end
    # array initialization
    C = rand(Float64, nx, ny)
    C2 = copy(C)
    A = copy(C)
    t_toc = 0.0
    if bench == "loop"
        # iteration loop
        t_tic = Base.time()
        for iter=1:nt
            compute_ap!(C2, C, A)
        end
        t_toc = Base.time() - t_tic
    elseif bench == "btool"
        t_toc = @belapsed compute_ap!($C2, $C, $A)
    end
    # Performance metrics
    A_eff = (3 * 2) / 1e9 * nx * ny * sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it = t_toc / nt                      # Execution time per iteration [s]
    T_eff = A_eff / t_it                     # Effective memory throughput [GB/s]
    println("----------------------------------------------------------------") 
    println("Complete the simulation of memcopy of nx = $(nx) and ny = $(ny) ")
    println("Total number of iterations = $(nt), Elapsed time = $(round(t_toc, sigdigits=3)) [s]")
    println("Execution time per iteration = $(round(t_it, sigdigits=3)) [s]")
    println("Memory access per iteration = $(round(A_eff, sigdigits=3)) [GB], Memory throughput = $(round(T_eff, sigdigits=3)) [GB/s]")
    return t_it, A_eff, T_eff
end

function main()
    # Benchmark results
    t_its = Float64[]
    T_effs = Float64[]
    ny = 512
    nxs = 16 * 2 .^ (1:8)
    for i in eachindex(nxs)
        t_it, A_eff, T_eff = memcopy(nxs[i], ny, "loop")
        push!(t_its, t_it)
        push!(T_effs, T_eff)
    end
    # Plot the results
    p1 = plot(nxs, t_its;  yscale=:log10,
            xlabel="nx (grid size)", ylabel="Execution time per iteration [s]",
            markersize=10, markershape=:circle)
    p2 = plot(nxs, T_effs;  yscale=:log10,
            xlabel="nx (grid size)", ylabel="Memory throughput [GB/s]",
            markersize=10, markershape=:circle)
    p3 = plot(p1, p2; layout=(1, 2))
    display(p3)
    png(p3, "images/pde/par/memcopy.png")
end
main()