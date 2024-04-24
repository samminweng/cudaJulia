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
    labelfontsize=16, tickfontsize=8, titlefontsize=18, 
    xtickfontsize=8)

# Create macros that computes difference 
macro d_xa(A)
    esc(:($A[ix+1, iy] - $A[ix, iy]))
end
macro d_ya(A)
    esc(:($A[ix, iy+1] - $A[ix, iy]))
end

function compute_kp!(C2, C, A)
    nx, ny = size(C2)
    for iy=1:ny
        for ix=1:nx
            C2[ix, iy] = C[ix, iy] + A[ix, iy]
        end
    end
    return nothing
end

#  a broadcasted version of the memory copy operation
function compute_ap!(C2, C, A)
    C2 .= C .+ A
    return nothing
end
# nx, ny: the number of grid points
function memcopy(nx, ny, bench="ap")
    # Numerics
    nt = 2e4
    if nx >=2048
        nt = 1000
    end
    # array initialisation
    C = rand(Float64, nx, ny)
    C2 = copy(C)
    A = copy(C)
    # iteration loop
    iter = 1
    t_toc = 0
    niter = 0
    # iteration loop
    t_tic = Base.time()
    while iter <= nt
        #Compute diffusion physics 
        if bench == "ap"
            compute_ap!(C2, C, A)
        else
            compute_kp!(C2, C, A)
        end
        iter += 1
    end
    t_toc = Base.time() - t_tic
    niter = nt # Total number of iterations
    # Compute the metrics
    # Performance metrics
    A_eff = (3 * 2) / 1e9 * nx * ny * sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it = t_toc / niter                      # Execution time per iteration [s]
    T_eff = A_eff / t_it                     # Effective memory throughput [GB/s] 
    println("----------------------------------------------------------------")
    println("Complete the simulation of memcopy of nx = $(nx) and ny = $(ny) ")
    println("Total number of iterations = $(niter), Elapsed time = $(round(t_toc, sigdigits=3)) [s]")
    println("Execution time per iteration = $(round(t_it, sigdigits=3)) [s]")
    println("Memory access per iteration = $(round(A_eff, sigdigits=3)) [GB]") 
    println("Memory throughput = $(round(T_eff, sigdigits=3)) [GB/s]")
    return T_eff
end
function main()
    benchmark_results = Dict() 
    for bench in ["ap", "kp"]
        # Benchmark results
        iter_evo = Int[]
        t_eff_evo = Float64[]
        nxs = nys = 16 * 2 .^ (1:8)
        for i =1:8
            nx, ny = nxs[i], nys[i]
            t_eff = memcopy(nx, ny, bench)
            push!(iter_evo, nx*ny)
            push!(t_eff_evo, t_eff)
        end
        t_eff_evo .= round.(t_eff_evo, sigdigits=3)
        println("benchmark = $(bench)")
        println("iter_evo = $(iter_evo)")
        println("t_eff_evo = $(t_eff_evo)")
        if bench == "ap"
            title = "array programming"
        else
            title = "kernel programming"
        end
        p = plot(iter_evo, t_eff_evo; 
                 ylimit=(0, 250), xlimit=(0, 4096*4096), 
                 xlabel="grid size [log(nx*ny)]", ylabel="Memory throughput [GB/s]",
                 title=title, # txt= t_eff_evo, 
                 markersize=3, markershape=:circle)
        benchmark_results[bench] = p
    end
    p1 = benchmark_results["ap"]
    p2 = benchmark_results["kp"]
    p3 = plot(p1, p2; layout=(1, 2))
    display(p3)
    png(p3, "images/pde/par/memcopy.png")
end
main()