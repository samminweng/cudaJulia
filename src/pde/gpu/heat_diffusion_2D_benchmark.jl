import Pkg
Pkg.add("CUDA") # Install Cuda 
Pkg.add("BenchmarkTools") # Install benchmark tools
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
GPU_ID = 0
device!(GPU_ID) #Select device 0 


@inbounds @views macro d_xa(A) esc(:( ($A[2:end  , :     ] .- $A[1:end-1, :     ]) )) end
@inbounds @views macro d_xi(A) esc(:( ($A[2:end  ,2:end-1] .- $A[1:end-1,2:end-1]) )) end
@inbounds @views macro d_ya(A) esc(:( ($A[ :     ,2:end  ] .- $A[ :     ,1:end-1]) )) end
@inbounds @views macro d_yi(A) esc(:( ($A[2:end-1,2:end  ] .- $A[2:end-1,1:end-1]) )) end
@inbounds @views macro  inn(A) esc(:( $A[2:end-1,2:end-1]                          )) end

@inbounds @views function diffusion2D_step_orignal!(T2, T, Ci, lam, dt, _dx, _dy)
    qTx     = .-lam.*@d_xi(T).*_dx                              # Fourier's law of heat conduction: qT_x  = -λ ∂T/∂x
    qTy     = .-lam.*@d_yi(T).*_dy                              # ...                               qT_y  = -λ ∂T/∂y
    dTdt    = @inn(Ci).*(.-@d_xa(qTx).*_dx .- @d_ya(qTy).*_dy)  # Conservation of energy:           ∂T/∂t = 1/cp (-∂qT_x/∂x - ∂qT_y/∂y)
    @inn(T2) .= @inn(T) .+ dt.*dTdt                               # Update of temperature             T_new = T_old + ∂t ∂T/∂t
end

@inbounds @views function diffusion2D_step!(T2, T, Ci, lam, dt, _dx, _dy)
    # qTx .= (-lam.*T[2:end  , 2:end-1].*_dx) .- (-lam .* T[1:end-1, 2:end-1].*_dx)   # Fourier's law of heat conduction: qT_x  = -λ ∂T/∂x
    # qTy .= (-lam.*T[2:end-1, 2:end  ].*_dy) .- (-lam .* T[2:end-1, 1:end-1].*_dy)    # ...                               qT_y  = -λ ∂T/∂y
    # dTdt    .= @inn(Ci).*(.-@d_xa(qTx).*_dx .- @d_ya(qTy).*_dy)  # Conservation of energy:           ∂T/∂t = 1/cp (-∂qT_x/∂x - ∂qT_y/∂y)
    # @inn(T) .= @inn(T) .+ dt.*dTdt                               # Update of temperature             T_new = T_old + ∂t ∂T/∂t
    T2[2:end-1, 2:end-1] .= T[2:end-1, 2:end-1] .+  dt .* ( Ci[2:end-1, 2:end-1] .* ( 
                                                           .-(     (-lam .* (T[3:end  , 2:end-1] .- T[2:end-1, 2:end-1]).*_dx) 
                                                                .- (-lam .* (T[2:end-1, 2:end-1] .- T[1:end-2, 2:end-1]).*_dx) ) .*_dx
                                                           .-(     (-lam .* (T[2:end-1, 3:end  ] .- T[2:end-1, 2:end-1]).*_dy) 
                                                                .- (-lam .* (T[2:end-1, 2:end-1] .- T[2:end-1, 1:end-2]).*_dy) ) .*_dy
                                                           )
                                                         )

end

# Test function check the output of kernel functions
function diffusion_2D_test()
    # Numerics
    nx, ny   = 32*2, 32*2                                   # Number of gridpoints in dimensions x and y
    nt       = 10                                         # Number of time steps
    lam = _dx = _dy = dt = rand();
    array_size = nx*ny*sizeof(Float64)
    # Array initializations
    T    = CUDA.rand(Float64, nx, ny);
    Ci   = CUDA.rand(Float64, nx, ny);                      # Temperature
    T2   = CUDA.zeros(Float64, nx, ny)                      # 2nd array for Temperature
    threads = (32, 2)
    blocks =(nx÷threads[1], ny÷threads[2])
    # Time loop
    for it = 1:nt
        # Run on CPU
        t_it = @belapsed begin diffusion2D_step_orignal!($T2, $T, $Ci, $lam, $dt, $_dx, $_dy); end 
        o_T2 = round.(Array(T2), sigdigits=5)
        t_it_cpu = t_it
        T_tot_lb = 3*1/1e9*nx*ny*sizeof(Float64)/t_it
        T_tot_lb_cpu = T_tot_lb
        # Run on GPU
        t_it = @belapsed begin 
                    diffusion2D_step!($T2, $T, $Ci, $lam, $dt, $_dx, $_dy)
                end
        # Test cases 
        g_T2 = round.(Array(T2), sigdigits=5)
        @testset "Pf_diffusion_2D Tests" begin
            @test o_T2[1, 1:5] == g_T2[1, 1:5]
            @test o_T2[2:6, 2] == g_T2[2:6, 2]
            @test o_T2[13:29, 37] == g_T2[13:29, 37]
            @test o_T2[33:39, end] == g_T2[33:39, end]
            @test all(o_T2 .== g_T2)
        end
        speedup = t_it_cpu/t_it
        T_tot_lb = 3*1/1e9*nx*ny*sizeof(Float64)/t_it
        ratio_T_tot_lb = T_tot_lb/T_tot_lb_cpu        
        println("--------------------------------------------------------")
        println("Complete iteration $(it) ")
        println("nx = $(nx), ny = $(ny), array size = $(array_size)")
        println("Speedup (CPU/GPU) = $(round(speedup, sigdigits=3)), CPU time = $(round(t_it_cpu, sigdigits=3)), GPU time = $(round(t_it, sigdigits=3))")
        println("Throughput Ratio (GPU/CPU) = $(round(ratio_T_tot_lb, sigdigits=3)), CPU throughput = $(round(T_tot_lb_cpu, sigdigits=3)), GPU throughput = $(round(T_tot_lb, sigdigits=3)), ")
        # Swap the aliases T and T2 (does not perform any array copy)
        T, T2 = T2, T                                      
    end
end
diffusion_2D_test()

function diffusion_2D(bench="cpu")
    # Physics
    lam      = 1.0                                          # Thermal conductivity
    c0       = 2.0                                          # Heat capacity
    lx, ly   = 10.0, 10.0                                   # Length of computational domain in dimension x and y
    # Numerics
    nx, ny   = 32*2, 32*2                                   # Number of gridpoints in dimensions x and y
    nt       = 100                                         # Number of time steps
    dx       = lx/(nx-1)                                    # Space step in x-dimension
    dy       = ly/(ny-1)                                    # Space step in y-dimension
    _dx, _dy = 1.0/dx, 1.0/dy
    array_size = nx*ny*sizeof(Float64)
    if 3array_size > 4e9 # > 4Gib
        return
    end
    # Array initializations
    T    = CUDA.zeros(Float64, nx, ny)                      # Temperature
    T2   = CUDA.zeros(Float64, nx, ny)                      # 2nd array for Temperature
    Ci   = CUDA.zeros(Float64, nx, ny)                      # 1/Heat capacity    
    # Initial conditions
    Ci .= 1/c0                                              # 1/Heat capacity (could vary in space)
    T  .= CuArray([10.0*exp(-(((ix-1)*dx-lx/2)/2)^2-(((iy-1)*dy-ly/2)/2)^2) for ix=1:size(T,1), iy=1:size(T,2)]) # Initialization of Gaussian temperature anomaly
    T2 .= T                                                 # Assign also T2 to get correct boundary conditions.
    # Time loop
    dt  = min(dx^2,dy^2)/lam/maximum(Ci)/4.1                # Time step for 2D Heat diffusion
    opts = (aspect_ratio=1, xlims=(1, nx), ylims=(1, ny), clims=(0.0, 10.0), c=:davos, xlabel="Lx", ylabel="Ly") # plotting options
    device = if bench == "cpu" Sys.cpu_info()[1].model else CUDA.name(CUDA.current_device()) end
    # Benchmark results
    iters = Int[]
    throughtputs = Float64[]
    anim = Animation() # Create animation object
    for it = 1:nt
        if bench =="cpu"
            # Run on CPU
            t_it = @belapsed begin diffusion2D_step!($T2, $T, $Ci, $lam, $dt, $_dx, $_dy); end    # Diffusion time step.
        else
            # Run on GPU
            t_it = @belapsed begin diffusion2D_step!($T2, $T, $Ci, $lam, $dt, $_dx, $_dy) ; synchronize() end    # Diffusion time step.
        end
        # Compute throughput
        T_tot_lb = 3*1/1e9*nx*ny*sizeof(Float64)/t_it
        if it % 10 == 0
            ## Display the results
            println("--------------------------------------------------------")
            println("Complete iteration $(it) ")
            println("array size = $(array_size), nx = $(nx), ny = $(ny)")
            println("selected device = $( device )")
            println("memory throughtput = $( round(T_tot_lb, sigdigits=5) ) [G/s]")
            p = heatmap(Array(T)'; opts)
            # display(p)            # Visualization
            frame(anim, p)
            push!(iters, it)
            push!(throughtputs, T_tot_lb)
        end
        T, T2 = T2, T                                       # Swap the aliases T and T2 (does not perform any array copy)
    end
    # Save the visualization as a gif 
    path = "images/pde/gpu/heat_diffusion_2D.gif"
    gif(anim, path, fps=5)
    println("Save gif result to $path") 
    # Save benchmark results
    throughtputs .= round.(throughtputs, sigdigits=5)
    # Write to a dataframe
    df = DataFrame()
    df.array_sizes = array_sizes
    df.throughtputs = throughtputs
    path = joinpath("data", "heat_diffusion_2D_benchmark.csv")
    CSV.write(path, df)
    println("Save benchmark results to $(path)")
end
# diffusion_2D("cpu")

