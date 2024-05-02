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
GPU_ID = 0
device!(GPU_ID) #Select device 0 

@inbounds @views macro d_xa(A) esc(:( ($A[2:end  , :     ] .- $A[1:end-1, :     ]) )) end
@inbounds @views macro d_xi(A) esc(:( ($A[2:end  ,2:end-1] .- $A[1:end-1,2:end-1]) )) end
@inbounds @views macro d_ya(A) esc(:( ($A[ :     ,2:end  ] .- $A[ :     ,1:end-1]) )) end
@inbounds @views macro d_yi(A) esc(:( ($A[2:end-1,2:end  ] .- $A[2:end-1,1:end-1]) )) end
@inbounds @views macro  inn(A) esc(:( $A[2:end-1,2:end-1]                          )) end

@inbounds @views function diffusion2D_step!(T, Ci, qTx, qTy, dTdt, lam, dt, _dx, _dy)
    qTx     .= .-lam.*@d_xi(T).*_dx                              # Fourier's law of heat conduction: qT_x  = -λ ∂T/∂x
    qTy     .= .-lam.*@d_yi(T).*_dy                              # ...                               qT_y  = -λ ∂T/∂y
    dTdt    .= @inn(Ci).*(.-@d_xa(qTx).*_dx .- @d_ya(qTy).*_dy)  # Conservation of energy:           ∂T/∂t = 1/cp (-∂qT_x/∂x - ∂qT_y/∂y)
    @inn(T) .= @inn(T) .+ dt.*dTdt                               # Update of temperature             T_new = T_old + ∂t ∂T/∂t
end

function diffusion2D()
    # Physics
    lam = 1.0           # Thermal conductivity
    c0 = 2.0            # Heat capacity
    lx, ly = 10.0, 10.0 # Length of computational domain in dimension x and y
    # Numerics
    nx, ny = 32*2, 32*2         # Number of gridpoints in dimensions x and y
    nt = 100                       # Number of time steps
    dx = lx / (nx - 1)              # Space step in x-dimension
    dy = ly / (ny - 1)              # Space step in y-dimension
    _dx, _dy = 1.0 / dx, 1.0 / dy
    # Array initializations
    T = CUDA.zeros(Float64, nx, ny)              # Temperature
    Ci = CUDA.zeros(Float64, nx, ny)             # 1/Heat capacity
    qTx = CUDA.zeros(Float64, nx - 1, ny - 2)    # Heat flux, x component
    qTy = CUDA.zeros(Float64, nx - 2, ny - 1)    # Heat flux, y component
    dTdt = CUDA.zeros(Float64, nx - 2, ny - 2)   # Change of Temperature in time
    # Initial conditions
    Ci .= 1 / c0                                              # 1/Heat capacity (could vary in space)
    # Initialization of Gaussian temperature anomaly
    # T_h = [10.0 * exp(-(((ix - 1) * dx - lx / 2) / 2)^2 - (((iy - 1) * dy - ly / 2) / 2)^2) for ix = 1:size(T, 1), iy = 1:size(T, 2)] 
    
    T_h = zeros(Float64, nx, ny)
    for ix=1:size(T, 1), iy=1:size(T, 2)
        # println("i = $(i), ix = $(ix), iy= $(iy)")
        T_h[ix, iy] = 10.0 * exp(-(((ix - 1) * dx - lx / 2) / 2)^2 - (((iy - 1) * dy - ly / 2) / 2)^2)
    end
    # Create the array on the device
    T .= CuArray(T_h) 
    # # Time loop
    dt = min(dx^2, dy^2) / lam / maximum(Ci) / 4.1                # Time step for 2D Heat diffusion
    opts = (aspect_ratio=1, xlims=(1, nx), ylims=(1, ny), clims=(0.0, 10.0), c=:davos, xlabel="Lx", ylabel="Ly") # plotting options
    
    anim = Animation() # Create animation object
    for it = 1:nt
        # diffusion2D_step!(T, Ci, qTx, qTy, dTdt, lam, dt, _dx, _dy) # Diffusion time step.
        t_it = @belapsed begin diffusion2D_step!($T, $Ci, $qTx, $qTy, $dTdt, $lam, $dt, $_dx, $_dy); synchronize() end
        numOp = 11 # 7 Read and 4 write operations
        T_tot_lb = numOp * 1/1e9 * nx * ny * sizeof(Float64)/t_it
        if it % 5 == 0
            println("Complete iteration $(it) ")
            p = heatmap(Array(T)'; opts)
            # display(p)            # Visualization
            frame(anim, p)
            #sleep(1)
        end
    end
    # Save a gif 
    path = "images/pde/gpu/heat_diffusion_2D.gif"
    gif(anim, path, fps=15)
    println("Save gif result to $path")
end
diffusion2D()