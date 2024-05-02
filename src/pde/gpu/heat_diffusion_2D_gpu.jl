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


@inbounds @views macro d_xa(A)
    esc(:(($A[2:end, :] .- $A[1:end-1, :])))
end
@inbounds @views macro d_xi(A)
    esc(:(($A[2:end, 2:end-1] .- $A[1:end-1, 2:end-1])))
end
@inbounds @views macro d_ya(A)
    esc(:(($A[:, 2:end] .- $A[:, 1:end-1])))
end
@inbounds @views macro d_yi(A)
    esc(:(($A[2:end-1, 2:end] .- $A[2:end-1, 1:end-1])))
end
@inbounds @views macro inn(A)
    esc(:($A[2:end-1, 2:end-1]))
end

@inbounds @views function diffusion2D_step_task1!(T2, T, Ci, lam, dt, _dx, _dy)
    qTx = .-lam .* @d_xi(T) .* _dx                              # Fourier's law of heat conduction: qT_x  = -λ ∂T/∂x
    qTy = .-lam .* @d_yi(T) .* _dy                              # ...                               qT_y  = -λ ∂T/∂y
    dTdt = @inn(Ci) .* (.-@d_xa(qTx) .* _dx .- @d_ya(qTy) .* _dy)  # Conservation of energy:           ∂T/∂t = 1/cp (-∂qT_x/∂x - ∂qT_y/∂y)
    @inn(T2) .= @inn(T) .+ dt .* dTdt                               # Update of temperature             T_new = T_old + ∂t ∂T/∂t
    return nothing
end


# Array programming
@inbounds @views function diffusion2D_step_task3!(T2, T, Ci, lam, dt, _dx, _dy)
    T2[2:end-1, 2:end-1] .= T[2:end-1, 2:end-1] .+ dt .* (Ci[2:end-1, 2:end-1] .* (
        .-((-lam .* (T[3:end, 2:end-1] .- T[2:end-1, 2:end-1]) .* _dx)
           .-(-lam .* (T[2:end-1, 2:end-1] .- T[1:end-2, 2:end-1]) .* _dx)) .* _dx
        .-((-lam .* (T[2:end-1, 3:end] .- T[2:end-1, 2:end-1]) .* _dy)
           .-(-lam .* (T[2:end-1, 2:end-1] .- T[2:end-1, 1:end-2]) .* _dy)) .* _dy
        ))
    return nothing
end
# Kernel programming
function diffusion2D_step_task5!(T2, T, Ci, lam, dt, _dx, _dy)
    nx, ny = size(T)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if ix > 1 && ix < nx && iy > 1 && iy < ny
        T2[ix, iy] = T[ix, iy] + dt * (Ci[ix, iy] * (
            -((-lam * (T[ix+1, iy] - T[ix, iy]) * _dx) - (-lam * (T[ix, iy] - T[ix-1, iy]) * _dx)) * _dx
            -
            ((-lam * (T[ix, iy+1] - T[ix, iy]) * _dy) - (-lam * (T[ix, iy] - T[ix, iy-1]) * _dy)) * _dy
        ))
    end
    return nothing
end

# Test function check the output of kernel functions
function diffusion_2D()
    # Physics
    lam = 1.0                                          # Thermal conductivity
    c0 = 2.0                                          # Heat capacity
    lx, ly = 10.0, 10.0                                   # Length of computational domain in dimension x and y
    # Numerics
    nx, ny = 32 * 2, 32 * 2                                   # Number of gridpoints in dimensions x and y
    nt = 10                                        # Number of time steps
    dx = lx / (nx - 1)                                    # Space step in x-dimension
    dy = ly / (ny - 1)                                    # Space step in y-dimension
    _dx, _dy = 1.0 / dx, 1.0 / dy
    array_size = nx * ny * sizeof(Float64)
    if 3array_size > 4e9 # > 4Gib
        return
    end
    # Array initializations
    T = CUDA.zeros(Float64, nx, ny)                      # Temperature
    T2 = CUDA.zeros(Float64, nx, ny)                      # 2nd array for Temperature
    Ci = CUDA.zeros(Float64, nx, ny)                      # 1/Heat capacity    
    # Initial conditions
    Ci .= 1 / c0                                              # 1/Heat capacity (could vary in space)
    T .= CuArray([10.0 * exp(-(((ix - 1) * dx - lx / 2) / 2)^2 - (((iy - 1) * dy - ly / 2) / 2)^2) for ix = 1:size(T, 1), iy = 1:size(T, 2)]) # Initialization of Gaussian temperature anomaly
    T2 .= T                                                 # Assign also T2 to get correct boundary conditions.
    # Time loop
    dt = min(dx^2, dy^2) / lam / maximum(Ci) / 4.1                # Time step for 2D Heat diffusion
    threads = (32, 8)
    blocks = (nx ÷ threads[1], ny ÷ threads[2])
    # Time loop
    for it = 1:nt
        # Run task1
        t_it = @belapsed begin
            diffusion2D_step_task1!($T2, $T, $Ci, $lam, $dt, $_dx, $_dy)
        end
        taks1_T2 = round.(Array(T2), sigdigits=5)
        t_it_task1 = t_it
        T_eff_task1 = (2 * 1 + 1) * 1 / 1e9 * nx * ny * sizeof(Float64) / t_it_task1
        # Run task 3       
        t_it = @belapsed begin
            diffusion2D_step_task3!($T2, $T, $Ci, $lam, $dt, $_dx, $_dy)
            synchronize()
        end
        task3_T2 = round.(Array(T2), sigdigits=5)
        t_it_task3 = t_it
        T_eff_task3 = (2 * 1 + 1) * 1 / 1e9 * nx * ny * sizeof(Float64) / t_it_task3
        # Run task 5
        t_it = @belapsed begin
            @cuda blocks = $blocks threads = $threads diffusion2D_step_task5!($T2, $T, $Ci, $lam, $dt, $_dx, $_dy)
            synchronize()
        end
        t_it_task5 = t_it
        T_eff_task5 = (2 * 1 + 1) * 1 / 1e9 * nx * ny * sizeof(Float64) / t_it_task5
        # Test cases 
        task5_T2 = round.(Array(T2), sigdigits=5)
        @testset "Pf_diffusion_2D Tests" begin
            @test taks1_T2[1, 1:5] == task3_T2[1, 1:5]
            @test taks1_T2[2:6, 2] == task3_T2[2:6, 2]
            @test taks1_T2[13:29, 37] == task3_T2[13:29, 37]
            @test taks1_T2[33:39, end] == task3_T2[33:39, end]
            @test all(taks1_T2 .== task3_T2)
            @test taks1_T2[1, 1:5] == task5_T2[1, 1:5]
            @test taks1_T2[2:6, 2] == task5_T2[2:6, 2]
            @test taks1_T2[13:29, 37] == task5_T2[13:29, 37]
            @test taks1_T2[33:39, end] == task5_T2[33:39, end]
            @test all(taks1_T2 .== task5_T2)
        end
        speedup_Teff_task3 = T_eff_task3/T_eff_task1
        speedup_Teff_task5 = T_eff_task5/T_eff_task1
        speedup = t_it_task1/t_it_task5
        println("--------------------------------------------------------")
        println("Complete iteration $(it) ")
        println("nx = $(nx), ny = $(ny), array size = $(array_size), threads = $(threads), blocks = $(blocks)")
        println("runtime [s]: Task1 = $(round(t_it_task1, sigdigits=3)), task3 = $(round(t_it_task3, sigdigits=3)), task5 = $(round(t_it_task5, sigdigits=3)), ")
        println("throughtputs[GB/s]: Task1 = $(round(T_eff_task1, sigdigits=3)), task3 = $(round(T_eff_task3, sigdigits=3)), task5 = $(round(T_eff_task5, sigdigits=3)), ")
        println("speedup (throughtputs task3/task1) = $(round(speedup_Teff_task3, sigdigits=3))")
        println("speedup (throughtputs task5/task1) = $(round(speedup_Teff_task5, sigdigits=3))")
        println("speedup (runtime task5/task1) = $(round(speedup, sigdigits=3))")
        # Swap the aliases T and T2 (does not perform any array copy)
        T, T2 = T2, T
    end
end
diffusion_2D()



