# # Install Cuda in Julia
import Pkg
Pkg.add("CUDA") # Install Cuda
Pkg.add("BenchmarkTools") # Install benchmark tools
Pkg.add("Plots") # Install Plots 
using CUDA 
using Plots, BenchmarkTools, Plots.PlotMeasures, Printf, Test, Base.Threads
using LoopVectorization # using @tturbo: Loop vectorization


# Default Plot options 
default(size=(1200, 400), framestyle=:box, label=false, margin=40px,
    grid=true, linewidth=6.0, thickness_scaling=1,
    labelfontsize=16, tickfontsize=8, titlefontsize=18)
# Select the first GPU device
GPU_ID = 0
device!(GPU_ID) #Select device 0 


# Create macros that computes difference 
macro d_xa(A)
    esc(:($A[ix+1, iy] - $A[ix, iy]))
end
macro d_ya(A)
    esc(:($A[ix, iy+1] - $A[ix, iy]))
end

function compute_flux!(qDx, qDy, Pf, _dc_dx, _dc_dy, _1_θ_dτ)
    nx, ny = size(Pf)
    #qDx[2:end-1, :] .-= (qDx[2:end-1, :] .+ _dc_dx .* diff(Pf, dims=1)) .* _1_θ_dτ
    @tturbo for iy=1:ny
        for ix=1:nx-1
            @inbounds qDx[ix+1, iy] -= (qDx[ix+1, iy] + _dc_dx * (Pf[ix+1, iy] - Pf[ix, iy]) ) * _1_θ_dτ
        end
    end
    # qDy[:, 2:end-1] .-= (qDy[:, 2:end-1] .+ _dc_dy .* diff(Pf, dims=2)) .* _1_θ_dτ
    @tturbo for iy=1:ny-1
        for ix=1:nx
            @inbounds qDy[ix, iy+1] -= (qDy[ix, iy+1] + _dc_dy * (Pf[ix, iy+1] - Pf[ix, iy])) * _1_θ_dτ
        end
    end
    return nothing
end

function update_Pf!(Pf, qDx, qDy, _dx, _dy, _β_dτ)
    nx, ny = size(Pf)
    # C .-= (diff(qDx, dims=1) ./ dx .+ diff(qDy, dims=2) ./ dy) ./ β_dτ
    @tturbo for iy = 1:ny
        for ix = 1:nx
            # C[ix, iy] -= ((qx[ix+1, iy] -qx[ix, iy]) * _dx + (qy[ix, iy+1] - qy[ix, iy]) * _dy) * _β_dτ
            @inbounds Pf[ix, iy] -= ((qDx[ix+1, iy] - qDx[ix, iy]) * _dx + (qDy[ix, iy+1] - qDy[ix, iy]) * _dy) * _β_dτ
        end
    end
    return nothing
end

function compute!(Pf, qDx, qDy, _dc_dx, _dc_dy, _1_θ_dτ, _dx, _dy, _β_dτ)
    compute_flux!(qDx, qDy, Pf, _dc_dx, _dc_dy, _1_θ_dτ)
    update_Pf!(Pf, qDx, qDy, _dx, _dy, _β_dτ)
    return nothing
end

function compute_flux_gpu!(qDx_d, qDy_d, Pf_d, _dc_dx, _dc_dy, _1_θ_dτ)
    nx, ny = size(Pf_d)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y 
    if (ix <=nx-1) && (iy<=ny)
        qDx_d[ix+1, iy] -= (qDx_d[ix+1, iy] + _dc_dx * (Pf_d[ix+1, iy] - Pf_d[ix, iy]) ) * _1_θ_dτ
    end
    if (ix <=nx) && (iy<=ny-1)
        qDy_d[ix, iy+1] -= (qDy_d[ix, iy+1] + _dc_dy * (Pf_d[ix, iy+1] - Pf_d[ix, iy])) * _1_θ_dτ
    end
    return nothing
end


function update_Pf_gpu!(Pf_d, qDx_d, qDy_d, _dx, _dy, _β_dτ)
    nx, ny = size(Pf_d)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y      
    if (ix >= 1 && ix <= nx) && (iy >= 1 && iy <= ny)
        Pf_d[ix, iy] -= ((qDx_d[ix+1, iy] - qDx_d[ix, iy]) * _dx + (qDy_d[ix, iy+1] - qDy_d[ix, iy]) * _dy) * _β_dτ
    end
    return nothing
end

function diffusion_2D_perf_loop_fun_gpu(do_check=false)
    nx, ny, maxiter = 32 * 2, 32 * 2, 10
    # physics
    lx, ly = 20.0, 20.0
    dc = 1.0
    # numerics
    cfl = 1.0 / sqrt(2.1)
    re = 2π
    # derived numerics
    dx, dy = lx / nx, ly / ny
    xc, yc = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)
    θ_dτ = max(lx, ly) / re / cfl / min(dx, dy)
    β_dτ = (re * dc) / (cfl * min(dx, dy) * max(lx, ly))
    # Precompute scalars, removing division and casual array
    _dc_dx, _dc_dy = dc / dx, dc / dy
    _1_θ_dτ = 1.0 / (1.0 + θ_dτ)
    _dx, _dy = 1.0 / dx, 1.0 / dy
    _β_dτ = 1.0 / β_dτ
    # Array Initialization on CPU
    Pf = @. exp(-(xc - lx / 2)^2 - (yc' - ly / 2)^2)
    println("=== CPU version ===\n Before: Pf[1:5, 2] = $(Array(Pf)[1:5, 2])")
    qDx = zeros(Float64, nx + 1, ny)
    qDy = zeros(Float64, nx, ny + 1)
    r_Pf = zeros(Float64, nx, ny)
    # Array initialization on GPU device
    Pf_d = CUDA.zeros(Float64, nx, ny)
    Pf_d .= CuArray(@. exp(-(xc - lx / 2)^2 - (yc' - ly / 2)^2))
    println("=== GPU version ===\n Before: Pf[1:5, 2] = $(Array(Pf_d)[1:5, 2])")
    @test round.(Array(Pf_d)[1:5, 2], sigdigits=15) == round.([2.909080350594841e-82,
                                                              1.2395757965481621e-79,
                                                              4.344775268612946e-77,
                                                              1.2526749897075509e-74,
                                                              2.970888512615668e-72], sigdigits=15)
    @test all(Pf .== Array(Pf_d))
    qDx_d = CUDA.zeros(Float64, nx + 1, ny)
    qDy_d = CUDA.zeros(Float64, nx, ny + 1)
    r_Pf_d = CUDA.zeros(Float64, nx, ny)    
    # CUDA Setting
    threads = (32, 2)
    blocks = (nx ÷ threads[1], ny ÷ threads[2])
    #iteration loop
    ncheck = ceil(Int, 0.25max(nx, ny)); println("ncheck = $(ncheck)")
    for iter=1:maxiter
        t_it = @belapsed begin 
                compute!($Pf, $qDx, $qDy, $_dc_dx, $_dc_dy, $_1_θ_dτ, $_dx, $_dy, $_β_dτ)
        end
        println("Complete iteration $(iter) on CPU in $(round(t_it, sigdigits=3)) [s]")
        
        t_it = @belapsed begin
                    CUDA.@sync @cuda blocks=$blocks threads=$threads compute_flux_gpu!($qDx_d, $qDy_d, $Pf_d, $_dc_dx, $_dc_dy, $_1_θ_dτ)
                    CUDA.@sync @cuda blocks=$blocks threads=$threads update_Pf_gpu!($Pf_d, $qDx_d, $qDy_d, $_dx, $_dy, $_β_dτ)
                end
        println("Complete iteration $(iter) on GPU in $(round(t_it, sigdigits=3)) [s]")
        
        Pf_cpu = round.(Pf, sigdigits=10)
        Pf_gpu = round.(Array(Pf_d), sigdigits=10)
        println("Pf_cpu[1:5, 2] = $(Pf_cpu[1:5, 2])")
        println("Pf_gpu[1:5, 2] = $(Pf_gpu[1:5, 2])")
        @test all(Pf_cpu .== Pf_gpu)
    end
    Pf_cpu = round.(Pf, sigdigits=10)
    Pf_gpu = round.(Array(Pf_d), sigdigits=10)
    println("After: Pf_cpu[1:5, 2] = $(Pf_cpu[1:5, 2])")
    println("After: Pf_gpu[1:5, 2] = $(Pf_gpu[1:5, 2])")
    @test all( Pf_gpu .== Pf_cpu)
end
diffusion_2D_perf_loop_fun_gpu()


