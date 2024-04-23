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
    labelfontsize=16, tickfontsize=8, titlefontsize=18)

# Create macros that computes difference 
macro d_xa(A)
    esc(:($A[ix+1, iy] - $A[ix, iy]))
end
macro d_ya(A)
    esc(:($A[ix, iy+1] - $A[ix, iy]))
end

# @tturbo: Loop vectorization
# Perform diffusion using 2D array using Parallel computing
function compute_flux!(qDx, qDy, Pf, _dc_dx, _dc_dy, _1_θ_dτ)
    nx, ny = size(Pf)
    #qDx[2:end-1, :] .-= (qDx[2:end-1, :] .+ _dc_dx .* diff(Pf, dims=1)) .* _1_θ_dτ
    @tturbo for iy=1:ny
        for ix=1:nx-1
            @inbounds qDx[ix+1, iy] -= (qDx[ix+1, iy] + _dc_dx * @d_xa(Pf)) * _1_θ_dτ
        end
    end
    # qDy[:, 2:end-1] .-= (qDy[:, 2:end-1] .+ _dc_dy .* diff(Pf, dims=2)) .* _1_θ_dτ
    @tturbo for iy=1:ny-1
        for ix=1:nx
            @inbounds qDy[ix, iy+1] -= (qDy[ix, iy+1] + _dc_dy * @d_ya(Pf)) * _1_θ_dτ
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
            @inbounds Pf[ix, iy] -= (@d_xa(qDx) * _dx + @d_ya(qDy) * _dy) * _β_dτ
        end
    end
    return nothing
end

function compute!(Pf, qDx, qDy, _dc_dx, _dc_dy, _1_θ_dτ, _dx, _dy, _β_dτ)
    compute_flux!(qDx, qDy, Pf, _dc_dx, _dc_dy, _1_θ_dτ)
    update_Pf!(Pf, qDx, qDy, _dx, _dy, _β_dτ)
    return nothing
end
# nx, ny: the number of grid points
function diffusion_2D_perf_loop_fun(nx, ny, maxiter, do_check=false)
    # physics
    lx, ly = 20.0, 20.0
    dc = 1.0
    # numerics
    ϵtol = 1e-8
    ncheck = ceil(Int, 0.25max(nx, ny))
    cfl = 1.0 / sqrt(2.1)
    re = 2π
    # derived numerics
    dx, dy = lx / nx, ly / ny
    xc, yc = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)
    θ_dτ = max(lx, ly) / re / cfl / min(dx, dy)
    β_dτ = (re * dc) / (cfl * min(dx, dy) * max(lx, ly))
    # array initialisation
    Pf = @. exp(-(xc - lx / 2)^2 - (yc' - ly / 2)^2)
    qDx = zeros(Float64, nx + 1, ny)
    qDy = zeros(Float64, nx, ny + 1)
    r_Pf = zeros(nx, ny)
    # iteration loop
    iter = 1; err_Pf = 2ϵtol
    iter_nxs = Float64[]
    errs = Float64[]
    t_toc = 0
    niter = 0
    # Precompute scalars, removing division and casual array
    _dc_dx, _dc_dy = dc/dx, dc/dy
    _1_θ_dτ = 1.0 ./ (1.0 + θ_dτ)
    _dx, _dy = 1.0/dx, 1.0/dy
    _β_dτ = 1.0 ./ β_dτ
    anim = Animation() # Create animation object
    # iteration loop
    t_tic = Base.time()
    while iter <= maxiter
    # while err >= ϵtol && iter < maxiter
        #Compute diffusion physics 
        compute!(Pf, qDx, qDy, _dc_dx, _dc_dy, _1_θ_dτ, _dx, _dy, _β_dτ)
        # Check the iteration results and calculate the errors 
        if do_check && (iter % ncheck == 0)
            r_Pf .= diff(qDx, dims=1) .* _dx  .+ diff(qDy, dims=2) .* _dy # residual
            err = maximum(abs.(r_Pf)) # Get the maximal value
            push!(iter_nxs, iter / nx)
            push!(errs, err)
            println("Complete iter/nx = $(round(iter/nx, sigdigits=3)), Error = $(round(err, sigdigits=3))")
            # Plot the results
            opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]),
                clims=(0.0, 1.0), c=:turbo, xlabel="Lx", ylabel="Ly",
                title="time = $(round(iter/nx, sigdigits=3))")
            p1 = heatmap(xc, yc, Pf'; opts)  # C' denotes transpose 
            p2 = plot(iter_nxs, errs;
                xlabel="iter/nx", ylabel="err",
                yscale=:log10, grid=true, markershape=:circle, markersize=10)
            p3 = plot(p1, p2; layout=(1, 2))
            frame(anim, p3)
            # display(p3)
            png(p3, "images/pde/par/diffusion_2D_perf_loop_fun.png")
        end
        niter = iter # Total number of iterations
        iter += 1
    end
    t_toc = Base.time() - t_tic
    # Save to gif file
    if do_check
        path = "images/pde/par/diffusion_2D_perf_loop_fun.gif"
        gif(anim, path, fps=FPS)
        println("Save gif result to $path")
        # Performance metrics
        A_eff = (3 * 2) / 1e9 * nx * ny * sizeof(Float64)  # Effective main memory access per iteration [GB]
        t_it = t_toc / niter                      # Execution time per iteration [s]
        T_eff = A_eff / t_it                     # Effective memory throughput [GB/s] 
        println("----------------------------------------------------------------") 
        println("Complete the simulation of memcopy of nx = $(nx) and ny = $(ny) ")
        println("Total number of iterations = $(niter), Elapsed time = $(round(t_toc, sigdigits=3)) [s]")
        println("Execution time per iteration = $(round(t_it, sigdigits=3)) [s]")
        println("Memory access per iteration = $(round(A_eff, sigdigits=3)) [GB], Memory throughput = $(round(T_eff, sigdigits=3)) [GB/s]")
    end
    
end





# Benchmark results
t_its = Float64[]
T_effs = Float64[]
ny = 512
nxs = 16 * 2 .^ (1:8)
for i in eachindex(nxs)
    t_it, A_eff, T_eff = memcopy(nxs[i], ny, "btool")
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
