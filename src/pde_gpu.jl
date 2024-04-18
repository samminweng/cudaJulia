# # Install Cuda in Julia
import Pkg
# Pkg.add("CUDA") # Install Cuda
Pkg.add("BenchmarkTools") # Install benchmark tools
Pkg.add("Plots") # Install Plots 
# Pkg.add("BenchmarkPlots")
using Plots, BenchmarkTools, Plots.PlotMeasures, Printf
# Default Plot options 
default(size=(1200, 400), framestyle=:box, label=false, margin=40px,
        grid=true, linewidth=6.0, thickness_scaling=1,
        labelfontsize=18, tickfontsize=8, titlefontsize=18)

# Perform diffusion using 2D array
function diffusion_2D_Teff(debug=true)
    # physics
    lx, ly = 20.0, 20.0 # domain length
    dc = 1.0 # diffusion coefficient
    # numerics 
    nx, ny = 127, 127 # the number of grid points
    ϵtol = 1e-8
    maxiter = 10max(nx, ny) # Maximal iterations
    ncheck = ceil(Int, 0.25max(nx, ny)) # NUmber of checks
    cfl = 1.0/sqrt(2.1)
    re = 2π  # Large value leads to a long convergence time
    # derived numerics
    dx, dy = lx/nx, ly/ny      #grid spacing dx 
    xc, yc = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny)# grid center
    # pseudo-time τ
    Θ_dτ = max(lx, ly)/ re / cfl/ min(dx, dy)
    β_dτ = (re * dc)/ (cfl * min(dx, dy) * max(lx, ly)) 
    # array initialisation
    C = @. exp( -(xc-lx/2)^2 - (yc'-ly/2)^2)
    qx = zeros(Float64, nx+1, ny) # diffusive flux (nx+1, ny)
    qy = zeros(Float64, nx, ny+1)
    r = zeros(nx, ny)
    anim = Animation() # Create animation object
    # Iteration loop replaces the time time step it with iteration counter iter
    iter = 1; err = 2ϵtol; iter_nxs = Float64[]; errs = Float64[]
    t_tic = Base.time() # Start time 
    t_toc = 0; niter = 0;
    println("Loop condition: err >= $(ϵtol) and  iter < $(maxiter)")  
    while err>= ϵtol && iter < maxiter
        #diffusive flux 
        qx[2:end-1, :] .-= (qx[2:end-1, :]  .+ dc .* (diff(C, dims=1) ./dx)) ./ (1.0 + Θ_dτ) 
        qy[:, 2:end-1] .-= (qy[:, 2:end-1]  .+ dc .* (diff(C, dims=2) ./dy)) ./ (1.0 + Θ_dτ) 
        # flux balance equation
        C .-= (diff(qx, dims=1) ./ dx .+ diff(qy, dims=2) ./ dy) ./ β_dτ
        # Check the iteration results and calculate the errors 
        if debug && (iter % ncheck == 0)
            r .= diff(qx, dims=1) ./ dx .+ diff(qy, dims=2) ./ dy # residual
            err = maximum(abs.(r)) # Get the maximal value 
            push!(iter_nxs, iter/nx)
            push!(errs, err)
            println("Complete iter/nx = $(round(iter/nx, sigdigits=3)), Error = $(round(err, sigdigits=3))")
            # Plot the results
            opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]),
                    clims=(0.0, 1.0), c=:turbo, xlabel="Lx", ylabel="Ly", 
                    title="time = $(round(iter/nx, sigdigits=3))")
            p1 = heatmap(xc, yc, C'; opts)  # C' denotes transpose 
            p2 = plot(iter_nxs, errs;
                    xlabel="iter/nx",ylabel="err",
                    yscale=:log10, markershape=:circle, markersize=10)
            p3 = plot(p1, p2; layout=(1,2))
            # display(p3)
            frame(anim, p3) # Save p3 as a frame
            png(p3, "images/pde_gpu/diffusion_2D_Teff.png")
        end
        niter = iter # Total number of iterations
        iter += 1    
    end # End while loop
    t_toc = Base.time() - t_tic # Elapsed time
    # Performance metrics
    A_eff = (3*2)/1e9*nx*ny*sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it = t_toc/niter                      # Execution time per iteration [s]
    T_eff = A_eff / t_it                     # Effective memory throughput [GB/s] 
    println("Total number of iterations = $(niter), Elapsed time = $(round(t_toc, sigdigits=3)) [s]")
    println("Execution time per iteration = $(round(t_it, sigdigits=3)) [s]")
    println("Memory access per iteration = $(round(A_eff, sigdigits=3)) [GB], Memory throughput = $(round(T_eff, sigdigits=3)) [GB/s]") 
    # Save to gif file
    # if (@isdefined anim) && isnothing(anim)
    path = "images/pde_gpu/diffusion_2D_Teff.gif"
    gif(anim, path, fps=1)   
    println("Save gif result to $path")
    # end
end
println("Complete the simulation of diffusion_2D_Teff ", diffusion_2D_Teff())


# Perform diffusion using 2D array using Parallel computing
function diffusion_2D_perf(debug=true)
    # physics
    lx, ly = 20.0, 20.0 # domain length
    dc = 1.0 # diffusion coefficient
    # numerics 
    nx, ny = 127, 127 # the number of grid points
    ϵtol = 1e-8
    maxiter = 10max(nx, ny) # Maximal iterations
    ncheck = ceil(Int, 0.25max(nx, ny)) # NUmber of checks
    cfl = 1.0/sqrt(2.1)
    re = 2π  # Large value leads to a long convergence time
    # derived numerics
    dx, dy = lx/nx, ly/ny      #grid spacing dx 
    xc, yc = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny)# grid center
    # pseudo-time τ
    Θ_dτ = max(lx, ly)/ re / cfl/ min(dx, dy)
    β_dτ = (re * dc)/ (cfl * min(dx, dy) * max(lx, ly)) 
    # array initialisation
    C = @. exp( -(xc-lx/2)^2 - (yc'-ly/2)^2)
    qx = zeros(Float64, nx+1, ny) # diffusive flux (nx+1, ny)
    qy = zeros(Float64, nx, ny+1)
    r = zeros(nx, ny)  
    # Iteration loop replaces the time time step it with iteration counter iter
    iter = 1; err = 2ϵtol; iter_nxs = Float64[]; errs = Float64[]
    t_tic = 0; t_toc = 0; niter = 0;
    # Precompute scalars, removing division and casual array
    _dc_dx, _dc_dy = dc/dx, dc/dy
    _1_θ_dτ = 1.0 ./ 1.0 + Θ_dτ
    _dx, _dy = 1.0/dx, 1.0/dy
    _β_dτ = 1.0 ./ β_dτ 
    println("Loop condition: err>= $(ϵtol) and  iter < $(maxiter)")  
    anim = @animate while err>= ϵtol && iter < maxiter
    # while err>= ϵtol && iter < maxiter
        if iter == 11
            t_tic = Base.time() # Start time 
        end
        #diffusive flux 
        qx[2:end-1, :] .-= (qx[2:end-1, :]  .+ _dc_dx .* diff(C, dims=1)) * _1_θ_dτ 
        qy[:, 2:end-1] .-= (qy[:, 2:end-1]  .+ _dc_dy .* diff(C, dims=2)) * _1_θ_dτ 
        # flux balance equation
        C .-= (diff(qx, dims=1) .* _dx .+ diff(qy, dims=2) .* _dy) * _β_dτ
        # Check the iteration results and calculate the errors 
        if debug && (iter % ncheck == 0)
            r .= diff(qx, dims=1) .* _dx .+ diff(qy, dims=2) .* _dy # residual
            err = maximum(abs.(r)) # Get the maximal value 
            push!(iter_nxs, iter/nx)
            push!(errs, err)
            println("Complete iter/nx = $(round(iter/nx, sigdigits=3)), Error = $(round(err, sigdigits=3))")
            # Plot the results
            opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]),
                    clims=(0.0, 1.0), c=:turbo, xlabel="Lx", ylabel="Ly",
                    title="time = $(round(iter/nx, sigdigits=3))")
            p1 = heatmap(xc, yc, C'; opts)  # C' denotes transpose 
            p2 = plot(iter_nxs, errs;
                      xlabel="iter/nx",ylabel="err",
                      yscale=:log10, grid=true, markershape=:circle, markersize=10)
            p3 = plot!(p1, p2; size=(1200, 400),
                       layout=(1,2), margin=10px, legend=false, 
                       labelfontsize=12, tickfontsize=12, titlefontsize=12)
            display(p3)
            png(p3, "images/pde_cpu/diffusion_2D_perf.png")
        end
        niter = iter # Total number of iterations
        t_toc = Base.time() - t_tic # Elapsed time
        iter += 1    
    end 
    # Performance metrics
    A_eff = (3*2)/1e9*nx*ny*sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it = t_toc/niter                      # Execution time per iteration [s]
    T_eff = A_eff / t_it                     # Effective memory throughput [GB/s] 
    println("Total number of iterations = $(niter), Elapsed time = $(round(t_toc, sigdigits=3)) [s]")
    println("Execution time per iteration = $(round(t_it, sigdigits=3)) [s]")
    println("Memory access per iteration = $(round(A_eff, sigdigits=3)) [GB], Memory throughput = $(round(T_eff, sigdigits=3)) [GB/s]")    
    # Save to gif file
    if (@isdefined anim) && isnothing(anim)
        path = "images/pde_gpu/diffusion_2D_perf.gif"
        gif(anim, path, fps=5)   
        println("Save gif result to $path")
    end
    return
end
# println("Complete the simulation of diffusion_2D_perf ", diffusion_2D_perf())