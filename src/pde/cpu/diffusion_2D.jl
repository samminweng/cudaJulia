
# # Install Cuda in Julia
import Pkg
# Pkg.add("CUDA") # Install Cuda
Pkg.add("BenchmarkTools") # Install benchmark tools
Pkg.add("Plots") # Install Plots 
# Pkg.add("BenchmarkPlots")
using Plots, BenchmarkTools, Plots.PlotMeasures, Printf
default(size=(1200, 400), framestyle=:box, label=false, 
        grid=true, linewidth=6.0, thickness_scaling=1,
        labelfontsize=12, tickfontsize=12, titlefontsize=12)

# Perform diffusion using 2D array
function diffusion_2D()
    # physics
    lx, ly = 20.0, 20.0 # domain length
    dc = 1.0 # diffusion coefficient
    # numerics 
    nx, ny = 127, 127 # the number of grid points
    ϵtol = 1e-8
    maxiter = 10max(nx, ny) # Maximal iterations
    ncheck = ceil(Int, 0.1max(nx, ny)) # NUmber of checks
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
    println("Loop condition: err>= $(ϵtol) and  iter < $(maxiter)")  
    anim = @animate while err>= ϵtol && iter < maxiter
    # while err>= ϵtol && iter < maxiter
        #diffusive flux 
        qx[2:end-1, :] .-= (qx[2:end-1, :]  .+ dc .* (diff(C, dims=1) ./dx)) ./ (1.0 + Θ_dτ) 
        qy[:, 2:end-1] .-= (qy[:, 2:end-1]  .+ dc .* (diff(C, dims=2) ./dy)) ./ (1.0 + Θ_dτ) 
        # flux balance equation
        C .-= (diff(qx, dims=1) ./ dx .+ diff(qy, dims=2) ./ dy) ./ β_dτ 
        if iter % ncheck == 0
            r .= diff(qx, dims=1) ./ dx .+ diff(qy, dims=2) ./ dy # residual
            err = maximum(abs.(r)) # Get the maximal value 
            push!(iter_nxs, iter/nx)
            push!(errs, err)
            println("Complete iter/nx = $(round(iter/nx, sigdigits=2)), Error = $(round(err, sigdigits=2))")     
            println("size(C) = $(size(C)) C[1, 2] = $(round(C[1, 2], sigdigits=3)) C'[2,1] = $(round(C'[2, 1], sigdigits=3))")
            # Plot the results
            opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]),
                    clims=(0.0, 1.0), c=:turbo, xlabel="Lx", ylabel="Ly",
                    title="time = $(round(iter/nx, sigdigits=3))")
            p1 = heatmap(xc, yc, C'; # ' denotes transpose 
                         opts)
            p2 = plot(iter_nxs, errs;
                      xlabel="iter/nx",ylabel="err",
                      yscale=:log10, grid=true, markershape=:circle, markersize=10)
            p3 = plot(p1, p2; size=(1200, 400),
                      layout=(1,2), margin=20px, legend=false, 
                      labelfontsize=12, tickfontsize=12, titlefontsize=12)
            display(p3)
            png(p3, "images/pde/cpu/diffusion_2D.png")
        end
        iter += 1
    end 
    # println("err_evo ", err_evo[length(err_evo)-5:length(err_evo)])    
    # Save to gif file
    path = "images/pde/cpu/diffusion_2D.gif"
    gif(anim, path, fps=5)   
    println("Save gif result to $path")
end
println("Complete the simulation of diffusion_2D ", diffusion_2D())