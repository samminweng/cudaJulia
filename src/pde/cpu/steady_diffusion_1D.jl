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

# Visualize steady diffusion + chemical reaction using 1D array
function steady_diffusion_1D()
    # physics
    lx = 20.0 # domain length
    dc = 1.0 # diffusion coefficient
    re = 2π  # Large value leads to a long convergence time
    ρ = (lx/(dc*re))^2 # Optimal value for convergence
    # numerics (Constants)
    nx = 100 # the number of grid points
    ϵtol = 1e-8
    maxiter = 100nx # Maximal iterations
    ncheck = ceil(Int, 0.25nx) # NUmber of checks
    # derived numerics
    dx = lx/nx      #grid spacing dx 
    dτ = dx/sqrt(1/ρ)   # Replace time t with pseudo-time τ
    xc = LinRange(dx/2, lx-dx/2, nx) # grid center
    # array initialisation
    # C = @. 0.5cos(9π*xc/lx)+0.5 # concentration field C
    C = @. 1.0 + exp( -(xc-lx/4)^2 ) - xc/lx
    C_i = copy(C)
    qx = zeros(Float64, nx-1) # diffusive flux in the x direction qx
    println("Constant: dc = $(dc) dx = $(dx) d⬆ = $(dτ) ρ = $(ρ)")   
    # Iteration loop replaces the time time step it with iteration counter iter
    iter = 1; err = 2ϵtol; iter_nxs = Float64[]; errs = Float64[]
    println("Loop condition: err>= $(ϵtol) and  iter < $(maxiter)")  
    anim = Animation() 
    while err>= ϵtol && iter < maxiter
    # @gif while err>= ϵtol && iter < maxiter
        # println("Before it = ", it, " C[1:5] = ", C[1:5])
        # diff: difference between C[ix+1] - C[ix]
        #diffusive flux 
        qx .-= dτ ./ (ρ * dc .+ dτ) .* (qx .+ dc .* diff(C) ./dx )
        # println("it = ", it,  " qx[1:5] = ", qx[1:5])
        # flux balance equation
        C[2:end-1] .-= dτ .* diff(qx) ./ dx
        if iter % ncheck == 0
            err = maximum(abs.(diff(dc .* diff(C) ./dx)/dx)) # Get the maximal value 
            push!(iter_nxs, iter/nx)
            push!(errs, err)
            println("Complete the iteration:  $(iter), iter/nx = $(iter/nx), Error = $(round(err, sigdigits=2))")     
            # Plot the results
            p1 = plot(xc, [C_i, C]; 
                       xlim = (0, lx), ylim = (-0.1, 2.0), 
                       linewidth=:5.0, legend=false,
                       xlabel="lx", ylabel="Concentration", 
                       title="Iter/nx = $(round(iter/nx, sigdigits=3))")
            p2 = plot(iter_nxs, errs;  
                      legend=false, xlim = (0, 7), yscale=:log10,
                      xlabel="iter/nx", ylabel="err",
                      title="Error = $(round(err, sigdigits=2))", 
                      markershape=:circle, markersize=5)
            p3 = plot(p1, p2; size=(1200, 400),
                      layout=(2,1), margin=20px, legend=false, 
                      labelfontsize=12, tickfontsize=12, titlefontsize=12)
            frame(anim, p3)
            #display(p3)
            png(p3, "images/pde/cpu/steady_diffusion_1D.png")
        end
        iter += 1
    end 
    # println("err_evo ", err_evo[length(err_evo)-5:length(err_evo)])    
    # Save to gif file
    path = "images/pde/cpu/steady_diffusion_1D.gif"
    gif(anim, path, fps=1)   
    println("Save gif result to $path")
end
println("Complete the simulation of steady diffusion ", steady_diffusion_1D())