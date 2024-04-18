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

# Perform wave propagation
function wave_propagation()
    # physics
    lx = 20.0 # domain length
    dc = 1.0  # coefficient 
    ρ, β = 1.0, 1.0
    # numerics
    nx   = 200 # the number of grid points
    nvis = 1 # frequency of updating the visualisation
    # derived numerics
    dx = lx/nx      #grid spacing dx 
    dt = dx/sqrt(1/β/ρ)
    nt = nx^2 ÷ 100 # time 
    # creates a linear range of numbers
    xc = LinRange(dx/2, lx-dx/2, nx)
    println("size(xc) = ", size(xc))
    println("Constant: nt = ", nt, " dc = ", dc, " dx = ", dx, " dt = ", dt)
    # array initialisation
    Pr = @. exp(-(xc - lx/4)^2) #pressure
    Pr_i = copy(Pr)
    Vx = zeros(Float64, nx-1)    
    # Go through each time step
    anim = @animate for it=1:nt
        # diffusion physics
        Vx .-= dt/ρ .* diff(Pr) ./dx  #velocity update
        # println("Vx = ", Vx)
        Pr[2:end-1] .-= dt/β .* diff(Vx) ./dx
        # Plot the results
        plot(xlim=(0, lx), ylim=(-1.0, 1.0), 
             linewidth=10.0, thickness_scaling=1,
             legend=:bottomright, xlabel="lx", ylabel="pressure")
        plot!(xc, [Pr_i, Pr], label="pressure", title="Time = $(round(it*dt, digits=1))")
        # plot!(xc[1:end-1].+dx/2, Vx, label="velocity update", markershape=:circle, markersize=5, framestyle=:box)
    end every nvis
    # # Save to gif file
    path = "images/pde_cpu/wave_propagation_1D.gif"
    gif(anim, path, fps=15)
    println("Save gif result to $path")
end
# println("Complete the simulation of wave_propagation ",  wave_propagation())


# Perform advection
function advection()
    # physics
    lx = 20.0 # domain length
    vx = 1.0  # coefficient 
    # numerics
    nx = 200  # the number of grid points
    nvis = 1 # frequency of updating the visualisation
    dx = lx/nx  # grid spacing dx 
    # derived numerics
    dt = dx/abs(vx)
    nt = nx # 
    dx = lx/nx  # grid spacing dx 
    println("Constant: dx = ", dx, " dt = ", dt, " nt = ", nt, " nt÷2 = ", nt÷2)
    # creates a linear range of numbers
    xc = LinRange(dx/2, lx-dx/2, nx)
    # array initialisation
    C = @. exp(-(xc - lx/4)^2)
    C_i  = copy(C)
    println("size(C) = ", size(C))
    anim = @animate for it=1:nt
        C_n = C.^2
        C[2:end] .-= dt.*max(vx, 0.0).*diff(C_n)./dx # if vx >0
        C[1:end-1] .-= dt.*min(vx, 0.0).*diff(C_n)./dx # if vx <0
        if (it % (nt÷2)) == 0 # ÷ integer divide
            vx = -vx
        end
        # println("title", round(it*dt, digits=1))
        #println("it = ", it, " C = ", C[1:5])
        plot(xlim = (0, lx), ylim = (-0.1, 1.1), 
            linewidth=:3.0, legend=:bottomright,
            xlabel="lx", ylabel="advection")
        plot!(xc,  [C_i, C], label="advection",  title="Time = $(round(it*dt, digits=1))")
    end every nvis
    # # Save to gif file
    path = "images/pde_cpu/advection_1D.gif"
    gif(anim, path, fps=15)
    println("Save gif result to $path")
end
# println("Complete the simulation of advection ",  advection())


# Perform diffusion using 1D arrays
function diffusion_1D()
    # physics
    lx = 20.0
    dc = 1.0
    # numerics
    nx = 200
    nvis = 5
    # derived numerics
    dx = lx / nx
    dt = dx^2 / dc / 2
    nt = nx^2 ÷ 100
    println("Constants: nt = $(nt)")
    xc = LinRange(dx / 2, lx - dx / 2, nx)
    # array initialisation
    C = @. 0.5cos(9π * xc / lx) + 0.5
    C_i = copy(C)
    qx = zeros(Float64, nx - 1)
    # time loop
    anim = @animate for it = 1:nt
        qx .= .-dc .* diff(C) ./ dx
        C[2:end-1] .-= dt .* diff(qx) ./ dx 
        plot(xlim=(0, lx), ylim=(-1.0, 1.0), 
             legend=:bottomright, xlabel="lx", ylabel="concentration")       
        plot!(xc, [C_i, C]; 
                xlims=(0, lx), ylims=(-0.1, 1.1),
                xlabel="lx", ylabel="Concentration",
                title="Time = $(round(it*dt, sigdigits=1))")
        # display(plot(xc, [C_i, C]; 
        #       xlims=(0, lx), ylims=(-0.1, 1.1),
        #       xlabel="lx", ylabel="Concentration",
        #       title="Time = $(round(it*dt, sigdigits=1))"))
    end every nvis
    # println("err_evo ", err_evo[length(err_evo)-5:length(err_evo)])    
    # Save to gif file
    path = "images/pde_cpu/diffusion_1D.gif"
    gif(anim, path, fps=10)
    println("Save gif result to $path")
end
# println("Complete the simulation of diffusion ", diffusion_1D())

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
    anim = @animate while err>= ϵtol && iter < maxiter
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
            display(p3)
            png(p3, "images/pde_cpu/steady_diffusion_1D.png")
        end
        iter += 1
    end 
    # println("err_evo ", err_evo[length(err_evo)-5:length(err_evo)])    
    # Save to gif file
    path = "images/pde_cpu/steady_diffusion_1D.gif"
    gif(anim, path, fps=15)   
    println("Save gif result to $path")
end
# println("Complete the simulation of steady diffusion ", steady_diffusion_1D())

# Perform diffusion + reaction using 1D array
function steady_diffusion_reaction_1D()
    # physics
    lx = 20.0 # domain length
    dc = 1.0 # diffusion coefficient
    C_eq = 0.1
    da = 10.0 #Damköhler number
    re = π + sqrt(π^2 + da)  # Large value leads to a long convergence time
    ρ = (lx/(dc*re))^2 # Optimal value for convergence
    ξ = lx^2/dc/da
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
    println("Constant: da = $(da) dc = $(dc) dx = $(dx) dτ = $(dτ) ρ = $(ρ)")   
    # Iteration loop replaces the time time step it with iteration counter iter
    iter = 1; err = 2ϵtol; iter_nxs = Float64[]; errs = Float64[]
    println("Loop condition: err>= $(ϵtol) and  iter < $(maxiter)")  
    anim = @animate while err>= ϵtol && iter < maxiter
    # @gif while err>= ϵtol && iter < maxiter
        #diffusive flux 
        qx .-= dτ ./ (ρ .+ dτ / dc) .* (qx ./ dc .+ diff(C) ./ dx) #dτ ./ (ρ * dc .+ dτ) .* (qx .+ dc .* diff(C) ./dx )
        # flux balance equation
        C[2:end-1] .-= dτ ./ (1.0 .+ dτ / ξ) .* ((C[2:end-1] .- C_eq) ./ ξ .+ diff(qx) ./ dx) #dτ ./ (1.0 + dτ/ξ) .* ( (C[2:end-1] .- C_eq)./ξ + diff(qx) ./dx)
        if iter % ncheck == 0
            err = maximum(abs.(diff(dc .* diff(C) ./ dx) ./ dx .- (C[2:end-1] .- C_eq) ./ ξ)) #err = maximum( abs.( diff(dc .* diff(C) ./dx)./dx .- (C[2:end-1] .- C_eq) ./ξ) ) # Update err 
            push!(iter_nxs, iter/nx)
            push!(errs, err)
            println("Complete iter/nx = $(iter/nx), da = $(da), Error = $(round(err, sigdigits=2))")     
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
            display(p3)
            png(p3, "images/pde_cpu/steady_diffusion_reaction_1D.png")
        end
        iter += 1
    end 
    # println("err_evo ", err_evo[length(err_evo)-5:length(err_evo)])    
    # Save to gif file
    path = "images/pde_cpu/steady_diffusion_reaction_1D.gif"
    gif(anim, path, fps=15)   
    println("Save gif result to $path")
end
# println("Complete the simulation of steady_diffusion_reaction_1D ", steady_diffusion_reaction_1D())

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
            png(p3, "images/pde_cpu/diffusion_2D.png")
        end
        iter += 1
    end 
    # println("err_evo ", err_evo[length(err_evo)-5:length(err_evo)])    
    # Save to gif file
    path = "images/pde_cpu/diffusion_2D.gif"
    gif(anim, path, fps=5)   
    println("Save gif result to $path")
end
# println("Complete the simulation of diffusion_2D ", diffusion_2D())


# Perform diffusion using 2D array
function diffusion_2D_Teff(display=false)
    # physics
    lx, ly = 20.0, 20.0 # domain length
    dc = 1.0 # diffusion coefficient
    # numerics 
    nx, ny = 127*5, 127*5 # the number of grid points
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
    println("Loop condition: err>= $(ϵtol) and  iter < $(maxiter)")  
    # anim = @animate while err>= ϵtol && iter < maxiter
    while err>= ϵtol && iter < maxiter
        if iter == 11
            t_tic = Base.time() # Start time 
        end
        #diffusive flux 
        qx[2:end-1, :] .-= (qx[2:end-1, :]  .+ dc .* (diff(C, dims=1) ./dx)) ./ (1.0 + Θ_dτ) 
        qy[:, 2:end-1] .-= (qy[:, 2:end-1]  .+ dc .* (diff(C, dims=2) ./dy)) ./ (1.0 + Θ_dτ) 
        # flux balance equation
        C .-= (diff(qx, dims=1) ./ dx .+ diff(qy, dims=2) ./ dy) ./ β_dτ
        # Check the iteration results and calculate the errors 
        if display && (iter % ncheck == 0)
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
                      yscale=:log10, grid=true, markershape=:circle, markersize=10)
            p3 = plot!(p1, p2; size=(1200, 400),
                       layout=(1,2), margin=10px, legend=false, 
                       labelfontsize=12, tickfontsize=12, titlefontsize=12)
            display(p3)
            png(p3, "images/pde_cpu/diffusion_2D_parallel.png")
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
        path = "images/pde_cpu/diffusion_2D_Teff.gif"
        gif(anim, path, fps=5)   
        println("Save gif result to $path")
    end
    return
end
println("Complete the simulation of diffusion_2D_Teff ", diffusion_2D_Teff())

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

# Perform wave propagation
function wave_propagation()
    # physics
    lx = 20.0 # domain length
    dc = 1.0  # coefficient 
    ρ, β = 1.0, 1.0
    # numerics
    nx   = 200 # the number of grid points
    nvis = 1 # frequency of updating the visualisation
    # derived numerics
    dx = lx/nx      #grid spacing dx 
    dt = dx/sqrt(1/β/ρ)
    nt = nx^2 ÷ 100 # time 
    # creates a linear range of numbers
    xc = LinRange(dx/2, lx-dx/2, nx)
    println("size(xc) = ", size(xc))
    println("Constant: nt = ", nt, " dc = ", dc, " dx = ", dx, " dt = ", dt)
    # array initialisation
    Pr = @. exp(-(xc - lx/4)^2) #pressure
    Pr_i = copy(Pr)
    Vx = zeros(Float64, nx-1)    
    # Go through each time step
    anim = @animate for it=1:nt
        # diffusion physics
        Vx .-= dt/ρ .* diff(Pr) ./dx  #velocity update
        # println("Vx = ", Vx)
        Pr[2:end-1] .-= dt/β .* diff(Vx) ./dx
        # Plot the results
        plot(xlim=(0, lx), ylim=(-1.0, 1.0), 
             linewidth=10.0, thickness_scaling=1,
             legend=:bottomright, xlabel="lx", ylabel="pressure")
        plot!(xc, [Pr_i, Pr], label="pressure", title="Time = $(round(it*dt, digits=1))")
        # plot!(xc[1:end-1].+dx/2, Vx, label="velocity update", markershape=:circle, markersize=5, framestyle=:box)
    end every nvis
    # # Save to gif file
    path = "images/pde_cpu/wave_propagation_1D.gif"
    gif(anim, path, fps=15)
    println("Save gif result to $path")
end
# println("Complete the simulation of wave_propagation ",  wave_propagation())


# Perform advection
function advection()
    # physics
    lx = 20.0 # domain length
    vx = 1.0  # coefficient 
    # numerics
    nx = 200  # the number of grid points
    nvis = 1 # frequency of updating the visualisation
    dx = lx/nx  # grid spacing dx 
    # derived numerics
    dt = dx/abs(vx)
    nt = nx # 
    dx = lx/nx  # grid spacing dx 
    println("Constant: dx = ", dx, " dt = ", dt, " nt = ", nt, " nt÷2 = ", nt÷2)
    # creates a linear range of numbers
    xc = LinRange(dx/2, lx-dx/2, nx)
    # array initialisation
    C = @. exp(-(xc - lx/4)^2)
    C_i  = copy(C)
    println("size(C) = ", size(C))
    anim = @animate for it=1:nt
        C_n = C.^2
        C[2:end] .-= dt.*max(vx, 0.0).*diff(C_n)./dx # if vx >0
        C[1:end-1] .-= dt.*min(vx, 0.0).*diff(C_n)./dx # if vx <0
        if (it % (nt÷2)) == 0 # ÷ integer divide
            vx = -vx
        end
        # println("title", round(it*dt, digits=1))
        #println("it = ", it, " C = ", C[1:5])
        plot(xlim = (0, lx), ylim = (-0.1, 1.1), 
            linewidth=:3.0, legend=:bottomright,
            xlabel="lx", ylabel="advection")
        plot!(xc,  [C_i, C], label="advection",  title="Time = $(round(it*dt, digits=1))")
    end every nvis
    # # Save to gif file
    path = "images/pde_cpu/advection_1D.gif"
    gif(anim, path, fps=15)
    println("Save gif result to $path")
end
# println("Complete the simulation of advection ",  advection())


# Perform diffusion using 1D arrays
function diffusion_1D()
    # physics
    lx = 20.0
    dc = 1.0
    # numerics
    nx = 200
    nvis = 5
    # derived numerics
    dx = lx / nx
    dt = dx^2 / dc / 2
    nt = nx^2 ÷ 100
    println("Constants: nt = $(nt)")
    xc = LinRange(dx / 2, lx - dx / 2, nx)
    # array initialisation
    C = @. 0.5cos(9π * xc / lx) + 0.5
    C_i = copy(C)
    qx = zeros(Float64, nx - 1)
    # time loop
    anim = @animate for it = 1:nt
        qx .= .-dc .* diff(C) ./ dx
        C[2:end-1] .-= dt .* diff(qx) ./ dx 
        plot(xlim=(0, lx), ylim=(-1.0, 1.0), 
             legend=:bottomright, xlabel="lx", ylabel="concentration")       
        plot!(xc, [C_i, C]; 
                xlims=(0, lx), ylims=(-0.1, 1.1),
                xlabel="lx", ylabel="Concentration",
                title="Time = $(round(it*dt, sigdigits=1))")
        # display(plot(xc, [C_i, C]; 
        #       xlims=(0, lx), ylims=(-0.1, 1.1),
        #       xlabel="lx", ylabel="Concentration",
        #       title="Time = $(round(it*dt, sigdigits=1))"))
    end every nvis
    # println("err_evo ", err_evo[length(err_evo)-5:length(err_evo)])    
    # Save to gif file
    path = "images/pde_cpu/diffusion_1D.gif"
    gif(anim, path, fps=10)
    println("Save gif result to $path")
end
# println("Complete the simulation of diffusion ", diffusion_1D())

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
    anim = @animate while err>= ϵtol && iter < maxiter
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
            display(p3)
            png(p3, "images/pde_cpu/steady_diffusion_1D.png")
        end
        iter += 1
    end 
    # println("err_evo ", err_evo[length(err_evo)-5:length(err_evo)])    
    # Save to gif file
    path = "images/pde_cpu/steady_diffusion_1D.gif"
    gif(anim, path, fps=15)   
    println("Save gif result to $path")
end
# println("Complete the simulation of steady diffusion ", steady_diffusion_1D())

# Perform diffusion + reaction using 1D array
function steady_diffusion_reaction_1D()
    # physics
    lx = 20.0 # domain length
    dc = 1.0 # diffusion coefficient
    C_eq = 0.1
    da = 10.0 #Damköhler number
    re = π + sqrt(π^2 + da)  # Large value leads to a long convergence time
    ρ = (lx/(dc*re))^2 # Optimal value for convergence
    ξ = lx^2/dc/da
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
    println("Constant: da = $(da) dc = $(dc) dx = $(dx) dτ = $(dτ) ρ = $(ρ)")   
    # Iteration loop replaces the time time step it with iteration counter iter
    iter = 1; err = 2ϵtol; iter_nxs = Float64[]; errs = Float64[]
    println("Loop condition: err>= $(ϵtol) and  iter < $(maxiter)")  
    anim = @animate while err>= ϵtol && iter < maxiter
    # @gif while err>= ϵtol && iter < maxiter
        #diffusive flux 
        qx .-= dτ ./ (ρ .+ dτ / dc) .* (qx ./ dc .+ diff(C) ./ dx) #dτ ./ (ρ * dc .+ dτ) .* (qx .+ dc .* diff(C) ./dx )
        # flux balance equation
        C[2:end-1] .-= dτ ./ (1.0 .+ dτ / ξ) .* ((C[2:end-1] .- C_eq) ./ ξ .+ diff(qx) ./ dx) #dτ ./ (1.0 + dτ/ξ) .* ( (C[2:end-1] .- C_eq)./ξ + diff(qx) ./dx)
        if iter % ncheck == 0
            err = maximum(abs.(diff(dc .* diff(C) ./ dx) ./ dx .- (C[2:end-1] .- C_eq) ./ ξ)) #err = maximum( abs.( diff(dc .* diff(C) ./dx)./dx .- (C[2:end-1] .- C_eq) ./ξ) ) # Update err 
            push!(iter_nxs, iter/nx)
            push!(errs, err)
            println("Complete iter/nx = $(iter/nx), da = $(da), Error = $(round(err, sigdigits=2))")     
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
            display(p3)
            png(p3, "images/pde_cpu/steady_diffusion_reaction_1D.png")
        end
        iter += 1
    end 
    # println("err_evo ", err_evo[length(err_evo)-5:length(err_evo)])    
    # Save to gif file
    path = "images/pde_cpu/steady_diffusion_reaction_1D.gif"
    gif(anim, path, fps=15)   
    println("Save gif result to $path")
end
# println("Complete the simulation of steady_diffusion_reaction_1D ", steady_diffusion_reaction_1D())

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
            png(p3, "images/pde_cpu/diffusion_2D.png")
        end
        iter += 1
    end 
    # println("err_evo ", err_evo[length(err_evo)-5:length(err_evo)])    
    # Save to gif file
    path = "images/pde_cpu/diffusion_2D.gif"
    gif(anim, path, fps=5)   
    println("Save gif result to $path")
end
# println("Complete the simulation of diffusion_2D ", diffusion_2D())




