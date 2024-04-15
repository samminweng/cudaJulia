# # Install Cuda in Julia
import Pkg
# Pkg.add("CUDA") # Install Cuda
Pkg.add("BenchmarkTools") # Install benchmark tools
# Pkg.add("BenchmarkPlots")
# Pkg.add("StatsPlots")
using Plots, BenchmarkTools
# using Images, FileIO
# # Use Cuda
# using CUDA, Test,  BenchmarkPlots, StatsPlots
const FPS = 15
# Visualize diffusion using 1D array
function diffusion_1D()
    # physics
    lx = 20.0 # domain length
    dc = 1.0 # diffusion coefficient
    #ρ = 20.0*10 # Large value leads to a long convergence time
    ρ = (lx/(dc*2π))^2 # Optimal value for convergence
    # numerics (Constants)
    nx   = 200 # the number of grid points
    nvis = 2 # frequency of updating the visualisation
    ϵtol = 1e-8
    maxiter = 20nx # Maximal iterations
    ncheck = ceil(Int, 0.1nx) # NUmber of checks
    # derived numerics
    dx = lx/nx      #grid spacing dx 
    dt = dx/sqrt(1/ρ) # dt = dx^2/dc/2  # diffusion
    nt = 5*nx# nt = nx^2 ÷ 5 # time 
    # creates a linear range of numbers, starting from dx/2, ending at lx-dx/2, and 200 numbers (nx)
    xc = LinRange(dx/2, lx-dx/2, nx)
    # array initialisation
    # C = @. 0.5cos(9π*xc/lx)+0.5 # concentration field C
    C = @. 1.0 + exp( -(xc-lx/4)^2 ) - xc/lx
    C_i = copy(C)
    qx = zeros(Float64, nx-1) # diffusive flux in the x direction qx
    println("Constant: nt = ", nt, " dc = ", dc, " dx = ", dx, " dt = ", dt)
    println("ρ = ", ρ)    
    # Go through each time step
    iter = 1; err = 2ϵtol
    iter_evo = Float64[]
    err_evo = Float64[]
    anim = @animate while err>= ϵtol && iter < maxiter
        # println("Before it = ", it, " C[1:5] = ", C[1:5])
        # diff: difference between C[ix+1] - C[ix]
        # qx = -dc .* diff(C)./dx  #diffusive flux dx =4.0
        qx .-= dt ./ (ρ * dc + dt) .* (qx .+ dc .* diff(C) ./dx )
        # println("it = ", it,  " qx[1:5] = ", qx[1:5])
        # flux balance equation
        C[2:end-1] .-= dt.* diff(qx) ./dx
        if iter % ncheck == 0
            err = maximum(abs.(diff(dc .* diff(C) ./dx)/dx)) # Get the maximal value 
            push!(iter_evo, iter)
            push!(err_evo, err)
            println("Complete the iteration: ", iter, " Error = ", round(err, sigdigits=2))
            # Plot the results
            p1 = plot(xc, [C_i, C]; 
                       xlim = (0, lx), ylim = (-0.1, 2.0), 
                       linewidth=:5.0, legend=:bottomright,
                       xlabel="lx", ylabel="Concentration", 
                       label="Concentration", title="Iteration = $(iter)")
            p2 = plot(iter_evo, err_evo; ylim = (-0.05, 1.0), xlim = (0, 1210),
                      xlabel="iter", ylabel="err",
                      label="error", title="Error = $(round(err, sigdigits=2))", 
                      markershape=:circle, markersize=2)
            plot!(p1, p2; layout=(2,1))
            # display(plot(p1, p2; layout=(2,1)))
        end
        iter += 1
    end 
    # println("err_evo ", err_evo[length(err_evo)-5:length(err_evo)])    
    # # Save to gif file
    path = "images/pde/diffusion_1D.gif"
    gif(anim, path, fps=50)
    println("Save gif result to $path")
end
println("Complete the simulation of diffusion ", @elapsed diffusion_1D())

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
    path = "images/pde/wave_propagation_1D.gif"
    gif(anim, path, fps=FPS)
    println("Save gif result to $path")
end
# wave_propagation()

# advection
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
    path = "images/pde/advection_1D.gif"
    gif(anim, path, fps=FPS)
    println("Save gif result to $path")
end
# advection()
