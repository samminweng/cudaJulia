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
    path = "images/pde/cpu/wave_propagation_1D.gif"
    gif(anim, path, fps=15)
    println("Save gif result to $path")
end
println("Complete the simulation of wave_propagation ",  wave_propagation())