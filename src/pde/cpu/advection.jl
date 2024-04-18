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
    path = "images/pde/cpu/advection_1D.gif"
    gif(anim, path, fps=15)
    println("Save gif result to $path")
end
println("Complete the simulation of advection ",  advection())