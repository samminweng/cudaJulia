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
    path = "images/pde/cpu/diffusion_1D.gif"
    gif(anim, path, fps=5)
    println("Save gif result to $path")
end
println("Complete the simulation of diffusion ", diffusion_1D())