# # Install Cuda in Julia
# import Pkg
# Pkg.add("CUDA") # Install Cuda
# Pkg.add("BenchmarkTools") # Install benchmark tools
# Pkg.add("BenchmarkPlots")
# Pkg.add("StatsPlots")
using Plots
# using Images, FileIO
# # Use Cuda
# using CUDA, Test, BenchmarkTools, BenchmarkPlots, StatsPlots
const FPS = 5
# physics
lx = 20.0 # domain length
dc = 1.0 # diffusion coefficient 
# numerics
nx   = 200 # the number of grid points
nvis = 5 # frequency of updating the visualisation
# derived numerics
dx = lx/nx      #grid spacing dx 
dt = dx^2/dc/2  
nt = nx^2 ÷ 100 # time 
# creates a linear range of numbers, starting from dx/2, ending at lx-dx/2, and 200 numbers (nx)
xc = LinRange(dx/2, lx-dx/2, nx)
println("xc = ", xc)
# array initialisation
# @.: dot product operator
C = @. 0.5cos(9π*xc/lx)+0.5 # concentration field C
# print("diff(C) = ", diff(C), " size = ", size(diff(C)))
C_i = copy(C)
qx = zeros(Float64, nx-1) # diffusive flux in the x direction qx
println("Constant: ", " nt =", nt, " dc = ", dc, " dx = ", dx, " dt = ", dt)
# Go through each time step
anim = @animate for it=1:nt
    # println("Before it = ", it, " C[1:5] = ", C[1:5])
    # diff: difference between C[ix+1] - C[ix]
    # ./dx element-wise division by dx, .-dc: elementwise multiplication by -dc
    # .= in-place update
    qx .= .-dc .* diff(C)./dx  #diffusive flux dx =4.0
    # println("it = ", it,  " qx[1:5] = ", qx[1:5])
    # flux balance equation
    # C[2:end-1] .-= dt.* diff(qx) ./dx
    tmp = dt .* diff(qx) ./dx 
    C[2:end-1] = C[2:end-1] .- tmp   
    # println("After it = ", it, " C[1:5] = ", C[1:5])
    # println("----------------------------------------")
    # Plot the results
    plot(xc, C, label="Concentration", linewidth=:1.0,
         xlim = (0, 20), ylim = (-1, 1), legend=:bottomright,
         markershape=:circle, markersize=5, framestyle=:box)
    plot!(xc[1:end-1].+dx/2, qx, label="flux of concentration", linewidth=:1.0,
         xlim = (0, 20), ylim = (-1, 1), legend=:bottomright,
         markershape=:circle, markersize=5, framestyle=:box)
end every nvis
path = "images/pde.gif"
gif(anim, path, fps=FPS)
println("Save gif result to $path")

