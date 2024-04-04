import Pkg
Pkg.add("Plots") # Install `Plots` package
Pkg.add("Images") 
Pkg.add("FileIO")

using Plots
using Images
using FileIO


# Reference: https://pde-on-gpu.vaw.ethz.ch/lecture1/#solving_partial_differential_equations_in_parallel_on_gpus

# # integrate dx/dt = lorenz(t,x) numerically for 10 steps
function lorenz(x)
    σ = 10
    β = 8/3
    ρ = 28
    [σ*(x[2]-x[1]),
     x[1]*(ρ-x[3]) - x[2],
     x[1]*x[2] - β*x[3]]
end


steps = 100
dt = 0.01
x = [2.0, 0.0, 0.0]
out = zeros(3, steps) # 3x500 arrays
out[:, 1] = x
println("size(out, 2) = ", size(out, 2))
# Animate the loren for 10 steps
for i = 2: size(out, 2)
    out[:, i] = out[:, i-1] + lorenz(out[:, i-1]) * dt 
end
plot(out[1,:], out[2,:], out[3,:])
path = "images/lorenz.png"
savefig(path)
print("Save to $path")
