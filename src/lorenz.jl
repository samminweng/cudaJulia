import Pkg
Pkg.add("Plots") # Install `Plots` package
Pkg.add("Images") 
Pkg.add("FileIO")

using Plots
using Images
using FileIO

x = 1:10; y = rand(10); # These are the plotting data
plot(x,y, label="my label")
#plot(heatmap(rand(10,10))) # Plot the heatmap

# # define the Lorenz attractor
# Base.@kwdef mutable struct Lorenz
#     dt::Float64 = 0.02
#     σ::Float64 = 10
#     ρ::Float64 = 28
#     β::Float64 = 8/3
#     x::Float64 = 1
#     y::Float64 = 1
#     z::Float64 = 1
# end

# # Solve lorenz system of ODEs:
# function step!(l::Lorenz)
#     dx = l.σ * (l.y - l.x)
#     dy = l.x * (l.ρ - l.z) - l.y
#     dz = l.x * l.y - l.β * l.z
#     l.x += l.dt * dx
#     l.y += l.dt * dy
#     l.z += l.dt * dz
# end
# # integrate dx/dt = lorenz(t,x) numerically for 10 steps
# steps = 10
# dt = 0.01
# x = [2.0, 0.0, 0.0]
# out = zeros(3, steps) # 3x500 arrays
# out[:, 1] = x
# print("size(out, 2) = ", size(out, 2))
# for i = 2: size(out, 2)
#     out[:, i] = out[:, i-1] + lorenz(out[:, i-1]) * dt 
#     #print("out = $out")
# end
# # build an animated gif by pushing new points to the plot, saving every 10th frame
# @gif for i=1:1500
#     step!(attractor)
#     push!(plt, attractor.x, attractor.y, attractor.z)
# end every 10

# plt = plot3d(1,
#             xlim = (-30, 30),
#             ylim = (-30, 30),
#             zlim = (0, 60),
#             title = "Lorenz Attractor",
#             legend = false,
#             marker = 2,
# )

# plt = plot(out[1,:], out[2,:], out[3,:])
