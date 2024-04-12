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
const FPS = 15
# Visualize diffusion using 1D array
function heat_diffusion_1D()
    # physics
    lx = 20.0 # domain length
    dc = 1.0 # diffusion coefficient 
    # numerics
    nx   = 200*2 # the number of grid points
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
        plot(xlim = (0, lx), ylim = (-1.1, 1.1), 
            linewidth=:1.0, legend=:bottomright,
            xlabel="lx", ylabel="Concentration",
            )
        plot!(xc, C, label="Concentration", markershape=:circle, markersize=5, framestyle=:box)
        plot!(xc[1:end-1].+dx/2, qx, label="flux of concentration", markershape=:circle, markersize=5, framestyle=:box)
    end every nvis    
    # Save to gif file
    path = "images/pde/heat_diffusion_1D.gif"
    gif(anim, path, fps=FPS)
    println("Save gif result to $path")
end

# heat_diffusion_1D()

function acoustic_wave()
    # physics
    lx = 20.0 # domain length
    dc = 1.0  # coefficient 
    ρ, β = 1.0, 1.0
    # numerics
    nx   = 200 # the number of grid points
    nvis = 5 # frequency of updating the visualisation
    # derived numerics
    dx = lx/nx      #grid spacing dx 
    dt = dx^2/dc/2  # derivatives in time
    nt = nx^2 ÷ 100 # time 
    # creates a linear range of numbers
    xc = LinRange(dx/2, lx-dx/2, nx)
    println("size(xc) = ", size(xc))
    println("Constant: ", " nt =", nt, " dc = ", dc, " dx = ", dx, " dt = ", dt)

    # array initialisation
    Pr = @. exp(-(xc - lx/4)^2) #pressure
    Vx = zeros(Float64, nx-1)
    # Go through each time step
    anim = @animate for it=1:nt
        # diffusion physics
        Vx .= Vx - dt/ρ .* diff(Pr) ./dx  #velocity update
        # println("Vx = ", Vx)
        Pr[2:end-1] .= Pr[2:end-1] - dt/β .* diff(Vx) ./dx
        # println("Pr = ", Pr)
        # Plot the results
        plot(xlim = (0, lx), ylim = (-1.1, 1.1), 
            linewidth=:1.0, legend=:bottomright,
            xlabel="lx", ylabel="pressure")
        plot!(xc, Pr, label="pressure", markershape=:circle, markersize=5, framestyle=:box)
        plot!(xc[1:end-1].+dx/2, Vx, label="velocity update", markershape=:circle, markersize=5, framestyle=:box)
    end
    # # Save to gif file
    path = "images/pde/wave_diffusion_1D.gif"
    gif(anim, path, fps=FPS)
    println("Save gif result to $path")
end
# acoustic_wave()


# advection
function advection()
    # physics
    lx = 20.0 # domain length
    vx = 1.0  # coefficient 
    # numerics
    nx = 200  # the number of grid points
    dx = lx/nx  # grid spacing dx 
    # derived numerics
    dt = dx/abs(vx)
    nt = nx # time 
    dx = lx/nx  # grid spacing dx 
    println("Constant: ", "dx = ", dx, " dt = ", dt, " nt = ", nt, " nt÷2 = ", nt÷2)
    # creates a linear range of numbers
    xc = LinRange(dx/2, lx-dx/2, nx)
    # array initialisation
    C = @. exp(-(xc - lx/4)^2)
    C_i  = copy(C)
    println("size(C)", size(C))
    anim = @animate for it=1:nt
        C[2:end] .-= dt.*max(vx, 0.0).*diff(C)./dx # if vx >0
        C[1:end-1] .-= dt.*min(vx, 0.0).*diff(C)./dx # if vx <0
        if (it % (nt÷2)) == 0 # ÷ integer divide
            vx = -vx
        end
        # println("title", round(it*dt, digits=1))
        #println("it = ", it, " C = ", C[1:5])
        plot(xlim = (0, lx), ylim = (-0.1, 1.1), 
            linewidth=:1.0, legend=:bottomright,
            xlabel="lx", ylabel="advection")
        plot!(xc,  [C_i, C], label="advection",  title="Time = $(round(it*dt, digits=1))",
             markershape=:circle, markersize=5, framestyle=:box)
    end
    # # Save to gif file
    path = "images/pde/advection_1D.gif"
    gif(anim, path, fps=FPS)
    println("Save gif result to $path")
end

advection()