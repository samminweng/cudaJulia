import Pkg
Pkg.add("CUDA") # Install Cuda
using CUDA #Use CUDA.jl library
using Test, BenchmarkTools
println(CUDA.versioninfo())
threads = (4, 3) # Specify the number of threads in a block
blocks  = (2, 2) # Specify the number of blocks in a grid
nx, ny  = threads[1]*blocks[1], threads[2]*blocks[2]
println("nx = $(nx), ny = $(ny)")
 



function copy!(A, B)
    # Compute thread ID (ix, iy)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    A[ix,iy] = B[ix,iy] # Copy arrays
    return
end

function test_copy!()
    A = CUDA.zeros(Float64, nx, ny) # Create array A on device memory
    B = CUDA.rand(Float64, nx, ny)     
    @cuda blocks=blocks threads=threads copy!(A, B)
    synchronize() # wait until finish
    h_A = round.(Array(A), sigdigits=5)  # Copy A to host memory
    h_B = round.(Array(B), sigdigits=5)
    println("A[1:5, 1] = $(h_A[1:5, 1])")
    println("B[1:5, 1] = $(h_B[1:5, 1])")
    @testset "Memory Copy Tests" begin
        @test h_A[1, 5] == h_B[1, 5]
        @test h_A[4, 3] == h_B[4, 3]
        @test h_A[8, 6] == h_B[8, 6]
        @test all(h_A .== h_B)
    end
end
test_copy!() 

function bench_gpu_copy()
    A = CUDA.zeros(Float64, nx, ny) # Create array A on device memory
    B = CUDA.rand(Float64, nx, ny) 
    CUDA.@sync begin
        @cuda copy!(A, B)
    end
end
println("Benchmark GPU copy ", @benchmark bench_gpu_copy())

function bench_cpu_copy()
    A = zeros(Float64, nx, ny) # Create array A on device memory
    B = rand(Float64, nx, ny) 
    A .= B
end
println("Benchmark CPU copy ", @benchmark bench_cpu_copy())



